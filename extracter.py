import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck


def _validate_depth_scaling(depth_scaling):
    if depth_scaling is None:
        return None
    if isinstance(depth_scaling, (list, tuple)) and len(depth_scaling) == 2:
        return float(depth_scaling[0]), float(depth_scaling[1])
    if isinstance(depth_scaling, dict):
        start = depth_scaling.get('start')
        end = depth_scaling.get('end')
        if start is not None and end is not None:
            return float(start), float(end)
    raise ValueError("depth_scaling must be None, a (start, end) tuple, or dict with 'start' and 'end'")


class ActivationExtractor:
    """ResNet의 multi-layer activation을 추출 (GPU)"""

    def __init__(
        self,
        model,
        feature_dim=128,
        spatial_size=14,
        normalize_steps=True,
        normalization_eps=1e-6,
        depth_scaling=None,
    ):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.normalize_steps = normalize_steps
        self.normalization_eps = normalization_eps
        self.depth_scaling_range = _validate_depth_scaling(depth_scaling)
        self.depth_scalars = None

        # Ensure BN/Dropout stay frozen
        self.model.eval()

        self.layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, Bottleneck):
                layer_idx = len(self.layer_names)
                hook = module.register_forward_hook(self._get_activation(name, layer_idx))
                self.hooks.append(hook)
                self.layer_names.append(name)

        self._init_depth_scalars()

    def _init_depth_scalars(self):
        if self.depth_scaling_range is None or len(self.layer_names) == 0:
            self.depth_scalars = None
            return

        start, end = self.depth_scaling_range
        if len(self.layer_names) == 1:
            scalars = torch.tensor([start], dtype=torch.float32)
        else:
            scalars = torch.linspace(start, end, steps=len(self.layer_names), dtype=torch.float32)
        self.depth_scalars = scalars

    def _resize_activation(self, tensor):
        _, _, h, w = tensor.shape
        if h == self.spatial_size and w == self.spatial_size:
            return tensor

        if h > self.spatial_size or w > self.spatial_size:
            return F.adaptive_avg_pool2d(tensor, (self.spatial_size, self.spatial_size))

        return F.interpolate(
            tensor,
            size=(self.spatial_size, self.spatial_size),
            mode='bilinear',
            align_corners=False,
        )

    def _channel_pool(self, tensor):
        B, C, H, W = tensor.shape
        if C > self.feature_dim:
            group_size = C // self.feature_dim
            remainder = C % self.feature_dim

            if remainder == 0:
                pooled_reshaped = tensor.view(B, self.feature_dim, group_size, H, W)
            else:
                tensor = tensor[:, :self.feature_dim * group_size, :, :]
                pooled_reshaped = tensor.view(B, self.feature_dim, group_size, H, W)

            return pooled_reshaped.mean(dim=2)

        return tensor

    def _normalize_step(self, tensor):
        mean = tensor.mean(dim=(1, 2, 3), keepdim=True)
        std = tensor.std(dim=(1, 2, 3), keepdim=True)
        return (tensor - mean) / (std + self.normalization_eps)

    def _apply_depth_scaling(self, tensor, layer_idx):
        if self.depth_scalars is None:
            return tensor
        scale = self.depth_scalars[layer_idx].to(tensor.device)
        return tensor * scale

    def _get_activation(self, name, layer_idx):
        def hook(module, inputs, output):
            resized = self._resize_activation(output)
            pooled = self._channel_pool(resized)
            if self.normalize_steps:
                pooled = self._normalize_step(pooled)
            pooled = self._apply_depth_scaling(pooled, layer_idx)

            self.activations[name] = pooled

        return hook

    def __call__(self, x):
        """GPU에서 activation 추출"""
        self.activations = {}
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return [self.activations[name] for name in self.layer_names]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

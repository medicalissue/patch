import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.convnext import CNBlock
from torchvision.models.efficientnet import FusedMBConv, MBConv
from torchvision.models.mobilenetv3 import InvertedResidual
from torchvision.models.resnet import BasicBlock, Bottleneck


def _validate_depth_scaling(depth_scaling):
    if depth_scaling is None:
        return None
    if isinstance(depth_scaling, (list, tuple)) and len(depth_scaling) == 2:
        return float(depth_scaling[0]), float(depth_scaling[1])
    if isinstance(depth_scaling, dict):
        start = depth_scaling.get("start")
        end = depth_scaling.get("end")
        if start is not None and end is not None:
            return float(start), float(end)
    raise ValueError("depth_scaling must be None, a (start, end) tuple, or dict with start/end")


class ActivationExtractor:
    """Collect intermediate activations from CNN or ViT backbones."""

    def __init__(
        self,
        model,
        feature_dim: int = 128,
        spatial_size: int = 14,
        normalize_steps: bool = True,
        normalization_eps: float = 1e-6,
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
        self._summary_logged = False

        self.model.eval()

        target_blocks = (Bottleneck, BasicBlock, CNBlock, InvertedResidual, MBConv, FusedMBConv)

        self.layer_names = []
        class_name = model.__class__.__name__.lower()
        for name, module in model.named_modules():
            if isinstance(module, target_blocks):
                layer_idx = len(self.layer_names)
                hook = module.register_forward_hook(self._get_activation(name, layer_idx))
                self.hooks.append(hook)
                self.layer_names.append(name)

        if hasattr(model, "encoder") and any(token in class_name for token in ("vit", "visiontransformer", "deit")):
            for name, module in model.encoder.named_modules():
                module_cls = module.__class__.__name__
                if "EncoderBlock" in module_cls or "TransformerEncoderLayer" in module_cls:
                    layer_idx = len(self.layer_names)
                    hook = module.register_forward_hook(
                        self._get_activation(f"encoder.{name}", layer_idx, channels_last=True)
                    )
                    self.hooks.append(hook)
                    self.layer_names.append(f"encoder.{name}")
        elif "swin" in class_name:
            for name, module in model.named_modules():
                module_cls = module.__class__.__name__
                if "SwinTransformerBlock" in module_cls or "SwinTransformerBlockV2" in module_cls:
                    layer_idx = len(self.layer_names)
                    hook = module.register_forward_hook(self._get_activation(name, layer_idx, channels_last=True))
                    self.hooks.append(hook)
                    self.layer_names.append(name)

        if not self.layer_names:
            print(f"⚠ Warning: no supported blocks found in {model.__class__.__name__}")

        self._init_depth_scalars()

    def _init_depth_scalars(self):
        if self.depth_scaling_range is None or not self.layer_names:
            self.depth_scalars = None
            return

        start, end = self.depth_scaling_range
        if len(self.layer_names) == 1:
            scalars = torch.tensor([start], dtype=torch.float32)
        else:
            scalars = torch.linspace(start, end, steps=len(self.layer_names), dtype=torch.float32)
        self.depth_scalars = scalars

    def _resize_activation(self, tensor: Tensor) -> Tensor:
        _, _, h, w = tensor.shape
        if h == self.spatial_size and w == self.spatial_size:
            return tensor

        if h > self.spatial_size or w > self.spatial_size:
            return F.adaptive_avg_pool2d(tensor, (self.spatial_size, self.spatial_size))

        return F.interpolate(tensor, size=(self.spatial_size, self.spatial_size), mode="bilinear", align_corners=False)

    def _channel_pool(self, tensor: Tensor) -> Tensor:
        b, c, h, w = tensor.shape
        if c > self.feature_dim:
            group_size = c // self.feature_dim
            remainder = c % self.feature_dim
            if remainder != 0:
                tensor = tensor[:, : self.feature_dim * group_size, :, :]
            pooled_reshaped = tensor.view(b, self.feature_dim, group_size, h, w)
            return pooled_reshaped.mean(dim=2)

        return tensor

    def _normalize_step(self, tensor: Tensor) -> Tensor:
        mean = tensor.mean(dim=(1, 2, 3), keepdim=True)
        std = tensor.std(dim=(1, 2, 3), keepdim=True)
        return (tensor - mean) / (std + self.normalization_eps)

    def _apply_depth_scaling(self, tensor: Tensor, layer_idx: int) -> Tensor:
        if self.depth_scalars is None:
            return tensor
        scale = self.depth_scalars[layer_idx].to(tensor.device)
        return tensor * scale

    def _get_activation(self, name, layer_idx, channels_last: bool = False):
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]

            if not isinstance(output, torch.Tensor):
                return

            if channels_last:
                if output.dim() == 4:
                    # [B, H, W, C] -> [B, C, H, W]
                    output = output.permute(0, 3, 1, 2).contiguous()
                elif output.dim() == 3:
                    # [B, N, C] -> square spatial map if possible
                    b, n, c = output.shape
                    spatial = int(round(n ** 0.5))
                    if spatial * spatial == n:
                        output = output.transpose(1, 2).reshape(b, c, spatial, spatial)
                    elif (n - 1) > 0 and int(round((n - 1) ** 0.5)) ** 2 == (n - 1):
                        spatial = int(round((n - 1) ** 0.5))
                        output = output[:, 1:, :].transpose(1, 2).reshape(b, c, spatial, spatial)
                    else:
                        # Unable to reshape into spatial map; skip this activation
                        return

            resized = self._resize_activation(output)
            pooled = self._channel_pool(resized)
            if self.normalize_steps:
                pooled = self._normalize_step(pooled)
            pooled = self._apply_depth_scaling(pooled, layer_idx)
            self.activations[name] = pooled

        return hook

    def __call__(self, x: Tensor):
        """Return activations as a list ordered by depth."""
        self.activations = {}
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)

        ordered = [self.activations[name] for name in self.layer_names if name in self.activations]
        if not self._summary_logged:
            print(f"  Extracted {len(ordered)} activation tensors")
            self._summary_logged = True
        return ordered

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

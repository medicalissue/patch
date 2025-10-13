import torch
import torch.nn as nn
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


class LearnableAttentionPool(nn.Module):
    """Learnable attention pooling for channel reduction via learned weighted sum."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Learnable attention weights: [out_channels, in_channels]
        # Use Xavier uniform initialization for stability
        self.attention_weights = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.xavier_uniform_(self.attention_weights, gain=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply learnable weighted sum across input channels.
        Args:
            x: [B, C, H, W] where C = in_channels
        Returns:
            [B, out_channels, H, W]
        """
        b, c, h, w = x.shape

        # Normalize attention weights via softmax over input channels
        # Shape: [out_channels, in_channels]
        attn = F.softmax(self.attention_weights, dim=1)

        # Reshape for matrix multiplication
        # x: [B, C, H, W] -> [B, C, H*W]
        x_reshaped = x.view(b, c, h * w)

        # Apply attention: [out_channels, in_channels] @ [B, in_channels, H*W]
        # -> [B, out_channels, H*W]
        pooled = torch.einsum('oi,bip->bop', attn, x_reshaped)

        # Reshape back to spatial: [B, out_channels, H*W] -> [B, out_channels, H, W]
        pooled = pooled.view(b, self.out_channels, h, w)

        return pooled


class ActivationExtractor(nn.Module):
    """Collect intermediate activations from CNN or ViT backbones."""

    def __init__(
        self,
        model,
        spatial_size: int = 14,
        normalize_steps: bool = True,
        normalization_eps: float = 1e-6,
        depth_scaling=None,
    ):
        super().__init__()
        self.model = model
        self.activations = {}
        self.hooks = []
        self.target_channels = None  # Will be auto-detected
        self.spatial_size = spatial_size
        self.normalize_steps = normalize_steps
        self.normalization_eps = normalization_eps
        self.depth_scaling_range = _validate_depth_scaling(depth_scaling)
        self.depth_scalars = None
        self._summary_logged = False
        self.channel_poolers = nn.ModuleDict()
        self.layer_channels = {}

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
        self._detect_and_init_channel_poolers()

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

    def _detect_and_init_channel_poolers(self):
        """Detect channel dimensions for each layer and initialize attention poolers."""
        # Temporary hook to collect channel dimensions
        temp_activations = {}

        def temp_hook(name):
            def hook(module, inputs, output):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    if output.dim() == 4:
                        temp_activations[name] = output.shape[1]  # C from [B, C, H, W]
                    elif output.dim() == 3:
                        temp_activations[name] = output.shape[2]  # C from [B, N, C]
            return hook

        # Register temporary hooks
        temp_hooks = []
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                h = module.register_forward_hook(temp_hook(name))
                temp_hooks.append(h)

        # Run a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            # Get the device of the model
            model_device = next(self.model.parameters()).device
            dummy_input = dummy_input.to(model_device)
            self.model(dummy_input)

        # Remove temporary hooks
        for h in temp_hooks:
            h.remove()

        # Store channel info
        for name in self.layer_names:
            if name in temp_activations:
                self.layer_channels[name] = temp_activations[name]

        # Find minimum channel count and use it as target
        if self.layer_channels:
            self.target_channels = min(self.layer_channels.values())

            print(f"  Auto-detected channel dimensions: min={self.target_channels}, using as target")

            # Initialize attention poolers for each layer that needs reduction
            for name, in_channels in self.layer_channels.items():
                if in_channels > self.target_channels:
                    safe_name = name.replace(".", "_")
                    self.channel_poolers[safe_name] = LearnableAttentionPool(in_channels, self.target_channels)

    def _resize_activation(self, tensor: Tensor) -> Tensor:
        _, _, h, w = tensor.shape
        if h == self.spatial_size and w == self.spatial_size:
            return tensor

        if h > self.spatial_size or w > self.spatial_size:
            return F.adaptive_avg_pool2d(tensor, (self.spatial_size, self.spatial_size))

        return F.interpolate(tensor, size=(self.spatial_size, self.spatial_size), mode="bilinear", align_corners=False)

    def _channel_pool(self, tensor: Tensor, layer_name: str) -> Tensor:
        b, c, h, w = tensor.shape
        if self.target_channels and c > self.target_channels:
            safe_name = layer_name.replace(".", "_")
            if safe_name in self.channel_poolers:
                # Use learnable attention pooling
                return self.channel_poolers[safe_name](tensor)
            else:
                # Fallback to average pooling if pooler not found
                group_size = c // self.target_channels
                remainder = c % self.target_channels
                if remainder != 0:
                    tensor = tensor[:, : self.target_channels * group_size, :, :]
                pooled_reshaped = tensor.view(b, self.target_channels, group_size, h, w)
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
            # Normalize before channel pooling for stability
            if self.normalize_steps:
                resized = self._normalize_step(resized)
            pooled = self._channel_pool(resized, name)
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

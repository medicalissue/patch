import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRALayer", "LoRAConv1d", "apply_lora_to_model"]


class LoRALayer(nn.Module):
    """Low-rank adaptation for linear layers."""

    def __init__(self, in_features, out_features, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        return original_output + self.scaling * lora_output


class LoRAConv1d(nn.Module):
    """Low-rank adaptation for 1D convolutional layers."""

    def __init__(self, module: nn.Conv1d, rank: int = 8, alpha: int = 16):
        super().__init__()
        if module.groups != 1:
            raise ValueError("LoRAConv1d currently supports groups=1 only")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups

        effective_in = self.in_channels * self.kernel_size[0]

        self.lora_A = nn.Parameter(torch.zeros(rank, effective_in))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, original_output: torch.Tensor) -> torch.Tensor:
        delta_weight = torch.matmul(self.lora_B, self.lora_A)
        delta_weight = delta_weight.view(self.out_channels, self.in_channels, self.kernel_size[0])
        delta = F.conv1d(
            x,
            delta_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return original_output + self.scaling * delta


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Sequence[str] = ("Linear",),
    target_name_keywords: Optional[Sequence[str]] = None,
):
    """
    Attach LoRA adapters to selected modules.

    Args:
        model: module to adapt
        rank: LoRA rank
        alpha: LoRA scaling
        target_modules: module types to adapt (e.g. ["Linear", "Conv1d"])
        target_name_keywords: optional substrings to filter module names

    Returns:
        Tuple[nn.Module, List[nn.Parameter]]: adapted model and trainable LoRA params
    """
    lora_params: List[nn.Parameter] = []
    target_name_keywords = [kw.lower() for kw in target_name_keywords or []]

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "Linear" in target_modules:
            if target_name_keywords and not any(keyword in name.lower() for keyword in target_name_keywords):
                continue

            weight_device = module.weight.device if module.weight is not None else next(module.parameters()).device
            lora = LoRALayer(module.in_features, module.out_features, rank, alpha).to(weight_device)
            setattr(module, "lora", lora)

            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

            lora_params.extend(lora.parameters())

            original_forward = module.forward

            def forward_with_lora(x, original_forward=original_forward, lora=lora):
                original_output = original_forward(x)
                return lora(x, original_output)

            module.forward = forward_with_lora

        elif isinstance(module, nn.Conv1d) and "Conv1d" in target_modules:
            if target_name_keywords and not any(keyword in name.lower() for keyword in target_name_keywords):
                continue

            weight_device = module.weight.device if module.weight is not None else next(module.parameters()).device
            lora = LoRAConv1d(module, rank, alpha).to(weight_device)
            setattr(module, "lora", lora)

            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

            lora_params.extend(lora.parameters())

            original_forward = module.forward

            def forward_with_lora_conv(x, original_forward=original_forward, lora=lora):
                original_output = original_forward(x)
                return lora(x, original_output)

            module.forward = forward_with_lora_conv

    print(f"  ✓ Applied LoRA: {len(lora_params)} trainable parameter tensors")
    return model, lora_params

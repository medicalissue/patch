import torch
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

class ActivationExtractor:
    """ResNet의 multi-layer activation을 추출 (GPU)"""
    def __init__(self, model, feature_dim=128, spatial_size=14):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        
        # # ResNet의 주요 layer에 hook 등록
        # self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        # for name in self.layer_names:
        #     layer = getattr(model, name)
        #     hook = layer.register_forward_hook(self._get_activation(name))
        #     self.hooks.append(hook)
            
        self.layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, Bottleneck):
                hook = module.register_forward_hook(self._get_activation(name))
                self.hooks.append(hook)
                self.layer_names.append(name)
    
    def _get_activation(self, name):
        def hook(module, input, output):
            pooled = F.adaptive_avg_pool2d(output, (self.spatial_size, self.spatial_size))
            B, C, H, W = pooled.shape
            
            if C > self.feature_dim:
                group_size = C // self.feature_dim
                remainder = C % self.feature_dim
                
                if remainder == 0:
                    pooled_reshaped = pooled.view(B, self.feature_dim, group_size, H, W)
                else:
                    pooled = pooled[:, :self.feature_dim * group_size, :, :]
                    pooled_reshaped = pooled.view(B, self.feature_dim, group_size, H, W)
                
                pooled_final = pooled_reshaped.mean(dim=2)
            else:
                pooled_final = pooled
            
            self.activations[name] = pooled_final
        return hook
    
    def __call__(self, x):
        """GPU에서 activation 추출"""
        self.activations = {}
        with torch.no_grad():
            _ = self.model(x)
        return [self.activations[name] for name in self.layer_names]
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
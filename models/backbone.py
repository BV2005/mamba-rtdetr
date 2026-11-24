import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """
    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for multi-scale feature extraction in RT-DETR.
    Extracts features from C3, C4, C5 (strides 8, 16, 32).
    """
    def __init__(self, name='resnet50', pretrained=True, freeze_at=1, freeze_bn=True):
        super().__init__()
        
        # Determine norm layer
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        
        # Load torchvision ResNet (supports both new weights= and legacy pretrained= API)
        resnet_factory = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
        }
        if name not in resnet_factory:
            raise ValueError(f"Unsupported backbone: {name}. Choose from {list(resnet_factory.keys())}")
        
        try:
            # New torchvision API (>= 0.13)
            weights = 'DEFAULT' if pretrained else None
            backbone = resnet_factory[name](weights=weights, norm_layer=norm_layer)
        except TypeError:
            # Legacy torchvision API
            backbone = resnet_factory[name](pretrained=pretrained, norm_layer=norm_layer)

        # Extract multi-scale features: layer2 (C3), layer3 (C4), layer4 (C5)
        return_layers = {"layer2": "C3", "layer3": "C4", "layer4": "C5"}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # Freeze early layers
        if freeze_at >= 1:
            for p in self.body.conv1.parameters(): p.requires_grad = False
            for p in self.body.bn1.parameters(): p.requires_grad = False
        if freeze_at >= 2:
            for p in self.body.layer1.parameters(): p.requires_grad = False
        if freeze_at >= 3:
            for p in self.body.layer2.parameters(): p.requires_grad = False
        if freeze_at >= 4:
            for p in self.body.layer3.parameters(): p.requires_grad = False
        if freeze_at >= 5:
            for p in self.body.layer4.parameters(): p.requires_grad = False

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """
        Forward pass for ResNetBackbone.
        Returns OrderedDict with keys 'C3', 'C4', 'C5'.
        Shapes for ResNet-50:
            C3 = (B, 512, H/8, W/8)
            C4 = (B, 1024, H/16, W/16)
            C5 = (B, 2048, H/32, W/32)
        """
        return self.body(x)

def get_backbone_channels(name='resnet50') -> List[int]:
    """Helper to get backbone channel dimensions for C3, C4, C5."""
    if name in ['resnet18', 'resnet34']:
        return [128, 256, 512]
    elif name in ['resnet50', 'resnet101']:
        return [512, 1024, 2048]
    else:
        raise ValueError(f"Unsupported backbone: {name}")

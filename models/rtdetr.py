import torch
import torch.nn as nn

class MambaRTDETR(nn.Module):
    """
    Mamba RT-DETR full model assembly.
    """
    def __init__(self, backbone, encoder, decoder, num_classes=80):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg) -> 'MambaRTDETR':
        # Placeholder for building from config
        # Expects cfg to have backbone, encoder, decoder configs
        raise NotImplementedError("Build from config logic here")

    def forward(self, images, targets=None):
        """
        images: NestedTensor or (B, 3, H, W)
        targets: list of dicts with 'labels' and 'boxes' (cxcywh, normalized)
        Returns:
            Training: dict with 'pred_logits', 'pred_boxes', 'aux_outputs'
            Inference: same structure
        """
        # 1. Backbone
        # Assume backbone returns multi-scale features. The encoder only
        # consumes C4/C5 (P3 is dropped — see HybridEncoder), but the
        # backbone may still produce C3 as a side effect of computing C4/C5
        # (e.g. ResNet's layer2 feeds layer3), so we tolerate either a 2- or
        # 3-tuple/dict here and just ignore C3 if present.
        backbone_features = self.backbone(images)
        
        if isinstance(backbone_features, list) or isinstance(backbone_features, tuple):
            from collections import OrderedDict
            if len(backbone_features) >= 3:
                features_dict = OrderedDict({
                    'C4': backbone_features[-2],
                    'C5': backbone_features[-1],
                })
            else:
                features_dict = OrderedDict({
                    'C4': backbone_features[0],
                    'C5': backbone_features[1],
                })
        else:
            features_dict = backbone_features
            
        # 2. Encoder
        encoder_features = self.encoder(features_dict)
        
        # 3. Decoder
        output = self.decoder(encoder_features, targets)
        
        return output

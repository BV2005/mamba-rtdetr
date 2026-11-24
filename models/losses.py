import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class RTDETRLoss(nn.Module):
    """
    Detection losses for RT-DETR.
    """
    def __init__(self, num_classes=80, matcher=None, 
                 weight_class=2.0, weight_bbox=5.0, weight_giou=2.0,
                 alpha=0.75, gamma=2.0, dn_loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.alpha = alpha
        self.gamma = gamma
        # Scales the whole DN-stream loss dict (loss_class_dn, loss_bbox_dn,
        # loss_giou_dn, and their aux variants). Kept at 1.0 by default to
        # match standard DN-DETR/DINO practice; expose as a knob in case the
        # DN stream ends up dominating the main stream's gradients.
        self.dn_loss_weight = dn_loss_weight

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Focal Loss)
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # One hot encoding for target classes
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        
        # Focal Loss
        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target_classes_onehot + (1 - self.alpha) * (1 - target_classes_onehot)
            loss = alpha_t * loss
            
        # Sum over queries/classes, then normalize by the number of target
        # boxes. Averaging over the query dimension would reduce every
        # classification gradient by num_queries (300 by default).
        loss = loss.sum() / num_boxes
        return {'loss_class': loss * self.weight_class}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Bounding box losses (L1 and GIoU)
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes
        
        return {
            'loss_bbox': loss_bbox * self.weight_bbox,
            'loss_giou': loss_giou * self.weight_giou
        }

    def loss_labels_iou_aware(self, outputs, targets, indices, num_boxes):
        """
        IoU-aware classification loss for the encoder-level auxiliary output
        (RT-DETR's actual 'IoU-aware query selection'). Identical to
        loss_labels, except the positive target for each matched (pred, gt)
        pair is that pair's real box IoU (detached), not a hard 1. This is
        what makes the query-selection score correlate with localization
        quality instead of classification confidence alone.
        """
        src_logits = outputs['pred_logits']
        src_boxes = outputs['pred_boxes']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        with torch.no_grad():
            matched_src_boxes = src_boxes[idx]
            matched_tgt_boxes = torch.cat([t['boxes'][J] for t, (_, J) in zip(targets, indices)], dim=0)
            ious = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(matched_src_boxes),
                box_cxcywh_to_xyxy(matched_tgt_boxes)
            )).clamp(min=0.0, max=1.0)  # GIoU can be negative; clamp for a valid soft target

        soft_target = target_classes_onehot.clone()
        soft_target[idx[0], idx[1], target_classes_o] = ious.type_as(soft_target)

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, soft_target, reduction="none")
        p_t = prob * soft_target + (1.0 - prob) * (1.0 - soft_target)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * soft_target + (1.0 - self.alpha) * (1.0 - soft_target)
            loss = alpha_t * loss
        loss = loss.sum() / num_boxes
        return {'loss_class_enc': loss * self.weight_class}

    def loss_dn(self, outputs, dn_meta, dn_num_boxes):
        """
        Denoising-stream loss. Correspondence between each dn query and its
        source GT is known by construction (see decoder._prepare_dn), so
        this is a masked classification + box loss over the valid dn slots
        -- no Hungarian matching needed.
        """
        src_logits = outputs['pred_logits']
        src_boxes = outputs['pred_boxes']
        target_labels = dn_meta['target_labels']
        target_boxes = dn_meta['target_boxes']
        valid_mask = dn_meta['valid_mask']

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        focal = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target_classes_onehot + (1 - self.alpha) * (1 - target_classes_onehot)
            focal = alpha_t * focal
        # Zero out padded/invalid dn slots entirely (no gradient at all from
        # them, not even a "push toward background" one) rather than trying
        # to give them a meaningful target.
        focal = focal * valid_mask.unsqueeze(-1)
        loss_class = focal.sum() / dn_num_boxes

        if valid_mask.any():
            loss_bbox = F.l1_loss(src_boxes[valid_mask], target_boxes[valid_mask], reduction='none').sum() / dn_num_boxes
            loss_giou = (1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes[valid_mask]),
                box_cxcywh_to_xyxy(target_boxes[valid_mask])
            ))).sum() / dn_num_boxes
        else:
            loss_bbox = src_boxes.sum() * 0.0
            loss_giou = src_boxes.sum() * 0.0

        return {
            'loss_class_dn': loss_class * self.weight_class,
            'loss_bbox_dn': loss_bbox * self.weight_bbox,
            'loss_giou_dn': loss_giou * self.weight_giou,
        }

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        """
        Returns: dict of losses {'loss_class': ..., 'loss_bbox': ..., 'loss_giou': ...}
        Also compute auxiliary losses from intermediate decoder layers
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve matching
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute number of boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        losses = {}
        losses.update(self.loss_labels(outputs_without_aux, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs_without_aux, targets, indices, num_boxes))
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                l_dict = self.loss_labels(aux_outputs, targets, indices, num_boxes)
                l_dict.update(self.loss_boxes(aux_outputs, targets, indices, num_boxes))
                losses.update({f"{k}_{i}": v for k, v in l_dict.items()})

        # IoU-aware query-selection loss: matched separately over ALL
        # encoder positions (not just the selected top-k), so the selection
        # head learns to favor well-localized boxes, not just confidently
        # classified ones.
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            enc_indices = self.matcher(enc_outputs, targets)
            enc_losses = self.loss_labels_iou_aware(enc_outputs, targets, enc_indices, num_boxes)
            enc_box_losses = self.loss_boxes(enc_outputs, targets, enc_indices, num_boxes)
            enc_losses.update({f"{k}_enc": v for k, v in enc_box_losses.items()})
            losses.update(enc_losses)

        # Denoising-stream loss: direct correspondence, no matcher needed.
        if 'dn_outputs' in outputs:
            dn_meta = outputs['dn_meta']
            dn_num_boxes = dn_meta['valid_mask'].sum().clamp(min=1).item()
            dn_losses = self.loss_dn(outputs['dn_outputs'], dn_meta, dn_num_boxes)

            if 'dn_aux_outputs' in outputs:
                for i, dn_aux in enumerate(outputs['dn_aux_outputs']):
                    l_dict = self.loss_dn(dn_aux, dn_meta, dn_num_boxes)
                    dn_losses.update({f"{k}_{i}": v for k, v in l_dict.items()})

            if self.dn_loss_weight != 1.0:
                dn_losses = {k: v * self.dn_loss_weight for k, v in dn_losses.items()}
            losses.update(dn_losses)

        return losses

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and targets.
    """
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.75, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # Must match RTDETRLoss's alpha/gamma -- the matcher's cost function
        # and the actual training loss need to agree on what "good" means,
        # otherwise Hungarian assignment optimizes for a different focal
        # curve than what the loss then penalizes.
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict with 'pred_logits' (B,Q,C) and 'pred_boxes' (B,Q,4)
        targets: list of dicts with 'labels' (N,) and 'boxes' (N,4)
        Returns: list of (src_indices, tgt_indices) tuples
        """
        B, Q, C = outputs["pred_logits"].shape
        
        # Flatten batch
        # NOTE: cast to fp32 before sigmoid/log. Under autocast, pred_logits/
        # pred_boxes are fp16; fp16's smallest representable value (~6e-8) is
        # larger than the 1e-8 epsilon below, so it underflows to exactly 0
        # and log(0) = -inf, which poisons the Hungarian cost matrix and
        # makes linear_sum_assignment raise "cost matrix is infeasible".
        out_prob = outputs["pred_logits"].flatten(0, 1).float().sigmoid() # (B*Q, C)
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float() # (B*Q, 4)
        
        # Concatenate targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).float()
        
        # Handle empty targets
        if len(tgt_ids) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(B)]
            
        # Compute classification cost (Focal loss style)
        alpha = self.alpha
        gamma = self.gamma
        
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        
        # Compute L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute GIoU cost
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Total cost matrix
        C_matrix = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C_matrix = C_matrix.view(B, Q, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C_matrix.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

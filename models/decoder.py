import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def inverse_sigmoid(x, eps=1e-5):
    """Inverse of torch.sigmoid, clamped for numerical stability. Used to
    accumulate box refinements across decoder layers in logit space (so a
    delta of 0 is a true no-op, unlike accumulating in [0,1] probability
    space)."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class MSDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention (Pure PyTorch implementation)
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=2, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        
        # Initialize sampling offsets for points
        import math
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        """
        query: (B, Q, C)
        reference_points: (B, Q, L, 2)
        value: (B, S, C)
        spatial_shapes: (L, 2)
        level_start_index: (L,)
        """
        B, Q, C = query.shape
        _, S, _ = value.shape
        
        value = self.value_proj(value)
        value = value.view(B, S, self.n_heads, C // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(B, Q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(B, Q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(B, Q, self.n_heads, self.n_levels, self.n_points)
        
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        
        sampling_locations = reference_points[:, :, None, :, None, :] + \
                             sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                             
        output = torch.zeros((B, Q, self.n_heads, C // self.n_heads), device=query.device, dtype=query.dtype)
        
        for l in range(self.n_levels):
            h, w = spatial_shapes[l]
            val_l = value[:, level_start_index[l]:level_start_index[l]+h*w, :, :].view(B, h, w, self.n_heads, C // self.n_heads)
            val_l = val_l.permute(0, 3, 4, 1, 2).contiguous() # (B, n_heads, C//n_heads, h, w)
            val_l = val_l.view(B * self.n_heads, C // self.n_heads, h, w)
            
            samp_loc_l = sampling_locations[:, :, :, l, :, :] # (B, Q, n_heads, n_points, 2)
            samp_loc_l = samp_loc_l.permute(0, 2, 1, 3, 4).contiguous().view(B * self.n_heads, Q, self.n_points, 2)
            samp_loc_l = samp_loc_l * 2 - 1.0 # convert to [-1, 1] for grid_sample
            
            sampled_val = F.grid_sample(val_l, samp_loc_l, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled_val = sampled_val.view(B, self.n_heads, C // self.n_heads, Q, self.n_points)
            sampled_val = sampled_val.permute(0, 3, 1, 4, 2) # (B, Q, n_heads, n_points, C//n_heads)
            
            attn_weight_l = attention_weights[:, :, :, l, :].unsqueeze(-1) # (B, Q, n_heads, n_points, 1)
            
            output += (sampled_val * attn_weight_l).sum(dim=3)
            
        output = output.view(B, Q, C)
        output = self.output_proj(output)
        return output

class RTDETRDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.0, n_levels=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformableAttention(d_model, n_heads, n_levels=n_levels, n_points=4)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        # Self Attention (Pre-Norm)
        q = self.norm1(query)
        q2, _ = self.self_attn(q, q, q)
        query = query + q2
        
        # Cross Attention (Pre-Norm)
        q = self.norm2(query)
        q2 = self.cross_attn(q, reference_points, value, spatial_shapes, level_start_index)
        query = query + q2
        
        # FFN (Pre-Norm)
        q = self.norm3(query)
        q2 = self.ffn(q)
        query = query + q2
        
        return query

class RTDETRDecoder(nn.Module):
    """
    RT-DETR Decoder.
    """
    def __init__(self, hidden_dim=256, num_queries=300, num_classes=80,
                 num_decoder_layers=6, num_heads=8, dim_feedforward=1024, dropout=0.0,
                 num_levels=3, use_denoising=True, dn_groups=5, dn_number=200,
                 label_noise_ratio=0.2, box_noise_scale=0.4, use_enc_aux_loss=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Query selection components
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_score_head = nn.Linear(hidden_dim, 1)

        # Per-class encoder-level head, applied to EVERY encoder position
        # (not just the top-k selected ones). Used only to build the
        # 'enc_outputs' auxiliary target for the IoU-aware query-selection
        # loss in losses.py -- it plays no role in inference/top-k selection
        # itself (enc_score_head still does that, unchanged).
        self.use_enc_aux_loss = use_enc_aux_loss
        if use_enc_aux_loss:
            self.enc_class_head = nn.Linear(hidden_dim, num_classes)

        # Denoising training (simplified single-group-type DN, box+label
        # noise only, no contrastive positive/negative groups like full
        # DN-DETR). Only active during training when targets are provided;
        # completely inert at inference.
        self.use_denoising = use_denoising
        if use_denoising:
            self.dn_groups = dn_groups
            self.dn_number = dn_number
            self.label_noise_ratio = label_noise_ratio
            self.box_noise_scale = box_noise_scale
            # +1 slot for the padding/no-object label used in empty dn slots
            self.dn_label_embed = nn.Embedding(num_classes + 1, hidden_dim)
            self.dn_bbox_embed = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Decoder layers
        # num_levels must match the number of feature-map levels the encoder
        # returns to forward() (3 for the P3/P4/P5 HybridEncoder).
        self.layers = nn.ModuleList([
            RTDETRDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout, n_levels=num_levels)
            for _ in range(num_decoder_layers)
        ])
        
        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        # box_head now predicts a REFINEMENT DELTA (added in inverse-sigmoid
        # space to the previous layer's box), not an absolute box. Init to
        # near-zero output so the first refinement step starts as a no-op
        # rather than immediately blowing up the initial box estimate.
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)
        )
        nn.init.constant_(self.box_head[-1].weight, 0.)
        nn.init.constant_(self.box_head[-1].bias, 0.)

        # Predicts the INITIAL box (before any decoder layer runs) from the
        # selected encoder query features. Without this, decoder layers had
        # nothing to refine and independently re-guessed an absolute box
        # from scratch every layer -- the bug this change fixes.
        self.enc_bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)
        )

        # v2 fix: focal-loss prior (as in RT-DETR / Deformable-DETR). Every class
        # logit starts at p = 0.01 instead of PyTorch's default ~0.5. With the
        # default, every background score of every query (and of every encoder
        # position, for enc_class_head) is penalised at once: the 2026-07-29 run
        # started at a loss of ~12,200, 99 % of it classification, and spent most
        # of epoch 0 recovering (test_v2_fixes.py checks this).
        prior_bias = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.class_head.bias, prior_bias)
        nn.init.constant_(self.enc_score_head.bias, prior_bias)
        if use_enc_aux_loss:
            nn.init.constant_(self.enc_class_head.bias, prior_bias)

    def _prepare_dn(self, targets, device):
        """
        Build denoising queries + their known (label, box) targets from
        ground truth. Unlike the main query stream, correspondence here is
        known by construction (no Hungarian matching needed at loss time),
        which gives the decoder a much more direct supervision signal early
        in training -- this is what lets RT-DETR-style models converge in
        tens of epochs instead of DETR's original hundreds.
        """
        num_gt_per_img = [len(t['labels']) for t in targets]
        max_gt = max(num_gt_per_img) if num_gt_per_img else 0
        if max_gt == 0:
            return None

        B = len(targets)
        groups = max(1, min(self.dn_groups, self.dn_number // max_gt))
        max_dn = max_gt * groups

        # Padding label = self.num_classes (the "no object" slot in the
        # +1-sized embedding table); overwritten below for real entries.
        dn_labels = torch.full((B, max_dn), self.num_classes, dtype=torch.long, device=device)
        dn_boxes = torch.zeros((B, max_dn, 4), device=device)
        target_labels = torch.full((B, max_dn), self.num_classes, dtype=torch.long, device=device)
        target_boxes = torch.zeros((B, max_dn, 4), device=device)
        valid_mask = torch.zeros((B, max_dn), dtype=torch.bool, device=device)

        for b, t in enumerate(targets):
            n = len(t['labels'])
            if n == 0:
                continue
            labels = t['labels']
            boxes = t['boxes']

            for g in range(groups):
                sl = slice(g * max_gt, g * max_gt + n)

                noisy_labels = labels.clone()
                if self.label_noise_ratio > 0:
                    flip_mask = torch.rand(n, device=device) < self.label_noise_ratio
                    random_labels = torch.randint(0, self.num_classes, (n,), device=device)
                    noisy_labels = torch.where(flip_mask, random_labels, noisy_labels)

                noisy_boxes = boxes.clone()
                if self.box_noise_scale > 0:
                    diff = torch.zeros_like(boxes)
                    diff[:, :2] = boxes[:, 2:] / 2 * self.box_noise_scale
                    diff[:, 2:] = boxes[:, 2:] * self.box_noise_scale
                    rand_sign = torch.rand_like(boxes) * 2 - 1
                    noisy_boxes = (boxes + rand_sign * diff).clamp(min=0.0, max=1.0)
                    noisy_boxes[:, 2:] = noisy_boxes[:, 2:].clamp(min=1e-3)

                dn_labels[b, sl] = noisy_labels
                dn_boxes[b, sl] = noisy_boxes
                target_labels[b, sl] = labels
                target_boxes[b, sl] = boxes
                valid_mask[b, sl] = True

        dn_queries = self.dn_label_embed(dn_labels) + self.dn_bbox_embed(dn_boxes)

        return {
            'queries': dn_queries,
            'init_boxes': dn_boxes,
            'ref_points_xy': dn_boxes[..., :2],
            'target_labels': target_labels,
            'target_boxes': target_boxes,
            'valid_mask': valid_mask,
        }

    def forward(self, features, targets=None):
        """
        features: list of [O3, O4, O5] from encoder
        Returns: dict with 'pred_logits' (B, num_queries, num_classes) and 'pred_boxes' (B, num_queries, 4) in cxcywh format normalized [0,1]
        Also returns 'aux_outputs' (intermediate main-stream layers),
        'enc_outputs' (all-position encoder predictions, for the IoU-aware
        auxiliary loss), and -- during training, when targets are given --
        'dn_outputs'/'dn_aux_outputs'/'dn_meta' (denoising stream).
        """
        B = features[0].shape[0]
        
        spatial_shapes = []
        flattened_features = []
        
        # Flatten multi-scale features
        for feat in features:
            _, _, H, W = feat.shape
            spatial_shapes.append((H, W))
            flattened_features.append(feat.flatten(2).transpose(1, 2))
            
        spatial_shapes = torch.tensor(spatial_shapes, device=features[0].device)
        level_start_index = torch.cat([spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]])
        value = torch.cat(flattened_features, dim=1) # (B, sum(H*W), C)
        
        # Query selection (top-k by enc_score_head, unchanged mechanism)
        enc_output = self.enc_output_norm(value)
        enc_scores = self.enc_score_head(enc_output).squeeze(-1) # (B, sum(H*W))

        # Per-class logits at EVERY position, only used for the IoU-aware
        # auxiliary loss (losses.py matches this against GT separately from
        # the top-k selected queries) -- plays no role in selection itself.
        if self.use_enc_aux_loss:
            enc_class_logits_all = self.enc_class_head(enc_output) # (B, S, num_classes)
        
        topk_scores, topk_indices = torch.topk(enc_scores, self.num_queries, dim=1)
        
        # Gather selected features
        batch_indices = torch.arange(B, device=value.device).unsqueeze(-1).expand(-1, self.num_queries)
        queries = enc_output[batch_indices, topk_indices] # (B, num_queries, C)

        # NOTE: torch.topk's returned indices carry no gradient, so without this
        # enc_score_head's weight/bias never receive a gradient -> DDP's
        # reducer hangs waiting for a bucket that will never be reduced
        # ("Expected to have finished reduction in the prior iteration...").
        # Gate the selected queries by their own selection score so
        # enc_score_head sits back in the autograd graph.
        queries = queries * torch.sigmoid(topk_scores).unsqueeze(-1)
        
        # Construct reference points based on selected indices
        grids = []
        for (H, W) in spatial_shapes:
            y, x = torch.meshgrid(
                torch.arange(H, device=value.device, dtype=torch.float32) + 0.5,
                torch.arange(W, device=value.device, dtype=torch.float32) + 0.5,
                indexing='ij'
            )
            grid = torch.stack((x / W, y / H), dim=-1) # (H, W, 2)
            grids.append(grid.view(H * W, 2))
        
        concat_grids = torch.cat(grids, dim=0).unsqueeze(0).expand(B, -1, -1) # (B, sum(H*W), 2)
        ref_points_xy = concat_grids[batch_indices, topk_indices] # (B, num_queries, 2)

        # Box prediction for ALL positions (not just the selected top-k) --
        # needed for the IoU-aware auxiliary loss below. The main stream's
        # initial box is just this same result gathered at topk_indices, so
        # there's no duplicate enc_bbox_head call for the selected subset.
        enc_bbox_delta_all = self.enc_bbox_head(enc_output) # (B, S, 4)
        enc_pred_boxes_all = torch.cat([
            inverse_sigmoid(concat_grids) + enc_bbox_delta_all[..., :2],
            enc_bbox_delta_all[..., 2:]
        ], dim=-1).sigmoid() # (B, S, 4), normalized [0,1] cxcywh

        pred_boxes = enc_pred_boxes_all[batch_indices, topk_indices] # (B, num_queries, 4)

        # Grid coordinates are already normalized to [0, 1].  Applying a
        # sigmoid here would remap them to roughly [0.50, 0.73], collapsing
        # deformable-attention reference points into one part of the image.
        reference_points = ref_points_xy.unsqueeze(2).expand(-1, -1, len(spatial_shapes), -1) # (B, num_queries, L, 2)
        
        aux_outputs = []

        # Denoising stream: built from ground truth, run through the SAME
        # decoder layers (shared weights) as a second, independent forward
        # pass per layer. Only active during training with targets given;
        # completely absent from the graph at inference.
        dn_data = None
        dn_queries = dn_pred_boxes = dn_reference_points = None
        dn_aux_outputs = []
        if self.training and self.use_denoising and targets is not None:
            dn_data = self._prepare_dn(targets, value.device)
            if dn_data is not None:
                dn_queries = dn_data['queries']
                dn_pred_boxes = dn_data['init_boxes']
                dn_reference_points = dn_data['ref_points_xy'].unsqueeze(2).expand(-1, -1, len(spatial_shapes), -1)
        
        # Iterative refinement: each layer's cross-attention samples around
        # the PREVIOUS layer's predicted box center, and each layer's
        # box_head predicts a small correction (delta) to the previous
        # layer's box rather than guessing a fresh absolute box. This is
        # the mechanism that lets RT-DETR-style models localize accurately
        # within a handful of layers instead of needing hundreds of epochs
        # like plain DETR (which lacks this feedback loop).
        for layer in self.layers:
            queries = layer(queries, reference_points, value, spatial_shapes, level_start_index)
            
            # Predict for this layer
            pred_logits = self.class_head(queries)
            box_delta = self.box_head(queries)
            pred_boxes = (inverse_sigmoid(pred_boxes) + box_delta).sigmoid()
            
            aux_outputs.append({
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes
            })

            # Feed the refined box center forward as the next layer's
            # reference point. Detached: gradients still flow through
            # pred_boxes via the loss on this layer's aux_outputs, but not
            # backward through the *sampling locations* of later layers --
            # this is standard practice (Deformable DETR / RT-DETR) since
            # letting reference-point gradients flow through grid_sample's
            # sampling coordinates across layers is unstable in fp16/AMP.
            reference_points = pred_boxes[..., :2].detach().unsqueeze(2).expand(-1, -1, len(spatial_shapes), -1)

            if dn_data is not None:
                dn_queries = layer(dn_queries, dn_reference_points, value, spatial_shapes, level_start_index)
                dn_logits = self.class_head(dn_queries)
                dn_box_delta = self.box_head(dn_queries)
                dn_pred_boxes = (inverse_sigmoid(dn_pred_boxes) + dn_box_delta).sigmoid()
                dn_aux_outputs.append({'pred_logits': dn_logits, 'pred_boxes': dn_pred_boxes})
                dn_reference_points = dn_pred_boxes[..., :2].detach().unsqueeze(2).expand(-1, -1, len(spatial_shapes), -1)
            
        final_output = aux_outputs.pop()
        
        if self.training:
            final_output['aux_outputs'] = aux_outputs

            if self.use_enc_aux_loss:
                final_output['enc_outputs'] = {
                    'pred_logits': enc_class_logits_all,
                    'pred_boxes': enc_pred_boxes_all,
                }

            if dn_data is not None:
                dn_final = dn_aux_outputs.pop()
                final_output['dn_outputs'] = dn_final
                final_output['dn_aux_outputs'] = dn_aux_outputs
                final_output['dn_meta'] = {
                    'target_labels': dn_data['target_labels'],
                    'target_boxes': dn_data['target_boxes'],
                    'valid_mask': dn_data['valid_mask'],
                }
            
        return final_output

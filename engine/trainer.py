import torch
import torch.nn as nn
from tqdm import tqdm
import json
import math
import sys
import copy
import os
import glob
import time

from .prefetcher import CUDAPrefetcher
from utils.box_ops import box_cxcywh_to_xyxy

# COCOeval wants the original 91-category ids back (the dataset/criterion
# work in the contiguous 0-79 space), so build the inverse of
# data.coco_dataset.coco91_to_coco80 once at import time.
from data.coco_dataset import coco91_to_coco80
_COCO80_TO_COCO91 = {v: k for k, v in coco91_to_coco80.items()}

class Trainer:
    def __init__(self, model, criterion, optimizer, lr_scheduler, 
                 train_loader, val_loader, device, logger, cfg,
                 train_sampler=None, rank=0, world_size=1, main_process=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.cfg = cfg

        # DistributedSampler needs set_epoch() called each epoch so it
        # reshuffles differently per epoch (otherwise every process sees
        # the exact same shard order every epoch). None when not distributed.
        self.train_sampler = train_sampler
        self.rank = rank
        self.world_size = world_size
        self.main_process = main_process
        self.is_distributed = world_size > 1
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_amp)
        self.use_amp = cfg.training.use_amp
        self.clip_max_norm = cfg.training.clip_max_norm
        
        self.use_ema = cfg.training.use_ema
        if self.use_ema:
            # Deep-copy the *unwrapped* model, not the DDP-wrapped one — DDP
            # holds C++ process-group/reducer state that isn't safely
            # deepcopy-able and isn't something the EMA shadow needs anyway.
            self.ema_model = copy.deepcopy(self._raw(model)).to(device)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_decay = cfg.training.ema_decay
            # v2: RT-DETR's decay warm-up, decay * (1 - exp(-updates / ema_warmups)): the shadow follows the model
            # closely at first instead of staying near the random initial weights for ~10k steps
            self.ema_warmups = getattr(cfg.training, 'ema_warmups', 2000)
            self.ema_updates = 0
        else:
            self.ema_model = None

        # Overlaps next-batch host->device transfer with current-batch compute
        # so the GPU stays fed even when the DataLoader workers (CPU-bound
        # decode/augment) can't quite keep up — see engine/prefetcher.py.
        self.train_prefetcher = CUDAPrefetcher(self.train_loader, self.device)

    def _raw(self, m):
        """Unwrap nn.DataParallel/DistributedDataParallel to get the
        underlying module for state_dict save/load and EMA — checkpoints
        should always be architecture-only, never tied to a particular
        multi-GPU wrapper."""
        return m.module if hasattr(m, 'module') else m

    def train_one_epoch(self, epoch):
        self.model.train()
        if hasattr(self, 'criterion') and self.criterion is not None:
            self.criterion.train()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        metrics = {}
        total_loss = 0
        total_main_loss = 0
        total_dn_loss = 0
        accum_steps = max(1, getattr(self.cfg.training, 'accum_steps', 1))

        # --- Linear LR warmup ---
        warmup_epochs = getattr(self.cfg.training, 'warmup_epochs', 0)
        warmup_active = warmup_epochs > 0 and epoch < warmup_epochs
        if warmup_active:
            total_warmup_steps = warmup_epochs * len(self.train_loader)
            global_warmup_step = epoch * len(self.train_loader)  # base for this epoch
            base_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
        
        # Only rank 0 renders a progress bar — every process printing its
        # own bar to the same terminal is unreadable noise.
        iterable = tqdm(self.train_prefetcher, total=len(self.train_loader), desc=f"Epoch {epoch}") \
            if self.main_process else self.train_prefetcher
        pbar = iterable
        self.optimizer.zero_grad()
        
        DEBUG_TIMING = os.environ.get('DEBUG_TIMING', '0') == '1'
        DEBUG_STEPS = 15

        if DEBUG_TIMING:
            t_prev_end = time.time()

        for step, (samples, targets) in enumerate(pbar):
            # Apply linear warmup: ramp LR from 0.1% to 100% of target
            if warmup_active:
                progress = (global_warmup_step + step + 1) / total_warmup_steps
                warmup_factor = 0.001 + 0.999 * progress  # 0.001 -> 1.0
                for pg, base_lr in zip(self.optimizer.param_groups, base_lrs):
                    pg['lr'] = base_lr * warmup_factor

            if DEBUG_TIMING and step < DEBUG_STEPS:
                torch.cuda.synchronize(); t_data = time.time()
                t_wait = t_data - t_prev_end  # time spent blocked waiting for this batch

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # targets is required here (not just at loss time) so the
                # decoder can build denoising queries from ground truth --
                # without this, DN silently never activates even though the
                # decoder/loss code paths support it.
                outputs = self.model(samples, targets)
                if DEBUG_TIMING and step < DEBUG_STEPS:
                    torch.cuda.synchronize(); t_fwd = time.time()

                loss_dict = self.criterion(outputs, targets) if self.criterion else {'loss': torch.tensor(0.0, requires_grad=True, device=self.device)}
                
                # Sum all losses (they are already weighted by the criterion)
                loss = sum(loss_dict.values())

                # Diagnostic split: DN-stream keys (loss_*_dn, loss_*_dn_N)
                # get their own subtotal, everything else (main matched
                # stream + its aux layers + the iou-aware enc loss) gets the
                # other. Loss is dropping doesn't tell you the main stream is
                # learning if DN dominates the sum -- these subtotals do.
                dn_loss = sum(v for k, v in loss_dict.items() if '_dn' in k)
                main_loss = loss - dn_loss
                if DEBUG_TIMING and step < DEBUG_STEPS:
                    torch.cuda.synchronize(); t_loss = time.time()
                
                if not math.isfinite(loss.item()):
                    self.logger.info(f"Loss is {loss.item()}, stopping training")
                    self.logger.info(f"Loss breakdown: {{{', '.join(f'{k}: {v.item():.4f}' for k, v in loss_dict.items())}}}")
                    sys.exit(1)
            
            # Scale down so accumulated gradients equal the effective (unaccumulated) loss
            self.scaler.scale(loss / accum_steps).backward()

            if DEBUG_TIMING and step < DEBUG_STEPS:
                torch.cuda.synchronize(); t_bwd = time.time()
            
            is_accum_boundary = ((step + 1) % accum_steps == 0) or (step + 1 == len(self.train_loader))
            if is_accum_boundary:
                if self.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.use_ema:
                    self.update_ema()

            if DEBUG_TIMING and step < DEBUG_STEPS:
                # NOTE: measured AFTER the accum-boundary block (grad clip /
                # optimizer step / EMA), not right after backward — the old
                # version stamped t_prev_end before that block ran, so its
                # cost silently leaked into the NEXT step's data_wait
                # instead of being attributed here. accum_opt_ema is 0 on
                # non-boundary steps and the real cost on boundary steps.
                torch.cuda.synchronize(); t_opt = time.time()
                print(f"[step {step}] shape={tuple(samples.shape)} "
                      f"data_wait={t_wait:.2f}s fwd={t_fwd - t_data:.2f}s loss={t_loss - t_fwd:.2f}s "
                      f"bwd={t_bwd - t_loss:.2f}s accum_opt_ema={t_opt - t_bwd:.2f}s "
                      f"total(excl_wait)={t_opt - t_data:.2f}s "
                      f"total(incl_wait)={t_opt - t_prev_end:.2f}s", flush=True)
                t_prev_end = t_opt
                
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_dn_loss += dn_loss.item() if torch.is_tensor(dn_loss) else dn_loss
            
            if self.main_process and step % self.cfg.training.log_interval == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'main': main_loss.item(),
                    'dn': dn_loss.item() if torch.is_tensor(dn_loss) else dn_loss,
                })
                gstep = epoch * len(self.train_loader) + step
                self.logger.log_scalar('train/loss', loss.item(), gstep)
                self.logger.log_scalar('train/loss_main', main_loss.item(), gstep)
                self.logger.log_scalar('train/loss_dn', dn_loss.item() if torch.is_tensor(dn_loss) else dn_loss, gstep)
                
        n_steps = len(self.train_loader) if len(self.train_loader) > 0 else 1
        return {
            'loss': total_loss / n_steps,
            'loss_main': total_main_loss / n_steps,
            'loss_dn': total_dn_loss / n_steps,
        }
    
    def update_ema(self):
        with torch.no_grad():
            self.ema_updates += 1
            decay = self.ema_decay * (1.0 - math.exp(-self.ema_updates / self.ema_warmups))
            raw_model = self._raw(self.model)
            # NOTE: build the EMA param lookup ONCE per call, not per
            # parameter. The old code called self.ema_model.state_dict()
            # TWICE INSIDE the per-parameter loop -- state_dict() rebuilds
            # an OrderedDict over the whole model every call, so that was
            # O(P^2) dict reconstruction across ~34M params / 200+ tensors,
            # entirely synchronous. In-place mul_/add_ also avoids
            # allocating a fresh tensor per parameter (the old
            # decay*a + (1-decay)*b expression created a new tensor before
            # copy_ even ran).
            ema_params = dict(self.ema_model.named_parameters())
            for name, param in raw_model.named_parameters():
                if name in ema_params:
                    if param.requires_grad:
                        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)
                    else:                          # frozen parameters: keep identical to the model
                        ema_params[name].copy_(param.data)
            # v2: buffers (BatchNorm running statistics etc.) were never copied, so the EMA model evaluated with its
            # initial statistics; copy them (as RT-DETR's ModelEMA does for non-float buffers, and averaging float
            # running stats the same way as the weights)
            ema_buffers = dict(self.ema_model.named_buffers())
            for name, buf in raw_model.named_buffers():
                if name in ema_buffers:
                    if buf.dtype.is_floating_point:
                        ema_buffers[name].mul_(decay).add_(buf, alpha=1.0 - decay)
                    else:
                        ema_buffers[name].copy_(buf)

    def evaluate(self, epoch):
        # Only rank 0 evaluates. This is still safe now that this actually
        # runs inference (not just the old stub): a bare forward pass on
        # self.model (even DDP-wrapped) doesn't issue any collective ops —
        # only backward's gradient sync does — so rank 0 running this alone
        # while other ranks do nothing can't deadlock. The is_distributed
        # barrier in train() after evaluate() is what keeps ranks in
        # lockstep before the next epoch starts.
        if not self.main_process:
            return {}

        self.model.eval()
        if self.use_ema:
            eval_model = self.ema_model
        else:
            eval_model = self.model
        eval_model.eval()

        self.logger.info("Running evaluation...")

        coco_gt = self.val_loader.dataset.coco
        results = []

        with torch.no_grad():
            for samples, targets in self.val_loader:
                samples = samples.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = eval_model(samples)

                # NOTE: matcher.py casts to fp32 before sigmoid for the same
                # underflow reason (see apply_fix_v2.py) — do the same here
                # so low-confidence scores don't get zeroed out under fp16.
                pred_logits = outputs['pred_logits'].float()
                pred_boxes = outputs['pred_boxes'].float()
                scores = pred_logits.sigmoid()

                B, Q, C = scores.shape
                for b in range(B):
                    img_id = int(targets[b]['image_id'].item())
                    orig_h, orig_w = targets[b]['orig_size'].tolist()

                    # Multi-label sigmoid scores (not softmax across classes),
                    # so the top-k candidates live in the flattened Q*C space,
                    # not just top-k queries with an argmax class each.
                    scores_b = scores[b].reshape(-1)
                    k = min(100, scores_b.numel())
                    topk_scores, topk_idx = torch.topk(scores_b, k)
                    topk_boxes_idx = topk_idx // C
                    topk_labels = topk_idx % C

                    boxes_b = pred_boxes[b][topk_boxes_idx]
                    boxes_xyxy = box_cxcywh_to_xyxy(boxes_b)
                    # Boxes are normalized [0, 1] relative to the whole image
                    # regardless of the fixed-size resize (see
                    # data/transforms.py RandomResize), so scaling directly
                    # by orig_w/orig_h maps them back to original pixels.
                    boxes_xyxy[:, 0].clamp_(0, 1)
                    boxes_xyxy[:, 1].clamp_(0, 1)
                    boxes_xyxy[:, 2].clamp_(0, 1)
                    boxes_xyxy[:, 3].clamp_(0, 1)
                    boxes_xyxy[:, 0] *= orig_w
                    boxes_xyxy[:, 2] *= orig_w
                    boxes_xyxy[:, 1] *= orig_h
                    boxes_xyxy[:, 3] *= orig_h

                    for score, label, box in zip(topk_scores.tolist(), topk_labels.tolist(), boxes_xyxy.tolist()):
                        x0, y0, x1, y1 = box
                        if x1 <= x0 or y1 <= y0:
                            continue
                        results.append({
                            'image_id': img_id,
                            # COCOeval needs the original 91-id category
                            # space, not the model's contiguous 0-79 labels.
                            'category_id': _COCO80_TO_COCO91[label],
                            'bbox': [x0, y0, x1 - x0, y1 - y0],
                            'score': score,
                        })

        if len(results) == 0:
            self.logger.info("Evaluation produced no predictions above threshold")
            metrics = {
                'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0,
                'mAP_S': 0.0, 'mAP_M': 0.0, 'mAP_L': 0.0,
            }
            self.logger.log_metrics(metrics, epoch, prefix='val')
            return metrics

        # Lazy import: only rank 0 (main_process) ever reaches this, and it
        # keeps pycocotools off the hot import path for pure-training runs.
        from pycocotools.cocoeval import COCOeval

        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # stats indices per pycocotools convention:
        # [0]=AP@[.5:.95] [1]=AP@.5 [2]=AP@.75
        # [3]=AP-small(area<32^2) [4]=AP-medium(32^2-96^2) [5]=AP-large(>96^2)
        # [6:9]=AR@1/10/100, [9:12]=AR-small/medium/large (not surfaced here)
        metrics = {
            'mAP': float(coco_eval.stats[0]),
            'mAP_50': float(coco_eval.stats[1]),
            'mAP_75': float(coco_eval.stats[2]),
            'mAP_S': float(coco_eval.stats[3]),
            'mAP_M': float(coco_eval.stats[4]),
            'mAP_L': float(coco_eval.stats[5]),
        }

        self.logger.log_metrics(metrics, epoch, prefix='val')
        return metrics

    def benchmark(self, img_size=None, batch_size=1, num_warmup=10, num_iters=100):
        """One-off efficiency report: params, FLOPs, latency, FPS, GPU memory.

        Separate from evaluate() on purpose — these are static/throughput
        properties of the model+hardware, not per-epoch validation metrics,
        so they're measured once (e.g. after training, or standalone from
        eval.py) rather than logged every epoch.
        """
        if not self.main_process:
            return {}

        eval_model = self.ema_model if self.use_ema else self.model
        raw_model = self._raw(eval_model)
        raw_model.eval()

        if img_size is None:
            img_size = self.cfg.data.img_size

        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

        dummy = torch.randn(batch_size, 3, img_size, img_size, device=self.device)

        # FLOPs via torch's built-in operator-level counter (torch>=2.1).
        # Best-effort: some ops (custom Mamba CUDA kernels in particular)
        # may not have a flop formula registered, in which case this
        # undercounts rather than crashes — reported params/latency/FPS/
        # memory below are unaffected either way.
        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with torch.no_grad():
                flop_counter = FlopCounterMode(display=False)
                with flop_counter:
                    raw_model(dummy)
                flops = flop_counter.get_total_flops()
        except Exception as e:
            self.logger.info(f"FLOPs counting unavailable ({e}); reporting FLOPs_G=None")

        is_cuda = self.device.type == 'cuda'
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(num_warmup):
                raw_model(dummy)
            if is_cuda:
                torch.cuda.synchronize()

            start = time.time()
            for _ in range(num_iters):
                raw_model(dummy)
            if is_cuda:
                torch.cuda.synchronize()
            elapsed = time.time() - start

        latency_ms = (elapsed / num_iters) * 1000.0 / batch_size
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if is_cuda else None

        metrics = {
            'params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6,
            'FLOPs_G': (flops / 1e9) if flops is not None else None,
            'latency_ms': latency_ms,
            'FPS': fps,
            'gpu_mem_MB': gpu_mem_mb,
        }

        flops_str = "N/A" if metrics['FLOPs_G'] is None else f"{metrics['FLOPs_G']:.2f}G"
        gpu_mem_str = "N/A" if gpu_mem_mb is None else f"{gpu_mem_mb:.0f}MB"
        self.logger.info(
            f"Benchmark @ {img_size}x{img_size}, batch={batch_size}: "
            f"params={metrics['params_M']:.2f}M "
            f"FLOPs={flops_str} "
            f"latency={latency_ms:.2f}ms FPS={fps:.1f} "
            f"GPU_mem={gpu_mem_str}"
        )
        return metrics
    
    def train(self, start_epoch=0, max_epochs=72):
        # v2: on --resume, keep the best AP reached before the restart (it used to restart at 0, so the first epoch
        # after a restart overwrote checkpoint_best.pth even when it was worse)
        best_map = 0.0
        best_file = os.path.join(self.cfg.output_dir, 'best_metrics.json')
        if start_epoch > 0 and os.path.isfile(best_file):
            with open(best_file) as f:
                best_map = float(json.load(f).get('mAP', 0.0))
            self.logger.info(f"Resuming with best mAP so far {best_map:.4f} ({best_file})")
        training_start = time.time()

        for epoch in range(start_epoch, max_epochs):
            epoch_start = time.time()
            train_metrics = self.train_one_epoch(epoch)
            epoch_time_sec = time.time() - epoch_start
            
            if self.lr_scheduler:
                self.lr_scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_lr(lr, epoch)
            
            val_metrics = self.evaluate(epoch)

            if self.is_distributed:
                # Keep all processes in lockstep at epoch boundaries — rank 0
                # does eval/checkpoint save while the others would otherwise
                # race ahead into the next epoch's dataloader/backward calls.
                torch.distributed.barrier()

            if not self.main_process:
                continue

            elapsed_total_sec = time.time() - training_start
            self.logger.log_scalar('train/epoch_time_sec', epoch_time_sec, epoch)
            self.logger.info(
                f"Epoch {epoch} took {epoch_time_sec / 60:.1f} min "
                f"(elapsed {elapsed_total_sec / 3600:.2f} h total) -- "
                f"loss={train_metrics.get('loss', 0):.4f} "
                f"main={train_metrics.get('loss_main', 0):.4f} "
                f"dn={train_metrics.get('loss_dn', 0):.4f}"
            )

            if val_metrics.get('mAP', 0) >= best_map:
                best_map = val_metrics.get('mAP', 0)
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                
            if epoch % self.cfg.training.save_interval == 0 or epoch == max_epochs - 1:
                self.save_checkpoint(epoch, val_metrics)

        if self.main_process:
            total_training_time_sec = time.time() - training_start
            self.logger.info(
                f"Training complete: {total_training_time_sec / 3600:.2f} h total "
                f"over {max_epochs - start_epoch} epochs"
            )
            return {'total_training_time_sec': total_training_time_sec}
        return {}
                
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model': self._raw(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'metrics': metrics,
        }
        
        if self.use_ema:
            checkpoint['ema_model'] = self._raw(self.ema_model).state_dict()
            checkpoint['ema_updates'] = self.ema_updates
            
        save_dir = self.cfg.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

        # Rotation: keep only the N most recent periodic checkpoints so disk
        # doesn't fill up over a long run (this is what caused the epoch-68
        # crash previously -- checkpoint_epoch_*.pth accumulated unbounded).
        # checkpoint_best.pth is handled separately below and always kept.
        keep_last_n = getattr(self.cfg.training, 'keep_last_n_checkpoints', 3)
        if keep_last_n > 0:
            existing = sorted(
                glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth')),
                key=lambda p: int(os.path.basename(p).rsplit('_', 1)[-1].split('.')[0])
            )
            for old_ckpt in existing[:-keep_last_n]:
                os.remove(old_ckpt)
                self.logger.info(f"Removed old checkpoint {old_ckpt} (rotation, keeping last {keep_last_n})")
        
        if is_best:
            best_path = os.path.join(save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
            with open(os.path.join(save_dir, 'best_metrics.json'), 'w') as f:
                json.dump({'epoch': epoch, **{k: float(v) for k, v in (metrics or {}).items()
                                              if isinstance(v, (int, float))}}, f, indent=1)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._raw(self.model).load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.lr_scheduler and checkpoint['lr_scheduler']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
        if self.use_ema and 'ema_model' in checkpoint:
            self._raw(self.ema_model).load_state_dict(checkpoint['ema_model'])
            self.ema_updates = checkpoint.get('ema_updates', 0)
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']

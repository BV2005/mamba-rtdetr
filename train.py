"""
MambaFusion-DETR Training Entry Point.

Usage:
    python train.py --config configs/rtdetr_mamba_r50_cocominitrain_v2.yaml --seed 1
    python train.py --config configs/rtdetr_mamba_r50_cocominitrain_v2.yaml --resume outputs/<run>/checkpoint_epoch_N.pth
    python train.py --config configs/rtdetr_mamba_r50_cocominitrain_v2.yaml --eval --resume outputs/<run>/checkpoint_best.pth
    (full COCO / minitrain in one command: bash train_full_coco.sh)
"""
import os

# Must be set before torch is imported / CUDA context is created.
# Reduces OOM from allocator fragmentation, which matters here since the
# Mamba fusion blocks (applied on P3/P4/P5, 4-way cross-scan by default)
# still produce variable-sized intermediate tensors that fragment the
# allocator faster than plain conv/attention blocks.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import json
import yaml
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

# Add project root to sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbone import ResNetBackbone, get_backbone_channels
from models.hybrid_encoder import HybridEncoder
from models.decoder import RTDETRDecoder
from models.rtdetr import MambaRTDETR
from models.matcher import HungarianMatcher
from models.losses import RTDETRLoss
from data.coco_dataset import build_coco_dataloader
from data.transforms import build_transforms
from engine.trainer import Trainer
from utils.logger import TrainingLogger
from utils.kaggle_utils import is_kaggle, patch_config_for_kaggle
from utils.dist_utils import setup_distributed, cleanup_distributed, is_main_process, NullLogger


class DictAsMember(dict):
    """Utility to access dict keys as attributes, recursively."""
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def parse_args():
    parser = argparse.ArgumentParser(description='MambaFusion-DETR Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--seed', type=int, default=None,
                        help='v2: seed Python / NumPy / torch (DataLoader workers follow torch) and append _seed<N> '
                             'to output_dir, so several seeds of one config do not overwrite each other')
    parser.add_argument('--keep-last-n-checkpoints', type=int, default=None,
                         help='How many periodic checkpoint_epoch_*.pth files to keep '
                              '(checkpoint_best.pth is always kept separately). Default 3.')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config and return as DictAsMember."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return DictAsMember(cfg)


def build_model(cfg):
    """
    Build the full MambaRTDETR model from config.
    
    Components:
        1. ResNet backbone (pretrained)
        2. HybridEncoder with AIFI + Mamba CCFF
        3. RTDETRDecoder with deformable cross-attention
    """
    model_cfg = cfg.model
    backbone_cfg = model_cfg.backbone
    encoder_cfg = model_cfg.encoder
    decoder_cfg = model_cfg.decoder
    
    # 1. Backbone
    backbone = ResNetBackbone(
        name=backbone_cfg.name,
        pretrained=backbone_cfg.pretrained,
        freeze_at=backbone_cfg.freeze_at,
        freeze_bn=backbone_cfg.freeze_bn,
    )
    # C3/C4/C5 channels — the encoder now processes all three levels for
    # full FPN-style multi-scale fusion including high-res P3.
    in_channels = get_backbone_channels(backbone_cfg.name)
    
    # 2. Hybrid Encoder (AIFI + Mamba CCFF, P3/P4/P5)
    encoder = HybridEncoder(
        in_channels=in_channels,
        hidden_dim=encoder_cfg.hidden_dim,
        num_heads=encoder_cfg.num_heads,
        dim_feedforward=encoder_cfg.dim_feedforward,
        d_state=encoder_cfg.d_state,
        d_conv=encoder_cfg.d_conv,
        expand=encoder_cfg.expand,
        mlp_ratio=encoder_cfg.mlp_ratio,
        num_scans=encoder_cfg.num_scans,
        dropout=encoder_cfg.dropout,
        use_checkpoint=getattr(encoder_cfg, 'use_checkpoint', False),
        fusion=getattr(encoder_cfg, 'fusion', 'mamba'),
    )
    
    # 3. Decoder (num_levels must match the encoder's 3 output levels)
    decoder = RTDETRDecoder(
        hidden_dim=decoder_cfg.hidden_dim,
        num_queries=decoder_cfg.num_queries,
        num_classes=model_cfg.num_classes,
        num_decoder_layers=decoder_cfg.num_decoder_layers,
        num_heads=decoder_cfg.num_heads,
        dim_feedforward=decoder_cfg.dim_feedforward,
        dropout=decoder_cfg.dropout,
        num_levels=getattr(decoder_cfg, 'num_levels', 3),
        use_denoising=getattr(decoder_cfg, 'use_denoising', True),
        dn_groups=getattr(decoder_cfg, 'dn_groups', 5),
        dn_number=getattr(decoder_cfg, 'dn_number', 200),
        label_noise_ratio=getattr(decoder_cfg, 'label_noise_ratio', 0.2),
        box_noise_scale=getattr(decoder_cfg, 'box_noise_scale', 0.4),
        use_enc_aux_loss=getattr(decoder_cfg, 'use_enc_aux_loss', True),
    )
    
    # 4. Full Model
    model = MambaRTDETR(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        num_classes=model_cfg.num_classes,
    )
    
    return model


def build_criterion(cfg):
    """Build the loss function with Hungarian matcher."""
    loss_cfg = cfg.loss
    
    matcher = HungarianMatcher(
        cost_class=loss_cfg.cost_class,
        cost_bbox=loss_cfg.cost_bbox,
        cost_giou=loss_cfg.cost_giou,
        alpha=loss_cfg.focal_alpha,
        gamma=loss_cfg.focal_gamma,
    )
    
    criterion = RTDETRLoss(
        num_classes=cfg.model.num_classes,
        matcher=matcher,
        weight_class=loss_cfg.weight_class,
        weight_bbox=loss_cfg.weight_bbox,
        weight_giou=loss_cfg.weight_giou,
        alpha=loss_cfg.focal_alpha,
        gamma=loss_cfg.focal_gamma,
        dn_loss_weight=getattr(loss_cfg, 'dn_loss_weight', 1.0),
    )
    
    return criterion


def build_optimizer(model, cfg):
    """
    Build AdamW optimizer with separate param groups:
      1. Backbone params: lr_backbone (0.1× base)
      2. Mamba params: lr_mamba (0.1× base)
      3. Other params: base lr
    Weight decay only on non-bias, non-norm parameters.
    """
    lr = cfg.training.lr
    lr_backbone = cfg.training.lr_backbone
    lr_mamba = cfg.training.lr_mamba
    weight_decay = cfg.training.weight_decay
    
    # Separate parameters into groups
    backbone_params = []
    mamba_params = []
    other_params = []
    backbone_params_no_decay = []
    mamba_params_no_decay = []
    other_params_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_bias_or_norm = 'bias' in name or 'norm' in name or 'bn' in name
        
        if 'backbone' in name:
            if is_bias_or_norm:
                backbone_params_no_decay.append(param)
            else:
                backbone_params.append(param)
        elif 'mamba' in name or 'ssm' in name:
            if is_bias_or_norm:
                mamba_params_no_decay.append(param)
            else:
                mamba_params.append(param)
        else:
            if is_bias_or_norm:
                other_params_no_decay.append(param)
            else:
                other_params.append(param)
    
    param_dicts = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {"params": other_params_no_decay, "lr": lr, "weight_decay": 0.0},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_params_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": mamba_params, "lr": lr_mamba, "weight_decay": weight_decay},
        {"params": mamba_params_no_decay, "lr": lr_mamba, "weight_decay": 0.0},
    ]
    
    # Filter out empty groups
    param_dicts = [g for g in param_dicts if len(g["params"]) > 0]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    return optimizer


def build_scheduler(optimizer, cfg):
    """Build multi-step LR scheduler (x0.1 at each epoch in training.lr_drop; stepped once per epoch)."""
    lr_drop = cfg.training.lr_drop
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_drop, gamma=0.1)


def lr_drop_warning(cfg):
    """v2: the 2026-07-29 run had lr_drop [60] with 30 epochs, so its learning rate never decayed. Returns a
    warning string when no milestone falls inside the run, else None."""
    drops = list(cfg.training.lr_drop or [])
    if not any(0 < d < cfg.training.epochs for d in drops):
        return (f"lr_drop {drops} has no milestone inside the {cfg.training.epochs} epochs: the learning rate will "
                f"never decay (set e.g. lr_drop: [{int(cfg.training.epochs * 0.75)}])")
    return None


def build_dataloaders(cfg, is_distributed=False, rank=0, world_size=1):
    """Build training and validation dataloaders.

    Returns (train_loader, val_loader, train_sampler). val stays
    non-distributed/non-shuffled — evaluation only runs on rank 0 (see
    Trainer.evaluate), so it doesn't need sharding.
    """
    data_cfg = cfg.data
    
    train_transforms = build_transforms(is_train=True, img_size=data_cfg.img_size,
                                        strong_aug=getattr(data_cfg, 'strong_aug', True))
    val_transforms = build_transforms(is_train=False, img_size=data_cfg.img_size)
    
    train_loader, train_sampler = build_coco_dataloader(
        img_dir=data_cfg.train_img_dir,
        ann_file=data_cfg.train_ann_file,
        transforms=train_transforms,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=getattr(data_cfg, 'pin_memory', True),
        persistent_workers=getattr(data_cfg, 'persistent_workers', True),
        prefetch_factor=getattr(data_cfg, 'prefetch_factor', 4),
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )
    
    val_loader, _ = build_coco_dataloader(
        img_dir=data_cfg.val_img_dir,
        ann_file=data_cfg.val_ann_file,
        transforms=val_transforms,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=getattr(data_cfg, 'pin_memory', True),
        persistent_workers=getattr(data_cfg, 'persistent_workers', True),
        prefetch_factor=getattr(data_cfg, 'prefetch_factor', 4),
        is_distributed=False,
    )
    
    return train_loader, val_loader, train_sampler


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Apply CLI overrides
    if args.batch_size is not None:
        cfg['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['training']['lr'] = args.lr
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.keep_last_n_checkpoints is not None:
        cfg['training']['keep_last_n_checkpoints'] = args.keep_last_n_checkpoints
    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cfg['output_dir'] = f"{cfg['output_dir']}_seed{args.seed}"

    # Distributed setup. No-op (is_distributed=False) unless launched via
    # `torchrun --nproc_per_node=<N> train.py ...`, so a plain
    # `python train.py ...` still runs single-process exactly as before.
    is_distributed, rank, local_rank, world_size = setup_distributed()
    main_process = is_main_process(rank)

    # Auto-detect COCO paths + redirect output_dir when running on Kaggle
    if is_kaggle():
        cfg = patch_config_for_kaggle(cfg)

    # Setup output directory (rank 0 only — avoids N processes racing to
    # mkdir/write the same files)
    if main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
    if is_distributed:
        torch.distributed.barrier()  # other ranks wait until the dir exists

    # Logger — real logger on rank 0, silent no-op everywhere else so
    # trainer/logging code doesn't need `if main_process` guards everywhere
    if main_process:
        logger = TrainingLogger(cfg.output_dir, use_tensorboard=True)
        logger.info(f"Loaded config from {args.config}")
        logger.info(f"Output directory: {cfg.output_dir}")
        # v2: keep the exact config of every run next to its log and checkpoints (the 2026-07-29 run's config
        # was lost because it only existed on the pod)
        with open(os.path.join(cfg.output_dir, 'config_used.yaml'), 'w') as f:
            yaml.safe_dump(json.loads(json.dumps(cfg)), f, sort_keys=False)
        logger.info(f"Saved the resolved config to {os.path.join(cfg.output_dir, 'config_used.yaml')}")
        _w = lr_drop_warning(cfg)
        if _w:
            logger.info(f"WARNING: {_w}")
        if is_distributed:
            logger.info(f"Distributed training: {world_size} processes (GPUs)")
    else:
        logger = NullLogger()
    
    # Device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    if main_process:
        logger.info(f"Using device: {device}" + (f" (world_size={world_size})" if is_distributed else ""))

    if device.type == 'cuda':
        # Autotune conv algorithms per input shape. RandomResize now emits a
        # fixed set of square shapes (see data/transforms.py), so cuDNN's
        # one-time profiling cost per shape amortizes over the epoch instead
        # of re-profiling almost every step.
        torch.backends.cudnn.benchmark = True
    
    # Build model
    if main_process:
        logger.info("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    if main_process:
        logger.log_model_info(model)

    if device.type == 'cuda':
        # cuDNN's benchmark-selected backward-weight kernel for bias-less 1x1
        # convs (specifically MambaFusionBlock.conv1x1, Conv2d(512,256,k=1))
        # produces the weight grad in channels-last layout, which mismatches
        # DDP's contiguous bucket view and costs a silent reformat copy every
        # step ("Grad strides do not match bucket view strides" warning).
        # Force these grads contiguous BEFORE DDP registers its own hooks
        # (hooks fire in registration order), so DDP never has to reformat.
        import torch.nn as _nn
        _n_hooked = 0
        for _name, _module in model.named_modules():
            if isinstance(_module, _nn.Conv2d) and _module.kernel_size == (1, 1) and _module.bias is None:
                _module.weight.register_hook(lambda g: g.contiguous(memory_format=torch.contiguous_format))
                _n_hooked += 1
        if main_process:
            logger.info(f"Registered contiguous-grad hooks on {_n_hooked} bias-free 1x1 conv weights")

    if is_distributed:
        # DDP, not DataParallel: DataParallel replicates/scatters the module
        # across GPUs every forward call, which corrupts the ctx buffers of
        # mamba-ssm's custom CUDA autograd Functions during backward
        # (illegal memory access). DDP instead runs one full model per
        # process/GPU and only syncs gradients via all-reduce, which custom
        # CUDA kernels handle fine.
        # find_unused_parameters=False is intentional now that decoder.py's
        # enc_score_head is wired back into the graph (see apply_fix.py).
        # If you ever add another branch whose params don't always
        # contribute to loss, flip this to True instead of letting rank
        # processes hang/crash on bucket rebuild.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        if main_process:
            logger.info(f"Wrapped model in DistributedDataParallel across {world_size} GPUs")

    if device.type == 'cuda':
        # Warm up cuDNN's autotuner for every shape RandomResize can emit
        # (see data/transforms.py) *before* real training starts. Without
        # this, the first occurrence of each of the 6 shapes mid-epoch costs
        # ~18s while cuDNN benchmarks conv algorithms for it, and those
        # stalls are scattered unpredictably through early training instead
        # of paid once, up front.
        _warmup_sizes = [480, 512, 544, 576, 608, 640]
        _warmup_bs = cfg.data.batch_size
        if main_process:
            logger.info(f"Warming up cuDNN for shapes {_warmup_sizes} (batch={_warmup_bs})...")
        model.train()
        _was_amp = cfg.training.use_amp
        _warmup_scaler = torch.cuda.amp.GradScaler(enabled=_was_amp)
        for _sz in _warmup_sizes:
            # Backward is where the ~18s cuDNN re-profiling cost actually
            # shows up in the trainer.py DEBUG_TIMING logs, so the warmup
            # needs a real backward pass per shape, not just forward.
            _dummy = torch.randn(_warmup_bs, 3, _sz, _sz, device=device)
            with torch.cuda.amp.autocast(enabled=_was_amp):
                _out = model(_dummy)
                _dummy_loss = sum(
                    v.float().mean() for v in _out.values()
                    if torch.is_tensor(v) and v.dtype.is_floating_point
                )
            _warmup_scaler.scale(_dummy_loss).backward()
            model.zero_grad(set_to_none=True)
            del _dummy, _out, _dummy_loss
        torch.cuda.synchronize()
        if main_process:
            logger.info("cuDNN warmup done.")
    
    # Build criterion
    criterion = build_criterion(cfg)
    criterion = criterion.to(device)
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = build_scheduler(optimizer, cfg)
    
    # Build dataloaders
    if main_process:
        logger.info("Building dataloaders...")
    train_loader, val_loader, train_sampler = build_dataloaders(
        cfg, is_distributed=is_distributed, rank=rank, world_size=world_size
    )
    if main_process:
        logger.info(f"Train: {len(train_loader)} batches/process, Val: {len(val_loader)} batches")
        if is_distributed:
            logger.info(
                f"Per-process batch_size={cfg.data.batch_size} x world_size={world_size} "
                f"x accum_steps={getattr(cfg.training, 'accum_steps', 1)} = "
                f"effective batch {cfg.data.batch_size * world_size * getattr(cfg.training, 'accum_steps', 1)}"
            )
    
    # Build trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        cfg=cfg,
        train_sampler=train_sampler,
        rank=rank,
        world_size=world_size,
        main_process=main_process,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if main_process:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        # v2: the checkpoint holds the last COMPLETED epoch; continue with the next one (training used to repeat it)
        start_epoch = trainer.load_checkpoint(args.resume) + (0 if args.eval else 1)
    
    # Eval-only mode
    if args.eval:
        if main_process:
            logger.info("Running evaluation only...")
        metrics = trainer.evaluate(start_epoch)
        if main_process:
            logger.info(f"Evaluation results: {metrics}")
        cleanup_distributed()
        return
    
    # Train
    if main_process:
        logger.info(f"Starting training from epoch {start_epoch} to {cfg.training.epochs}")
    trainer.train(start_epoch=start_epoch, max_epochs=cfg.training.epochs)
    
    if main_process:
        logger.info("Training complete!")
        logger.close()

    cleanup_distributed()


if __name__ == '__main__':
    main()

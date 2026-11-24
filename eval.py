"""
MambaFusion-DETR Standalone Evaluation Script.

Usage:
    python eval.py --config configs/rtdetr_mamba_r50_cocominitrain_v2.yaml --checkpoint outputs/<run>/checkpoint_best.pth [--benchmark]
"""
import argparse
import yaml
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import load_config, build_model, build_criterion, DictAsMember
from data.coco_dataset import build_coco_dataloader
from data.transforms import build_transforms
from engine.trainer import Trainer
from utils.logger import TrainingLogger


def parse_args():
    parser = argparse.ArgumentParser(description='MambaFusion-DETR Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--benchmark', action='store_true',
                         help='Also report params/FLOPs/latency/FPS/GPU memory')
    parser.add_argument('--benchmark-batch-size', type=int, default=1,
                         help='Batch size used for the latency/FPS/memory benchmark (default: 1, i.e. single-image inference)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    if args.batch_size is not None:
        cfg['data']['batch_size'] = args.batch_size
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    logger = TrainingLogger(cfg.output_dir, use_tensorboard=False)
    
    logger.info(f"Loading model from {args.checkpoint}")
    
    # Build model
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'ema_model' in checkpoint:
        logger.info("Loading EMA model weights")
        model.load_state_dict(checkpoint['ema_model'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    
    # Build val dataloader
    val_transforms = build_transforms(is_train=False, img_size=cfg.data.img_size)
    val_loader = build_coco_dataloader(
        img_dir=cfg.data.val_img_dir,
        ann_file=cfg.data.val_ann_file,
        transforms=val_transforms,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=False,
    )
    
    # Create a minimal trainer for evaluation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        train_loader=[],
        val_loader=val_loader,
        device=device,
        logger=logger,
        cfg=cfg,
    )
    
    # Run evaluation
    metrics = trainer.evaluate(epoch)

    if args.benchmark:
        metrics.update(trainer.benchmark(batch_size=args.benchmark_batch_size))

    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        if v is None:
            logger.info(f"  {k}: N/A")
        else:
            logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)
    
    logger.close()


if __name__ == '__main__':
    main()

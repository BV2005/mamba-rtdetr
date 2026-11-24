# MambaFusion-DETR

RT-DETR (real-time DETR object detector, ResNet-50 backbone) whose CCFF feature-fusion blocks are replaced by
**MambaFusion** blocks: a selective state-space model (Mamba) that scans the fused feature maps in four directions,
plus 1×1 / 3×3 depthwise convolutions and pre-norm residuals. The encoder fuses P3, P4 and P5 (strides 8, 16, 32)
top-down and bottom-up, with AIFI self-attention on P5, as in RT-DETR.

The repository also contains a **like-for-like baseline**: the same model and training recipe with RT-DETR's own
fusion block (CSPRepLayer / RepC3) in the same four places (`encoder.fusion: repc3`), so the effect of the Mamba block
can be measured directly.

## Status

| | Data | Result |
|---|---|---|
| Best run so far (v1 code, 2026-07-29, one GPU, 12.2 h) | COCO minitrain-25k, 30 epochs | AP 0.194 / AP50 0.379 (best epoch), still rising; no baseline yet |
| **This version (v2)** | | fixes below, CPU-tested; not yet trained on a GPU |

v1 started from a loss of ~12,200 (classification heads without the focal-loss prior), trained the new Mamba blocks at
the backbone's learning rate, never decayed the learning rate and had no working EMA. v2 fixes these, adds RT-DETR's
augmentations, the RepC3 baseline, seeds and one-command training. Details and every check: **[V2_CHANGES.md](V2_CHANGES.md)**.

## Train it (Linux + NVIDIA GPU, e.g. a RunPod pod)

```bash
git clone https://github.com/BV2005/mamba-rtdetr.git
cd mamba-rtdetr
DATASET=minitrain FUSION=both bash train_full_coco.sh    # 25k-image COCO subset: MambaFusion, then the RepC3 baseline
bash train_full_coco.sh                                  # full COCO 2017, MambaFusion, 72 epochs
```

`train_full_coco.sh` does everything and can be re-run at any time (finished steps are skipped, training resumes
from the newest checkpoint):

1. a project venv `.venv` with everything downloaded into it: CUDA PyTorch (2.7.1 by default, the version with
   prebuilt mamba-ssm wheels), `requirements.txt`, mamba-ssm and causal-conv1d with their CUDA kernels; it then runs
   Mamba forward + backward on the GPU and stops if that fails;
2. COCO 2017 from cocodataset.org (~19 GB; ~45 GB free disk while unpacking) and, for `DATASET=minitrain`, the
   official annotations of the 25,000 minitrain images (`make_minitrain_annotations.py`);
3. the CPU checks (`test_v2_fixes.py`);
4. a config in `configs/generated/` and training (all GPUs via `torch.distributed.run`), detached with `nohup`.

Settings are environment variables: `DATASET` (full | minitrain), `FUSION` (mamba | repc3 | both), `SEED`, `EPOCHS`,
`LR_DROP`, `BATCH` (per GPU, default 20), `LR`, `GPUS`, `WORKERS`, `DATA`, `VENV`, `TORCH_VERSION`, `TORCH_CUDA`,
`STOP_AFTER` (python | data | tests), `FOREGROUND=1`, `DRY_RUN=1`. See the header of `train_full_coco.sh`.

Outputs per run (`outputs/<dataset>_<fusion>_seed<N>/`): `training.log` with the COCO evaluation after every epoch,
`config_used.yaml`, `checkpoint_epoch_N.pth` (last 3), `checkpoint_best.pth`, `best_metrics.json`.

## Manual use

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
MAMBA_KEEP_CUDA_BUILD=TRUE pip install causal-conv1d mamba-ssm --no-build-isolation
python test_v2_fixes.py                                        # CPU checks
python check_mamba_gpu.py                                      # Mamba's CUDA kernels on the GPU
python train.py --config configs/rtdetr_mamba_r50_cocominitrain_v2.yaml --seed 1           # paths: data/coco/...
python train.py --config configs/rtdetr_repc3_r50_cocominitrain_baseline.yaml --seed 1
python train.py --config <config> --resume outputs/<run>/checkpoint_epoch_N.pth            # continue a run
python train.py --config <config> --eval --resume outputs/<run>/checkpoint_best.pth        # evaluate only
python eval.py --config <config> --checkpoint outputs/<run>/checkpoint_best.pth --benchmark # + latency / FLOPs
```
Without mamba-ssm the model falls back to a pure-PyTorch selective scan with the same parameters: fine for the CPU
checks, far too slow to train.

## Layout

```
models/        backbone (ResNet-50), hybrid_encoder (AIFI + CCFF with MambaFusion or RepC3), mamba_fusion,
               decoder (RT-DETR decoder, denoising, IoU-aware query selection), losses, matcher
engine/        trainer (AMP, EMA, evaluation with pycocotools, checkpoints), prefetcher
data/          COCO dataset and augmentations (zoom-out, IoU crop, flip, multi-scale resize, colour jitter)
configs/       v2 MambaFusion and RepC3-baseline configs (COCO minitrain-25k); generated/ for train_full_coco.sh
train.py       training / evaluation entry point (single GPU or torchrun)
train_full_coco.sh, make_config.py, make_minitrain_annotations.py, check_mamba_gpu.py   one-command training
test_v2_fixes.py                                                                         CPU checks of every v2 fix
eval.py        evaluation of a checkpoint; --benchmark adds params / FLOPs / latency / FPS / GPU memory
data_lists/    the 25,000 image names of COCO minitrain (from Ultralytics' coco_minitrain_25k)
```

## Citation

```bibtex
@misc{mambafusion_detr,
  title  = {MambaFusion-DETR: Multi-Scale Feature Fusion with Selective State Spaces for RT-DETR},
  author = {Bharath},
  year   = {2026},
  url    = {https://github.com/BV2005/mamba-rtdetr}
}
```

## License

Apache 2.0 (`LICENSE`). COCO images and annotations are under their own terms (cocodataset.org); the minitrain image
list comes from Ultralytics' coco_minitrain_25k.

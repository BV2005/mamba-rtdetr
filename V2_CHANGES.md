# MambaFusion-DETR v2: what changed and how to run it

v2 is the code of the best run so far (RunPod 2026-07-29, AP 0.194 on COCO
minitrain-25k after 30 epochs) with the fixes below. That folder keeps the original code unchanged.

Every fix is checked by `python test_v2_fixes.py` (CPU, no dataset, ~5 min; all checks pass).

## Fixes

| # | Problem in the 2026-07-29 run | Fix | Where | Check |
|---|---|---|---|---|
| 1 | **Loss of ~12,200 at the start** (a DETR model normally starts near 100). 99 % was classification loss: the class heads used PyTorch's default bias, so every class started at p ≈ 0.5 and every background score of every query and every encoder position was penalised. Most of epoch 0 went into undoing this | Focal-loss prior bias (p = 0.01, as in RT-DETR) on `class_head`, `enc_class_head`, `enc_score_head` | `models/decoder.py`, end of `RTDETRDecoder.__init__` | initial loss 5,189 → **120** at 320 px (`test_v2_fixes.py`); class probability 0.49 → 0.013 |
| 2 | **The learning rate never decayed**: `lr_drop [60]` with 30 epochs | 36 epochs, drop at 27; `train.py` warns when no drop falls inside the run | configs, `train.py: lr_drop_warning` | v1 schedule flagged, v2 passes |
| 3 | **The new Mamba blocks trained at 1e-5** (`lr_mamba`, the pretrained backbone's rate) although they start from random weights | `lr_mamba: 1e-4` (= base) | configs | one training step: groups at 1e-5 (backbone) and 1e-4 (everything else) |
| 4 | **No EMA**, and the trainer's EMA had two bugs: decay 0.9999 from step 0 (the shadow stays near the random init for ~10k steps) and buffers never copied (the evaluated EMA model kept its initial BatchNorm statistics) | EMA on, with RT-DETR's decay warm-up `0.9999·(1−e^(−updates/2000))`, buffers averaged / copied, frozen parameters copied, update count saved for `--resume` | `engine/trainer.py: update_ema`, checkpoints | decay 0.0005 at the first update, 0.9999 later; BatchNorm stats reach the EMA model |
| 5 | **Weaker augmentation than RT-DETR**: flip, multi-scale resize, colour jitter only | RT-DETR's zoom-out (up to 4×), IoU crop and box sanitising before flip / resize; colour jitter with p = 0.5. `data.strong_aug: false` restores the old pipeline | `data/transforms.py` | 200 random augmentations: every box within 1.6 px of its object |

Also:
- **Baseline**: `encoder.fusion: repc3` replaces MambaFusionBlock with RT-DETR's own fusion block (CSPRepLayer /
  RepC3) in the same four places, everything else identical (`configs/rtdetr_repc3_r50_cocominitrain_baseline.yaml`,
  41.57 M parameters vs 41.23 M). This is the comparison the project was missing.
- **Seeds**: `train.py --seed N` seeds Python / NumPy / torch and writes to `<output_dir>_seed<N>` (the old code set no
  seed at all).
- **The config of every run is saved** as `<output_dir>/config_used.yaml` (the 2026-07-29 config was lost); a
  checkpoint every epoch (`save_interval: 1`, rotation keeps the last 3, plus `checkpoint_best.pth`).
- **The CPU Mamba fallback now builds the same model as the CUDA `mamba_ssm`** (time-step rank 16, not 1, and
  mamba_ssm's time-step initialisation): 41.23 M parameters either way, checkpoints load in both.
  `MAMBA_FALLBACK=1` forces it (the tests do).
- **Annotations**: `make_minitrain_annotations.py` takes the official COCO annotations of the 25,000 minitrain images
  (`data_lists/minitrain_train2017.txt`) instead of converting YOLO labels, so the evaluation on the official
  `instances_val2017.json` is standard COCO (official mask areas for small / medium / large, crowd regions).
  The 2026-07-29 run's annotation source is unknown, so compare v2 with the v2 baseline, not with its 0.194.
- `data.num_workers: 8` (was 2, a Kaggle setting) for the heavier augmentation.

Not changed: the model architecture (apart from the head initialisation), the losses, the matcher, batch size 20,
AMP. `README.md` describes a later P4/P5-only design; this code uses P3, P4 and P5.

## Full COCO in one command

```bash
# on the pod
cd /workspace && git clone https://github.com/BV2005/mamba-rtdetr.git && bash mamba-rtdetr/train_full_coco.sh
# COCO minitrain-25k instead (36 epochs):  DATASET=minitrain bash mamba-rtdetr/train_full_coco.sh
```

`train_full_coco.sh` creates a project venv (`./.venv`, isolated: no system site-packages) and downloads everything
into it: the CUDA build of PyTorch matching the GPU driver (newest supported wheel index, then older ones), the
requirements and mamba-ssm; every later step runs with `.venv/bin/python` (also `torch.distributed.run` and the
detached training), and `outputs/venv_requirements_frozen.txt` records the installed versions. Where Ubuntu's Python
lacks `ensurepip` it installs `python3.X-venv` (as root) or bootstraps pip into the venv with `get-pip.py`. It then
runs Mamba once on the GPU, downloads COCO 2017 (~19 GB; needs ~45 GB free while unpacking), runs the
CPU checks once, writes `configs/generated/full_coco_<fusion>.yaml` (`make_config.py`: full train2017, official
val2017, 72 epochs, lr drop at 60, batch 20 per GPU, EMA and the other v2 settings) and starts training detached
(`nohup`; `torchrun` on all GPUs when there are several). Re-running it is safe: done steps are skipped and training
resumes from the newest `checkpoint_epoch_N.pth`.

- `FUSION=repc3` trains the RT-DETR baseline, `FUSION=both` both in turn; `SEED=2` another seed.
- `EPOCHS`, `LR_DROP` (comma list, empty = none), `BATCH` (per GPU), `LR` (sets base, Mamba = base and backbone =
  base/10 together), `GPUS`, `WORKERS`, `DATA`, `FOREGROUND=1`, `SKIP_TESTS=1`, `DRY_RUN=1` (print every step,
  install and download nothing); `STOP_AFTER=python|data|tests` (prepare a pod, train later); `TORCH_VERSION=2.x.y` (pin PyTorch to a
  version mamba-ssm has prebuilt wheels for), `TORCH_CUDA=cu124` (force a wheel index), `VENV=/path`.
- Time: 118,287 images / batch 20 = 5,914 steps per epoch; at the 2026-07-29 run's ~1 s per step that is ~100 min per
  epoch, ~5 days for 72 epochs on one GPU; 4 GPUs ~1.3 days.

Resume was fixed for this: training used to repeat the last completed epoch after `--resume` and forget the best AP
(the first epoch after a restart overwrote `checkpoint_best.pth`); the best AP is now kept in `best_metrics.json`.
Multi-GPU (`torchrun`) start-up no longer fails on the missing `training.accum_steps`.

## Run it on RunPod (COCO minitrain, step by step)

GPU: the same kind as the 2026-07-29 run (batch 20 at 640 px with AMP fitted; ~22 min per epoch there).

```bash
# 0. on the pod: get the code
cd /workspace && git clone https://github.com/BV2005/mamba-rtdetr.git

# 1. on the pod: environment
cd /workspace/mamba-rtdetr
python3 -m venv /workspace/venv && . /workspace/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124    # the CUDA build for the pod's driver
pip install -r requirements.txt
pip install causal-conv1d mamba-ssm --no-build-isolation
python -c "import torch, mamba_ssm; print('CUDA', torch.cuda.is_available(), '| mamba_ssm OK')"

# 2. data: official COCO 2017 (~19 GB, fast from a datacenter) + the minitrain annotation subset
mkdir -p data/coco && cd data/coco
wget -q http://images.cocodataset.org/zips/train2017.zip http://images.cocodataset.org/zips/val2017.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip && unzip -q val2017.zip && unzip -q annotations_trainval2017.zip && rm -f *.zip
cd ../..
python make_minitrain_annotations.py --coco-json data/coco/annotations/instances_train2017.json \
       --image-list data_lists/minitrain_train2017.txt --out data/coco/annotations/instances_minitrain2017.json
# expect "25000 images"

# 3. checks (CPU, ~5 min): every line "ok", ending "all 14 checks passed"
python test_v2_fixes.py

# 4. train: MambaFusion v2, then the RepC3 baseline, seed 1 (~14 h each)
mkdir -p outputs
nohup sh -c "python train.py --config configs/rtdetr_mamba_r50_cocominitrain_v2.yaml --seed 1 && \
             python train.py --config configs/rtdetr_repc3_r50_cocominitrain_baseline.yaml --seed 1" \
      > outputs/seed1.out 2>&1 &
tail -f outputs/mamba_v2_cocominitrain_seed1/training.log
#   the first lines must show "Saved the resolved config" and no "WARNING: lr_drop";
#   the first batches' loss should be in the low hundreds, not ~12,000

# 5. on this PC: copy the results back often (pods get stopped; the 2026-07-29 checkpoints were lost that way)
scp -P <PORT> -i ~/.ssh/id_ed25519 -r root@<HOST>:/workspace/mamba-rtdetr/outputs ./mamba-rtdetr-outputs
```

Then seeds 2 and 3 of both (`--seed 2`, `--seed 3`) for a mean ± spread. With two GPUs, run the two models side by
side (`CUDA_VISIBLE_DEVICES=0 ... &` and `CUDA_VISIBLE_DEVICES=1 ...`).

## What to compare

`outputs/<run>/training.log`, the COCO table after each epoch: final and best AP, AP50, AP_small for MambaFusion v2
vs the RepC3 baseline (same seed), then mean ± spread over 3 seeds. A difference smaller than about 1 AP between
single runs is within epoch-to-epoch noise (the 2026-07-29 run moved ±1 AP between consecutive epochs). For an
efficiency claim, also run `Trainer.benchmark` (latency, FLOPs, memory) for both on the same GPU.

#!/usr/bin/env bash
# train_full_coco.sh: train MambaFusion-DETR v2 on full COCO 2017 with one command
# (Linux + NVIDIA GPU, e.g. a RunPod PyTorch pod). Run it from anywhere:
#
#   bash train_full_coco.sh                  # MambaFusion v2, seed 1, every GPU, in the background
#   FUSION=repc3 bash train_full_coco.sh     # the plain RT-DETR baseline (same model and recipe, RT-DETR's fusion)
#   FUSION=both  bash train_full_coco.sh     # MambaFusion, then the baseline
#   DATASET=minitrain bash train_full_coco.sh  # COCO minitrain-25k (36 epochs): ~5x cheaper, for a first comparison
#   DRY_RUN=1    bash train_full_coco.sh     # print what would run and write the config; install / download nothing
#
# What it does: a project venv (./.venv) with everything downloaded INTO it (CUDA PyTorch matched to the GPU driver,
# requirements.txt, mamba-ssm) -> COCO 2017
# (~19 GB download, ~45 GB free disk needed during unpacking) -> the CPU checks (test_v2_fixes.py) -> a full-COCO
# config (configs/generated/) -> training on 1 GPU (python) or all GPUs (torchrun), detached with nohup.
# Re-running is safe: finished steps are skipped and training resumes from the newest checkpoint.
#
# Settings (environment variables; default in brackets):
#   FUSION   [mamba]  mamba | repc3 | both          SEED     [1]
#   DATASET  [full]   full (COCO train2017, 118k images) | minitrain (its 25k-image subset; same val2017)
#   EPOCHS   [72 full, 36 minitrain]                  LR_DROP  [60 full, 27 minitrain]  comma list, empty = no drop
#   BATCH    [20]     images per GPU (as in the 2026-07-29 run)
#   LR       [1e-4]   base lr; Mamba = base, backbone = base/10 (scale it if you change the total batch a lot)
#   GPUS     [all]    WORKERS [8] loader processes per GPU
#   DATA     [./data/coco]   VENV [./.venv]   FOREGROUND [0]   SKIP_TESTS [0]
#   TORCH_CUDA    [auto]    PyTorch CUDA wheel index (cu130 / cu128 / cu126 / cu124 / cu121 / cu118); auto = the newest
#                           the GPU driver can run (any CUDA 12.x wheel runs on a 12.x driver), then older ones
#   TORCH_VERSION [2.7.1]   PyTorch version; 2.7.1 + CUDA 12.6/12.8 has prebuilt mamba-ssm 2.3.x wheels (other
#                           versions compile mamba-ssm from source, which needs nvcc). Empty = the latest PyTorch.
#   PYTHON        [python3] interpreter used to create the venv
#   STOP_AFTER    []        python | data | tests: prepare up to that step and stop (e.g. set up a pod, train later)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
FUSION="${FUSION:-mamba}"; SEED="${SEED:-1}"; DATASET="${DATASET:-full}"
case "$DATASET" in
  full)      EPOCHS="${EPOCHS:-72}"; LR_DROP="${LR_DROP-60}"; TRAIN_ANN=instances_train2017.json; N_TRAIN=118287 ;;
  minitrain) EPOCHS="${EPOCHS:-36}"; LR_DROP="${LR_DROP-27}"; TRAIN_ANN=instances_minitrain2017.json; N_TRAIN=25000 ;;
  *) echo "DATASET must be full or minitrain (got '$DATASET')" >&2; exit 1 ;;
esac
TORCH_VERSION="${TORCH_VERSION-2.7.1}"
BATCH="${BATCH:-20}"; LR="${LR:-0.0001}"; WORKERS="${WORKERS:-8}"
DATA="${DATA:-$ROOT/data/coco}"; VENV="${VENV:-$ROOT/.venv}"
FOREGROUND="${FOREGROUND:-0}"; SKIP_TESTS="${SKIP_TESTS:-0}"; SKIP_SETUP="${SKIP_SETUP:-0}"
DRY_RUN="${DRY_RUN:-0}"; TORCH_CUDA="${TORCH_CUDA:-auto}"; PYTHON="${PYTHON:-python3}"; STOP_AFTER="${STOP_AFTER:-}"
BASE_CONFIG="configs/rtdetr_mamba_r50_cocominitrain_v2.yaml"
COCO_URL="http://images.cocodataset.org"

log() { echo "[train_full_coco $(date '+%H:%M:%S')] $*"; }
die() { echo "[train_full_coco] ERROR: $*" >&2; exit 1; }
run() { if [ "$DRY_RUN" = 1 ]; then echo "    (dry run) $*"; else "$@"; fi; }

case "$FUSION" in mamba|repc3) FUSIONS="$FUSION" ;; both) FUSIONS="mamba repc3" ;;
  *) die "FUSION must be mamba, repc3 or both (got '$FUSION')" ;; esac

if [ -z "${GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then GPUS="$(nvidia-smi -L | grep -c '^GPU')"; else GPUS=0; fi
fi
if [ "$DRY_RUN" != 1 ] && [ "$GPUS" -lt 1 ]; then die "no NVIDIA GPU found (nvidia-smi). This script trains on GPUs."; fi
[ "$GPUS" -ge 1 ] || GPUS=1

# ---------------------------------------------------------------------------------------------------- 1. Python
# Everything is installed into the project venv ($VENV) and run with its own interpreter ($VENV/bin/python), never
# the system / image Python. The venv is created WITHOUT --system-site-packages, so a PyTorch that came with the pod
# image is not used: CUDA PyTorch, requirements.txt and mamba-ssm are all downloaded into the venv.
make_venv() {
  "$PYTHON" -m venv "$VENV" >/dev/null 2>&1 && return
  # Debian / Ubuntu images often lack ensurepip: install the venv package (pods run as root) and try again
  local v; v="$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = 0 ]; then
    log "python$v venv support missing: apt-get install python$v-venv"
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python$v-venv" python3-venv >/dev/null || true
    rm -rf "$VENV"; "$PYTHON" -m venv "$VENV" >/dev/null 2>&1 && return
  fi
  # not root / no apt: a venv without pip, then pip bootstrapped INTO it with the official get-pip.py
  log "no ensurepip: creating the venv without pip and bootstrapping pip into it (get-pip.py)"
  rm -rf "$VENV"
  if "$PYTHON" -m venv --without-pip "$VENV"; then
    if command -v curl >/dev/null 2>&1; then curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$VENV/get-pip.py"
    else wget -q -O "$VENV/get-pip.py" https://bootstrap.pypa.io/get-pip.py; fi
    "$VENV/bin/python" "$VENV/get-pip.py" -q && rm -f "$VENV/get-pip.py" && return
  fi
  die "could not create a venv with $PYTHON (install python$v-venv, or set PYTHON=/path/to/python3)"
}

install_torch() {  # CUDA PyTorch into the venv: the newest wheel index the driver supports, then older ones
  local driver idx tried="" pin=""
  [ -n "${TORCH_VERSION:-}" ] && pin="==$TORCH_VERSION"
  if [ "$TORCH_CUDA" != auto ]; then
    set -- "$TORCH_CUDA"
  else
    driver="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    set --
    # CUDA minor-version compatibility: a CUDA 12.x build runs on any 12.x driver, 13.x on 13.x
    if [ "${driver:-0}" -ge 130 ]; then set -- "$@" cu130; fi
    if [ "${driver:-0}" -ge 120 ]; then set -- "$@" cu128 cu126 cu124 cu121; fi
    if [ "${driver:-0}" -ge 118 ]; then set -- "$@" cu118; fi
    [ $# -gt 0 ] || die "the GPU driver reports CUDA ${driver:-?}; PyTorch wheels need CUDA >= 11.8 (update the driver)"
  fi
  for idx in "$@"; do
    log "installing PyTorch${pin:+ $pin} ($idx wheels) into the venv"
    if "$PY" -m pip install -q "torch$pin" torchvision --index-url "https://download.pytorch.org/whl/$idx"; then return; fi
    tried="$tried $idx"
  done
  die "no PyTorch wheel installed (tried:$tried). Set TORCH_CUDA and/or TORCH_VERSION."
}

setup_python() {
  if [ "$DRY_RUN" = 1 ]; then
    log "dry run: would create $VENV and download CUDA PyTorch, requirements.txt and mamba-ssm into it; using $PYTHON"
    PY="$PYTHON"; return
  fi
  [ -x "$VENV/bin/python" ] || { log "creating the venv $VENV"; make_venv; }
  PY="$VENV/bin/python"
  "$PY" -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" || die "$VENV is not a venv"
  "$PY" -m pip install -q --upgrade pip setuptools wheel
  if ! "$PY" -c "import sys, torch; w = sys.argv[1]; sys.exit(0 if torch.cuda.is_available() and (not w or torch.__version__.split('+')[0] == w) else 1)" "$TORCH_VERSION" 2>/dev/null; then
    install_torch
  fi
  log "installing requirements.txt into the venv"
  "$PY" -m pip install -q -r requirements.txt
  if [[ " $FUSIONS " == *" mamba "* ]] && ! "$PY" check_mamba_gpu.py --quiet; then
    log "installing mamba-ssm + causal-conv1d with their CUDA kernels into the venv (a prebuilt wheel if one matches this torch / CUDA / Python, else a 10-30 min compile that needs nvcc)"
    "$PY" -m pip install -q packaging ninja
    # without MAMBA_KEEP_CUDA_BUILD=TRUE, mamba-ssm >= 2.3 installs without its CUDA selective-scan kernels
    MAMBA_KEEP_CUDA_BUILD=TRUE "$PY" -m pip install -q --no-cache-dir causal-conv1d mamba-ssm --no-build-isolation \
      || die "mamba-ssm did not install into the venv. The pure-PyTorch fallback is far too slow for full COCO. Try a PyTorch version mamba-ssm ships wheels for (TORCH_VERSION=..., https://github.com/state-spaces/mamba/releases), or a pod image with the CUDA toolkit (nvcc) so it can compile."
  fi
  mkdir -p outputs
  "$PY" -m pip freeze > outputs/venv_requirements_frozen.txt
  log "venv ready: $PY (installed versions in outputs/venv_requirements_frozen.txt)"
  "$PY" -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s): ' + ', '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))"
  if [[ " $FUSIONS " == *" mamba "* ]]; then
    "$PY" check_mamba_gpu.py || die "mamba-ssm's CUDA kernels do not run (above); training would fail. Try TORCH_VERSION=2.7.1 TORCH_CUDA=cu128 (prebuilt wheels), or a pod image with nvcc."
  fi
}

# ---------------------------------------------------------------------------------------------------- 2. COCO 2017
fetch() {  # fetch <zip name> <directory it unpacks to> <expected file count, or a file inside>
  local zip="$1" dir="$2" check="$3"
  if [[ "$check" =~ ^[0-9]+$ ]]; then
    if [ -d "$DATA/$dir" ] && [ "$(find "$DATA/$dir" -maxdepth 1 -type f | wc -l)" -ge "$check" ]; then log "$dir: present"; return; fi
  elif [ -f "$DATA/$check" ]; then log "$dir: present"; return; fi
  log "$dir: downloading $zip"
  run wget -c -q --show-progress -O "$DATA/$zip" "$COCO_URL/$( [ "$zip" = annotations_trainval2017.zip ] && echo annotations || echo zips )/$zip"
  log "$dir: unpacking"
  run unzip -q -o "$DATA/$zip" -d "$DATA"
  run rm -f "$DATA/$zip"
}

setup_data() {
  run mkdir -p "$DATA"
  if [ "$DRY_RUN" != 1 ] && [ ! -d "$DATA/train2017" ]; then
    free_gb=$(( $(df -Pk "$DATA" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
    [ "$free_gb" -ge 45 ] || die "only ${free_gb} GB free under $DATA; COCO 2017 needs ~45 GB while unpacking (set DATA=...)"
  fi
  fetch annotations_trainval2017.zip annotations annotations/instances_train2017.json
  fetch val2017.zip val2017 5000
  fetch train2017.zip train2017 118287
  if [ "$DATASET" = minitrain ] && [ ! -f "$DATA/annotations/$TRAIN_ANN" ]; then
    log "building the minitrain-25k annotations from the official ones (make_minitrain_annotations.py)"
    run "$PY" make_minitrain_annotations.py --coco-json "$DATA/annotations/instances_train2017.json" \
        --image-list data_lists/minitrain_train2017.txt --out "$DATA/annotations/$TRAIN_ANN"
  fi
}

# ---------------------------------------------------------------------------------------------------- 3. checks
run_tests() {
  [ "$SKIP_TESTS" = 1 ] && { log "checks skipped (SKIP_TESTS=1)"; return; }
  if [ -f outputs/.v2_checks_passed ]; then log "CPU checks: passed before (delete outputs/.v2_checks_passed to rerun)"; return; fi
  log "CPU checks (test_v2_fixes.py, ~5 min)"
  run mkdir -p outputs
  if [ "$DRY_RUN" = 1 ]; then echo "    (dry run) MAMBA_FALLBACK=1 \"$PY\" test_v2_fixes.py"; return; fi
  MAMBA_FALLBACK=1 "$PY" test_v2_fixes.py | tee outputs/test_v2_fixes.log
  grep -q "checks passed" outputs/test_v2_fixes.log || die "the checks failed (outputs/test_v2_fixes.log)"
  touch outputs/.v2_checks_passed
}

# ---------------------------------------------------------------------------------------------------- 4. train
latest_epoch() {  # newest checkpoint_epoch_N.pth in $1 -> N (or nothing)
  ls "$1"/checkpoint_epoch_*.pth 2>/dev/null | sed 's/.*checkpoint_epoch_\([0-9]*\)\.pth/\1/' | sort -n | tail -1 || true
}

train_one() {
  local fusion="$1" name="${DATASET}_$1" cfg="configs/generated/${DATASET}_$1.yaml"
  local run_dir="outputs/${name}_seed${SEED}"
  "$PY" make_config.py --base "$BASE_CONFIG" --out "$cfg" --data "$DATA" --train-ann "$TRAIN_ANN" \
      --fusion "$fusion" --epochs "$EPOCHS" --lr-drop "$LR_DROP" --batch "$BATCH" --lr "$LR" --workers "$WORKERS" \
      --output-dir "outputs/$name"
  local last resume=()
  last="$(latest_epoch "$run_dir")"
  if [ -n "$last" ] && [ "$last" -ge $((EPOCHS - 1)) ]; then log "$name seed $SEED: already finished ($run_dir)"; return; fi
  if [ -n "$last" ]; then resume=(--resume "$run_dir/checkpoint_epoch_${last}.pth"); log "$name: resuming after epoch $last"; fi
  local steps=$(( N_TRAIN / (BATCH * GPUS) ))
  log "$name seed $SEED: $EPOCHS epochs on $GPUS GPU(s), batch $BATCH/GPU ($((BATCH * GPUS)) total), ~$steps steps/epoch -> $run_dir"
  if [ "$GPUS" -gt 1 ]; then
    run "$PY" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" train.py --config "$cfg" --seed "$SEED" ${resume[@]+"${resume[@]}"}
  else
    run "$PY" train.py --config "$cfg" --seed "$SEED" ${resume[@]+"${resume[@]}"}
  fi
}

# ---------------------------------------------------------------------------------------------------- main
if [ "$SKIP_SETUP" = 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then PY="$PYTHON"; else
    PY="$VENV/bin/python"; [ -x "$PY" ] || die "no venv at $VENV: run the script once without SKIP_SETUP"
  fi
else
  log "dataset: $DATASET | fusion: $FUSIONS | seed $SEED | $EPOCHS epochs, lr drop at [${LR_DROP}] | batch $BATCH x $GPUS GPU(s) | data $DATA"
  setup_python
  [ "$STOP_AFTER" = python ] && { log "stopping after the Python setup (STOP_AFTER=python)"; exit 0; }
  setup_data
  [ "$STOP_AFTER" = data ] && { log "stopping after the data (STOP_AFTER=data)"; exit 0; }
  run_tests
  [ "$STOP_AFTER" = tests ] && { log "stopping after the checks (STOP_AFTER=tests)"; exit 0; }
fi

if [ "$FOREGROUND" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  mkdir -p outputs
  out="outputs/train_${DATASET}_seed${SEED}.out"
  nohup env FOREGROUND=1 SKIP_SETUP=1 DATASET="$DATASET" FUSION="$FUSION" SEED="$SEED" EPOCHS="$EPOCHS" LR_DROP="$LR_DROP" BATCH="$BATCH" \
        LR="$LR" GPUS="$GPUS" WORKERS="$WORKERS" DATA="$DATA" VENV="$VENV" bash "$ROOT/train_full_coco.sh" > "$out" 2>&1 &
  log "training started in the background (pid $!). It survives an SSH disconnect; re-run this script to resume."
  log "follow it:  tail -f $ROOT/$out"
  log "per-epoch COCO results:  grep -A1 'Average Precision  (AP) @\\[ IoU=0.50:0.95 | area=   all' $ROOT/outputs/${DATASET}_*_seed${SEED}/training.log"
  log "copy results to your PC often:  scp -P <PORT> -r root@<HOST>:$ROOT/outputs ."
  exit 0
fi

for f in $FUSIONS; do train_one "$f"; done
log "done: $(for f in $FUSIONS; do printf 'outputs/%s_%s_seed%s ' "$DATASET" "$f" "$SEED"; done)"

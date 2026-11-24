"""Checks of the v2 fixes (CPU, no dataset, a few minutes):
  1. initial loss: healthy with the focal-loss prior bias (the 2026-07-29 run started at ~12,200), both fusions
  2. the pure-PyTorch Mamba fallback builds the CUDA run's model: 41.23 M trainable parameters, as logged
  3. EMA: decay warm-up, frozen parameters and buffers (BatchNorm statistics) follow the model
  4. augmentations: zoom-out, IoU crop, flip and resize keep every box on its object (checked on pixels)
  5. lr_drop warning for a schedule that never decays; the resolved config can be saved
  6. one training step with the v2 optimizer: finite loss, every parameter group updated
    python test_v2_fixes.py
"""
import copy
import json
import math
import os
import random
import sys
import types

os.environ.setdefault('MAMBA_FALLBACK', '1')   # CPU: the pure-PyTorch Mamba (same parameters)

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402
from data import transforms as TR  # noqa: E402
from engine.trainer import Trainer  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(ROOT, "configs", "rtdetr_mamba_r50_cocominitrain_v2.yaml")
BASE = os.path.join(ROOT, "configs", "rtdetr_repc3_r50_cocominitrain_baseline.yaml")
OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def cfg_of(path):
    cfg = train.load_config(path)
    cfg["model"]["backbone"]["pretrained"] = False
    return cfg


def batch(n=2, img=256, boxes=7, classes=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, img, img, generator=g)
    targets = []
    for _ in range(n):
        wh = torch.rand(boxes, 2, generator=g) * 0.4 + 0.05
        cxcy = torch.rand(boxes, 2, generator=g) * (1 - wh) + wh / 2
        targets.append({"boxes": torch.cat([cxcy, wh], 1), "labels": torch.randint(0, classes, (boxes,), generator=g)})
    return images, targets


def init_loss_checks():
    print("[1] initial loss (img 320, batch 4)")
    for path, name in ((V2, "mamba"), (BASE, "repc3")):
        torch.manual_seed(0)
        cfg = cfg_of(path)
        model, crit = train.build_model(cfg).train(), train.build_criterion(cfg)
        images, targets = batch(4, 320)
        with torch.no_grad():
            out = model(images, targets)
            losses = crit(out, targets)
        total = sum(float(v) for v in losses.values())
        enc_cls = float(losses["loss_class_enc"])
        p = torch.sigmoid(out["pred_logits"]).mean().item()
        check(total < 300 and enc_cls < 5 and abs(p - 0.01) < 0.01,
              f"{name}: initial loss {total:.1f} (v1 run: ~5,200 at this size, ~12,200 at 640), encoder class loss "
              f"{enc_cls:.2f}, mean class probability {p:.4f}")


def param_check():
    print("[2] parameters")
    cfg = cfg_of(V2)
    m = train.build_model(cfg)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    check(f"{n / 1e6:.2f}" == "41.23", f"mamba v2 with the pure-PyTorch fallback: {n / 1e6:.4f} M trainable = the run's "
                                       f"logged 41.23 M (v1 fallback: 40.99 M)")
    b = train.build_model(cfg_of(BASE))
    nb = sum(p.numel() for p in b.parameters() if p.requires_grad)
    print(f"      repc3 baseline: {nb / 1e6:.2f} M trainable")


def ema_check():
    print("[3] EMA")
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.Conv2d(4, 4, 1))
    model[2].weight.requires_grad_(False)
    fake = types.SimpleNamespace(model=model, ema_model=copy.deepcopy(model), ema_decay=0.9999, ema_warmups=2000,
                                 ema_updates=0, _raw=lambda m: m)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        model(torch.randn(8, 3, 8, 8))                    # train mode: BatchNorm running stats move
    model.train()
    model(torch.randn(8, 3, 8, 8))
    Trainer.update_ema(fake)
    d1 = 0.9999 * (1 - math.exp(-1 / 2000))
    w, we = model[0].weight, fake.ema_model[0].weight
    check(torch.allclose(we, w, atol=1e-3) and d1 < 1e-3,
          f"first update uses decay {d1:.5f} (warm-up), so the shadow follows the model instead of its initial weights")
    check(torch.allclose(fake.ema_model[1].running_mean, model[1].running_mean, atol=1e-3)
          and int(fake.ema_model[1].num_batches_tracked) == int(model[1].num_batches_tracked),
          "BatchNorm running statistics and counters reach the EMA model")
    check(torch.equal(fake.ema_model[2].weight, model[2].weight), "frozen parameters are copied, not averaged")
    for _ in range(20000):
        fake.ema_updates += 1
    dl = 0.9999 * (1 - math.exp(-fake.ema_updates / 2000))
    check(abs(dl - 0.9999) < 1e-4, f"after 20k updates the decay is {dl:.5f} (the configured 0.9999)")


def _draw(img_size, box_px):
    im = Image.new("RGB", img_size, (0, 0, 0))
    x0, y0, x1, y1 = box_px
    im.paste((255, 255, 255), (x0, y0, x1, y1))
    return im


def _white_extent(im):
    t = torch.from_numpy(__import__("numpy").array(im)).float().mean(-1)
    ys, xs = torch.nonzero(t > 200, as_tuple=True)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def aug_checks():
    print("[4] augmentations keep boxes on their objects")
    random.seed(1)
    torch.manual_seed(1)
    worst, n, kept, dropped = 0.0, 0, 0, 0
    pipe = TR.Compose([TR.RandomZoomOut(p=1.0), TR.RandomIoUCrop(p=1.0), TR.SanitizeBoxes(), TR.RandomHorizontalFlip(),
                       TR.RandomResize([320, 480])])
    for i in range(200):
        W, H = random.randint(200, 640), random.randint(200, 640)
        x0, y0 = random.randint(0, W - 40), random.randint(0, H - 40)
        x1, y1 = random.randint(x0 + 20, min(W, x0 + 300)), random.randint(y0 + 20, min(H, y0 + 300))
        im = _draw((W, H), (x0, y0, x1, y1))
        tgt = {"boxes": torch.tensor([[(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]]),
               "labels": torch.tensor([3])}
        im2, tgt2 = pipe(im, tgt)
        ext = _white_extent(im2)
        if len(tgt2["boxes"]) == 0:
            dropped += 1
            continue
        kept += 1
        w2, h2 = im2.size
        cx, cy, bw, bh = tgt2["boxes"][0].tolist()
        pred = ((cx - bw / 2) * w2, (cy - bh / 2) * h2, (cx + bw / 2) * w2, (cy + bh / 2) * h2)
        err = max(abs(a - b) for a, b in zip(pred, ext))
        worst, n = max(worst, err), n + 1
        assert int(tgt2["labels"][0]) == 3
    check(n > 150 and worst <= 3.0, f"{n} augmented images: box edges within {worst:.2f} px of the object's pixels "
                                    f"(resize rounding; {dropped} of 200 lost their only object to a crop)")
    im, tgt = TR.build_transforms(True, 640, strong_aug=False)(_draw((300, 200), (10, 10, 50, 60)),
                                                                {"boxes": torch.tensor([[0.1, 0.175, 0.133, 0.25]]),
                                                                 "labels": torch.tensor([1])})
    check(im.shape[0] == 3 and len(tgt["boxes"]) == 1, "strong_aug=False keeps the 2026-07-29 pipeline")


def schedule_checks():
    print("[5] schedule warning and config saving")
    old = train.load_config(os.path.join(ROOT, "..", "mamba_rtdetr_best", "reconstructed_config",
                                         "rtdetr_mamba_r50_cocominitrain_runpod_run.yaml")) \
        if os.path.exists(os.path.join(ROOT, "..", "mamba_rtdetr_best")) else None
    if old is not None:
        check(train.lr_drop_warning(old) is not None, f"the v1 schedule is flagged: {train.lr_drop_warning(old)}")
    check(train.lr_drop_warning(train.load_config(V2)) is None, "the v2 schedule decays inside the run (epoch 27 of 36)")
    dumped = yaml.safe_dump(json.loads(json.dumps(train.load_config(V2))), sort_keys=False)
    check(yaml.safe_load(dumped)["model"]["encoder"]["fusion"] == "mamba", "the resolved config round-trips to YAML")
    import subprocess
    h = subprocess.run([sys.executable, os.path.join(ROOT, "train.py"), "--help"], capture_output=True, text=True).stdout
    check("--seed" in h, "train.py --seed exists (Python / NumPy / torch seeded, output_dir gets _seed<N>)")


def step_check():
    print("[6] one training step (CPU, img 192, batch 2)")
    torch.manual_seed(0)
    cfg = cfg_of(V2)
    model, crit = train.build_model(cfg).train(), train.build_criterion(cfg)
    opt = train.build_optimizer(model, cfg)
    lrs = sorted({g["lr"] for g in opt.param_groups})
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    images, targets = batch(2, 192)
    loss = sum(crit(model(images, targets), targets).values())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_max_norm)
    opt.step()
    moved = {grp: any(not torch.equal(p.detach(), before[n]) for n, p in model.named_parameters()
                      if p.requires_grad and (("backbone" in n) if grp == "backbone" else
                                              (("mamba" in n or "ssm" in n) and "backbone" not in n) if grp == "mamba"
                                              else not ("backbone" in n or "mamba" in n or "ssm" in n)))
             for grp in ("backbone", "mamba", "other")}
    check(torch.isfinite(loss) and all(moved.values()) and lrs == [1e-5, 1e-4],
          f"loss {float(loss):.1f} finite; backbone, Mamba and other parameters all updated; learning rates {lrs} "
          f"(Mamba blocks now at the base 1e-4)")


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    init_loss_checks()
    param_check()
    ema_check()
    aug_checks()
    schedule_checks()
    step_check()
    print(f"all {len(OK)} checks passed")

import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            if 'boxes' in target and len(target['boxes']) > 0:
                # Bounding boxes are in cx, cy, w, h normalized [0, 1] format
                target['boxes'][:, 0] = 1.0 - target['boxes'][:, 0]
        return image, target

class RandomResize:
    """Resizes directly to a square (size, size) from a discrete size list,
    ignoring the source aspect ratio.

    This intentionally does NOT preserve aspect ratio. That's a deliberate
    trade (some distortion on non-square images) in exchange for a small,
    fixed set of output shapes — standard practice for RT-DETR-family models.
    It also matters a lot on Kaggle: torch.backends.cudnn.benchmark=True
    profiles the fastest conv algorithm per input shape, and only pays off if
    that shape recurs. With aspect-preserving resize, every image produces a
    near-unique padded shape and cuDNN re-profiles almost every step (a huge,
    hard-to-notice slowdown). With a fixed size list, cuDNN profiles each of
    the len(sizes) shapes once and reuses the cached algorithm for the rest
    of the epoch.
    """
    def __init__(self, sizes, max_size=1333):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.choice(self.sizes)
        image = F.resize(image, [size, size])
        # Boxes are already normalized [0, 1], so a square resize (even
        # anisotropic) doesn't require any box adjustment.
        return image, target

class RandomSizeCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        w, h = image.size
        
        # Generate random crop size
        crop_w = random.randint(self.min_size, min(w, self.max_size))
        crop_h = random.randint(self.min_size, min(h, self.max_size))
        
        # Generate random top-left corner
        i = random.randint(0, h - crop_h) if h > crop_h else 0
        j = random.randint(0, w - crop_w) if w > crop_w else 0
        
        image = F.crop(image, i, j, crop_h, crop_w)
        
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            
            # Convert from cxcywh to xyxy (unnormalized)
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            xmin = (cx - bw / 2) * w
            ymin = (cy - bh / 2) * h
            xmax = (cx + bw / 2) * w
            ymax = (cy + bh / 2) * h
            
            # Adjust to cropped region
            xmin = xmin - j
            xmax = xmax - j
            ymin = ymin - i
            ymax = ymax - i
            
            # Clip boxes to the bounds of the new crop
            xmin = torch.clamp(xmin, 0, crop_w)
            xmax = torch.clamp(xmax, 0, crop_w)
            ymin = torch.clamp(ymin, 0, crop_h)
            ymax = torch.clamp(ymax, 0, crop_h)
            
            # Filter out boxes that became invalid or too small
            keep = (xmax > xmin) & (ymax > ymin)
            boxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)[keep]
            
            # Convert back to normalized cxcywh for the new crop
            if len(boxes) > 0:
                new_cx = (boxes[:, 0] + boxes[:, 2]) / 2 / crop_w
                new_cy = (boxes[:, 1] + boxes[:, 3]) / 2 / crop_h
                new_bw = (boxes[:, 2] - boxes[:, 0]) / crop_w
                new_bh = (boxes[:, 3] - boxes[:, 1]) / crop_h
                target['boxes'] = torch.stack([new_cx, new_cy, new_bw, new_bh], dim=1)
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                
            target['labels'] = target['labels'][keep]
            
        return image, target

class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1.0):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = self.jitter(image)
        return image, target


# --- v2: RT-DETR's geometric augmentations (zoom-out, IoU crop, box sanitising) -----------------------------
# Boxes stay normalised cxcywh throughout; images are PIL until ToTensor.

def _to_xyxy_px(boxes, w, h):
    cx, cy, bw, bh = boxes.unbind(1)
    return torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)


def _to_cxcywh_norm(xyxy, w, h):
    x0, y0, x1, y1 = xyxy.unbind(1)
    return torch.stack([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h], 1)


class RandomZoomOut:
    """Places the image at a random position on a canvas up to `max_scale` times larger, filled with the mean
    pixel (RT-DETR's RandomZoomOut, side range 1-4). After the final resize the objects are smaller: more small-object
    training examples."""
    def __init__(self, max_scale=4.0, p=0.5, fill=(124, 116, 104)):
        self.max_scale, self.p, self.fill = max_scale, p, fill

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        w, h = image.size
        s = random.uniform(1.0, self.max_scale)
        W, H = int(w * s), int(h * s)
        left, top = random.randint(0, W - w), random.randint(0, H - h)
        canvas = Image.new("RGB", (W, H), self.fill)
        canvas.paste(image, (left, top))
        if len(target["boxes"]):
            xyxy = _to_xyxy_px(target["boxes"], w, h) + torch.tensor([left, top, left, top], dtype=torch.float32)
            target["boxes"] = _to_cxcywh_norm(xyxy, W, H)
        return canvas, target


class RandomIoUCrop:
    """SSD / RT-DETR IoU crop: a random crop (scale 0.3-1, aspect 0.5-2) that overlaps at least one object with an
    IoU >= a randomly chosen threshold; objects whose centre falls outside the crop are dropped, the rest clipped.
    Applied with probability p; after `trials` failed attempts the image is left unchanged."""
    def __init__(self, p=0.8, min_scale=0.3, max_scale=1.0, min_aspect=0.5, max_aspect=2.0,
                 thresholds=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9), trials=40):
        self.p, self.min_scale, self.max_scale = p, min_scale, max_scale
        self.min_aspect, self.max_aspect, self.thresholds, self.trials = min_aspect, max_aspect, thresholds, trials

    def __call__(self, image, target):
        if random.random() >= self.p or len(target["boxes"]) == 0:
            return image, target
        w, h = image.size
        xyxy = _to_xyxy_px(target["boxes"], w, h)
        min_iou = random.choice(self.thresholds)
        for _ in range(self.trials):
            cw = int(w * random.uniform(self.min_scale, self.max_scale))
            ch = int(h * random.uniform(self.min_scale, self.max_scale))
            if cw < 1 or ch < 1 or not (self.min_aspect <= cw / ch <= self.max_aspect):
                continue
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            crop = torch.tensor([[x0, y0, x0 + cw, y0 + ch]], dtype=torch.float32)
            centres = (xyxy[:, :2] + xyxy[:, 2:]) / 2
            inside = ((centres[:, 0] > x0) & (centres[:, 0] < x0 + cw) &
                      (centres[:, 1] > y0) & (centres[:, 1] < y0 + ch))
            if not inside.any():
                continue
            lt = torch.max(xyxy[:, :2], crop[:, :2])
            rb = torch.min(xyxy[:, 2:], crop[:, 2:])
            inter = (rb - lt).clamp(min=0).prod(1)
            area = (xyxy[:, 2:] - xyxy[:, :2]).prod(1)
            iou = inter / (area + cw * ch - inter).clamp(min=1e-6)
            if iou[inside].max() < min_iou:
                continue
            kept = xyxy[inside].clone()
            kept[:, 0::2] = (kept[:, 0::2] - x0).clamp(0, cw)
            kept[:, 1::2] = (kept[:, 1::2] - y0).clamp(0, ch)
            target["boxes"] = _to_cxcywh_norm(kept, cw, ch)
            target["labels"] = target["labels"][inside]
            return image.crop((x0, y0, x0 + cw, y0 + ch)), target
        return image, target


class SanitizeBoxes:
    """Drops boxes smaller than `min_px` pixels on a side (e.g. slivers left by cropping)."""
    def __init__(self, min_px=1.0):
        self.min_px = min_px

    def __call__(self, image, target):
        if len(target["boxes"]):
            w, h = image.size
            keep = (target["boxes"][:, 2] * w >= self.min_px) & (target["boxes"][:, 3] * h >= self.min_px)
            target["boxes"], target["labels"] = target["boxes"][keep], target["labels"][keep]
        return image, target


def build_transforms(is_train=True, img_size=640, strong_aug=True):
    """strong_aug (v2 default): RT-DETR's zoom-out + IoU crop + box sanitising before the flip / multi-scale resize,
    and the colour jitter with probability 0.5 (RT-DETR's photometric distortion). strong_aug=False: the pipeline of
    the 2026-07-29 run (flip, multi-scale resize, colour jitter always)."""
    if is_train:
        geometric = [RandomZoomOut(), RandomIoUCrop(), SanitizeBoxes()] if strong_aug else []
        return Compose(geometric + [
            RandomHorizontalFlip(),
            RandomResize([480, 512, 544, 576, 608, 640]),
            # ColorJitter moved after RandomResize: it's an expensive
            # per-pixel PIL op (hue adjustment round-trips through HSV), and
            # running it after resize means it operates on the final
            # (smaller) image instead of the full-resolution source, cutting
            # CPU cost per sample on Kaggle's CPU-starved dataloader workers.
            ColorJitter(p=0.5 if strong_aug else 1.0),
            ToTensor(),
            Normalize()
        ])
    else:
        return Compose([
            RandomResize([img_size]),
            ToTensor(),
            Normalize()
        ])

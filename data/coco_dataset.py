import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import cv2
from pycocotools.coco import COCO

# 80 standard COCO classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Mapping from COCO's 91-category id to contiguous 0-79
coco91_to_coco80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
    14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
    24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
    37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
    58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
    72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
    82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

class CocoDetection(Dataset):
    #\"\"\"
    #COCO Detection Dataset.
    #Loads images and annotations, maps category IDs, and formats bounding boxes.
    #\"\"\"
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Returns: (image_tensor, target_dict)
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        # cv2.imread uses libjpeg-turbo under the hood and decodes noticeably
        # faster than PIL for the same JPEG files. Everything downstream
        # (data/transforms.py) expects a PIL Image, so we decode with cv2
        # for speed and immediately wrap it back into one — no behavior
        # change beyond decode speed. cv2 loads BGR, so convert to RGB.
        cv2_image = cv2.imread(path, cv2.IMREAD_COLOR)
        if cv2_image is None:
            # Fall back to PIL for any file cv2 can't handle (e.g. a
            # non-standard/corrupt JPEG, grayscale edge cases, etc.) rather
            # than crashing the whole training run on one bad file.
            image = Image.open(path).convert('RGB')
        else:
            image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        
        orig_w, orig_h = image.size
        
        boxes = []
        labels = []
        
        for ann in anns:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
                
            # COCO boxes are [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to cx, cy, w, h and normalize
            cx = x + w / 2
            cy = y + h / 2
            
            boxes.append([cx / orig_w, cy / orig_h, w / orig_w, h / orig_h])
            labels.append(coco91_to_coco80[ann['category_id']])
            
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_h, orig_w])
        }
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target

def collate_fn(batch):
   # \"\"\"
    #Custom collate function to handle variable length targets and pad images.
    #Returns: (batched_images, list_of_targets)
    #\"\"\"
    images, targets = list(zip(*batch))
    
    # Find max dimensions in the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Create batched images and padding mask
    batched_images = torch.zeros((len(images), 3, max_h, max_w), dtype=torch.float32)
    padding_mask = torch.ones((len(images), max_h, max_w), dtype=torch.bool)
    
    for i, img in enumerate(images):
        c, h, w = img.shape
        batched_images[i, :c, :h, :w] = img
        padding_mask[i, :h, :w] = False
        
    return batched_images, list(targets)

def build_coco_dataloader(img_dir, ann_file, transforms=None, batch_size=4,
                           num_workers=4, shuffle=True, drop_last=True,
                           pin_memory=True, persistent_workers=True, prefetch_factor=4,
                           is_distributed=False, rank=0, world_size=1):
    """
    Builds a PyTorch DataLoader for the COCO dataset.

    pin_memory: allocates the batch in page-locked host memory so the
      CUDAPrefetcher's non_blocking H2D copies actually overlap with compute
      instead of silently falling back to a synchronous copy.
    persistent_workers: keeps worker processes alive across epochs instead of
      tearing them down and re-forking (and re-running COCO() JSON parsing +
      image-list sorting in each worker) every epoch — meaningful on Kaggle
      where CPU cycles are the scarce resource.
    prefetch_factor: how many batches each worker stages ahead of time; only
      valid when num_workers > 0.
    is_distributed/rank/world_size: when is_distributed=True, wraps the
      dataset in a DistributedSampler so each process (GPU) sees a disjoint
      shard of the data instead of every process redundantly iterating the
      whole dataset. `batch_size` here is still the PER-PROCESS batch size —
      effective batch = batch_size * world_size * accum_steps.

    Returns:
        (dataloader, sampler) — sampler is None when is_distributed=False.
        Call sampler.set_epoch(epoch) at the start of every epoch when it's
        not None, or DistributedSampler will hand out the same shard order
        every epoch instead of reshuffling.
    """
    dataset = CocoDetection(img_dir, ann_file, transforms)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                      shuffle=shuffle, drop_last=drop_last)
        loader_kwargs['sampler'] = sampler
        # shuffle and sampler are mutually exclusive in DataLoader — shuffling
        # is handled by the sampler (via set_epoch) instead.
    else:
        loader_kwargs['shuffle'] = shuffle

    if num_workers > 0:
        # persistent_workers/prefetch_factor require num_workers > 0.
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader, sampler

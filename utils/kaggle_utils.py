"""
Kaggle environment helpers.

Kaggle's public COCO datasets ship under different folder layouts depending
on which dataset you attach, e.g.:

    /kaggle/input/coco-2017-dataset/coco2017/train2017/*.jpg
    /kaggle/input/coco-2017-dataset/coco2017/val2017/*.jpg
    /kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json
    /kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json

or occasionally nested one level differently. Rather than hardcode a single
path, we search /kaggle/input for the two annotation files and infer the
image directories from their location.
"""
import os
import glob


def is_kaggle():
    return os.path.exists('/kaggle/input') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


def find_coco_paths(search_root='/kaggle/input'):
    """
    Scans search_root for instances_train2017.json / instances_val2017.json
    and returns a dict of resolved paths, or None if not found.
    """
    if not os.path.isdir(search_root):
        return None

    train_ann_candidates = glob.glob(
        os.path.join(search_root, '**', 'instances_train2017.json'), recursive=True
    )
    val_ann_candidates = glob.glob(
        os.path.join(search_root, '**', 'instances_val2017.json'), recursive=True
    )

    if not train_ann_candidates or not val_ann_candidates:
        return None

    train_ann_file = train_ann_candidates[0]
    val_ann_file = val_ann_candidates[0]

    # annotations/ is typically a sibling of train2017/ and val2017/
    dataset_root = os.path.dirname(os.path.dirname(train_ann_file))

    def _resolve_img_dir(preferred_name):
        # Case 1: <dataset_root>/train2017
        direct = os.path.join(dataset_root, preferred_name)
        if os.path.isdir(direct):
            return direct
        # Case 2: some Kaggle uploads nest an extra folder, e.g. .../train2017/train2017
        nested = glob.glob(os.path.join(search_root, '**', preferred_name), recursive=True)
        dirs = [p for p in nested if os.path.isdir(p)]
        if dirs:
            # Prefer the one with the most files (actual image folder vs. empty stub)
            dirs.sort(key=lambda p: len(os.listdir(p)) if os.path.isdir(p) else 0, reverse=True)
            return dirs[0]
        return direct  # fall back, may not exist

    train_img_dir = _resolve_img_dir('train2017')
    val_img_dir = _resolve_img_dir('val2017')

    return {
        'train_img_dir': train_img_dir,
        'train_ann_file': train_ann_file,
        'val_img_dir': val_img_dir,
        'val_ann_file': val_ann_file,
    }


def patch_config_for_kaggle(cfg, logger=None):
    """
    If running on Kaggle, auto-detect COCO paths and rewrite cfg.data / cfg.output_dir
    in-place. Falls back to whatever the YAML already specifies if nothing is found.
    """
    if not is_kaggle():
        return cfg

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _log("Kaggle environment detected — auto-locating COCO dataset under /kaggle/input ...")
    paths = find_coco_paths()

    if paths is None:
        _log("WARNING: Could not auto-locate instances_train2017.json / instances_val2017.json "
             "under /kaggle/input. Make sure a COCO dataset is attached to this notebook "
             "(e.g. 'COCO 2017 Dataset' by awsaf49). Falling back to config-specified paths.")
    else:
        for k, v in paths.items():
            cfg['data'][k] = v
            _log(f"  {k} -> {v}")

    # /kaggle/input is read-only; checkpoints/logs must go to /kaggle/working
    if not str(cfg.get('output_dir', '')).startswith('/kaggle/working'):
        new_out = '/kaggle/working/outputs/rtdetr_mamba_r50'
        cfg['output_dir'] = new_out
        _log(f"  output_dir -> {new_out} (Kaggle working directory, downloadable/persisted as notebook output)")

    return cfg

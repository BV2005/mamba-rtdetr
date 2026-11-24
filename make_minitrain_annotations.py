"""Builds COCO-format annotations for COCO minitrain-25k from the OFFICIAL COCO 2017 annotations.

coco_minitrain_25k ships YOLO .txt labels only. Converting those (setup_and_train.py's yolo_to_coco) assumes the
YOLO class order and replaces COCO's mask areas with box areas, which changes the small / medium / large split of
the COCO evaluation. This script instead keeps the official annotations of the 25,000 minitrain images
(the image list in coco_minitrain_25k/train2017.txt): official category ids, mask-based areas and iscrowd flags,
so training and the evaluation on val2017 match standard COCO.

    # official annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    python make_minitrain_annotations.py --coco-json annotations/instances_train2017.json \
        --image-list coco_minitrain_25k/train2017.txt \
        --out data/coco_minitrain_25k/annotations/instances_train2017.json
    # validation: copy annotations/instances_val2017.json unchanged (minitrain's val2017 = COCO val2017, 5,000 images)
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-json", required=True, help="official instances_train2017.json")
    ap.add_argument("--image-list", required=True, help="coco_minitrain_25k/train2017.txt (one image path per line)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wanted = {os.path.basename(line.strip()) for line in open(a.image_list) if line.strip()}
    with open(a.coco_json) as f:
        coco = json.load(f)
    images = [im for im in coco["images"] if im["file_name"] in wanted]
    ids = {im["id"] for im in images}
    anns = [an for an in coco["annotations"] if an["image_id"] in ids]
    missing = wanted - {im["file_name"] for im in images}
    if missing:
        raise SystemExit(f"{len(missing)} listed images are not in {a.coco_json} (e.g. {sorted(missing)[:3]})")
    out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
    out.update(images=images, annotations=anns)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{a.out}: {len(images)} images, {len(anns)} annotations "
          f"({sum(an.get('iscrowd', 0) for an in anns)} crowd), {len(out.get('categories', []))} categories")


if __name__ == "__main__":
    main()

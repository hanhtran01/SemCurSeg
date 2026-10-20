"""SCGM: grounding annotations (ODVG JSON lines) and size thresholds for the training split.

Lesion boxes are read from a COCO box file or derived from binary masks as the bounding box of each
connected component (only the box is used afterwards). The size thresholds are calibrated on these
training boxes and saved, so that Stage 1 inference applies the same thresholds to predicted boxes.

Captions for validation and test images must not be built from their annotations; they are
produced from CG-TVCC predictions by tools/stage1_inference.py.

Example
    python tools/build_scgm_captions.py --image-dir data/polyp/train/images --mask-dir data/polyp/train/masks \
        --task polyp --output data/polyp/annotations/train_odvg.jsonl \
        --thresholds-out data/polyp/annotations/size_thresholds.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semcurseg.scgm.attributes import (TASKS, box_area_ratio, calibrate_size_thresholds,  # noqa: E402
                                       load_thresholds, odvg_record, save_thresholds, shape_word, size_word)

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def boxes_from_mask(mask_path, min_area_ratio=1e-4, min_pixels=10):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    h, w = mask.shape
    binary = (mask > 127).astype('uint8')
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(min_pixels, min_area_ratio * h * w)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
    return boxes, w, h


def boxes_from_coco(coco_path):
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {im['id']: im for im in coco['images']}
    boxes = {im['file_name']: [] for im in coco['images']}
    for ann in coco['annotations']:
        im = images[ann['image_id']]
        x, y, bw, bh = ann['bbox']
        boxes[im['file_name']].append([x, y, x + bw, y + bh])
    sizes = {im['file_name']: (im['width'], im['height']) for im in coco['images']}
    return boxes, sizes


def collect_boxes(args):
    """{image file name: (boxes, width, height)} for every image in --image-dir."""
    images = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(IMG_EXTS))
    out = {}
    if args.coco:
        coco_boxes, sizes = boxes_from_coco(args.coco)
        for name in images:
            if name not in sizes:
                print(f'[warn] {name} is not in {args.coco}; skipped')
                continue
            out[name] = (coco_boxes[name], *sizes[name])
        return out
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(args.mask_dir) if f.lower().endswith(IMG_EXTS)}
    for name in images:
        stem = os.path.splitext(name)[0]
        if stem not in masks:
            print(f'[warn] no mask for {name}; skipped')
            continue
        boxes, w, h = boxes_from_mask(os.path.join(args.mask_dir, masks[stem]), args.min_area_ratio)
        out[name] = (boxes, w, h)
    return out


def main():
    p = argparse.ArgumentParser(description='SCGM: training captions and size thresholds from lesion boxes')
    p.add_argument('--image-dir', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--mask-dir', help='binary masks; one box per connected component')
    src.add_argument('--coco', help='COCO file with the lesion boxes')
    p.add_argument('--task', default='polyp', choices=sorted(TASKS))
    p.add_argument('--output', required=True, help='ODVG JSON lines file')
    p.add_argument('--thresholds-out', help='where to save the calibrated size thresholds')
    p.add_argument('--thresholds', help='reuse saved thresholds instead of calibrating')
    p.add_argument('--p-small', type=float, default=33.0)
    p.add_argument('--p-medium', type=float, default=66.0)
    p.add_argument('--min-area-ratio', type=float, default=1e-4,
                   help='ignore mask components smaller than this fraction of the image')
    args = p.parse_args()

    data = collect_boxes(args)
    if args.thresholds:
        thr = load_thresholds(args.thresholds)
    else:
        ratios = [box_area_ratio(b, w, h) for boxes, w, h in data.values() for b in boxes]
        thr = calibrate_size_thresholds(ratios, args.p_small, args.p_medium)
        if args.thresholds_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.thresholds_out)), exist_ok=True)
            save_thresholds(thr, args.thresholds_out)
    print(f'size thresholds (box area / image area): small < {thr["small_max"]:.5f} '
          f'<= medium < {thr["medium_max"]:.5f} <= large')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sizes, shapes, n_neg = Counter(), Counter(), 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for name, (boxes, w, h) in data.items():
            rec = odvg_record(name, w, h, boxes, thr, args.task)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_neg += not boxes
            for b in boxes:
                sizes[size_word(b, w, h, thr)] += 1
                shapes[shape_word(b)] += 1
    n_les = sum(sizes.values())
    print(f'{len(data)} images ({n_neg} without lesions), {n_les} lesions -> {args.output}')
    print('size :', dict(sizes))
    print('shape:', dict(shapes))


if __name__ == '__main__':
    main()

"""Convert binary masks to a COCO box file (one box per connected component).

CG-TVCC is validated and tested on these boxes with the class name as the text prompt, so no
caption derived from the validation or test annotations is given to the model.

Example
    python tools/build_coco_boxes.py --image-dir data/polyp/val/images --mask-dir data/polyp/val/masks \
        --class-name polyp --output data/polyp/annotations/val_coco.json
"""
import argparse
import json
import os

import cv2

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def main():
    p = argparse.ArgumentParser(description='Binary masks to COCO boxes')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--mask-dir', required=True)
    p.add_argument('--class-name', default='polyp')
    p.add_argument('--output', required=True)
    p.add_argument('--min-area-ratio', type=float, default=1e-4)
    args = p.parse_args()

    masks = {os.path.splitext(f)[0]: f for f in os.listdir(args.mask_dir) if f.lower().endswith(IMG_EXTS)}
    images, annotations = [], []
    ann_id = 1
    for img_id, name in enumerate(sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(IMG_EXTS)), 1):
        stem = os.path.splitext(name)[0]
        if stem not in masks:
            print(f'[warn] no mask for {name}; skipped')
            continue
        mask = cv2.imread(os.path.join(args.mask_dir, masks[stem]), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        images.append({'id': img_id, 'file_name': name, 'width': int(w), 'height': int(h)})
        n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype('uint8'), connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < max(10, args.min_area_ratio * w * h):
                continue
            annotations.append({'id': ann_id, 'image_id': img_id, 'category_id': 1,
                                'bbox': [x, y, bw, bh], 'area': float(bw * bh), 'iscrowd': 0})
            ann_id += 1

    coco = {'images': images, 'annotations': annotations,
            'categories': [{'id': 1, 'name': args.class_name, 'supercategory': args.class_name}]}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f'{len(images)} images, {len(annotations)} boxes -> {args.output}')


if __name__ == '__main__':
    main()

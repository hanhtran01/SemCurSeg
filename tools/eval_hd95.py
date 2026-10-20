"""HD95 between predicted and ground-truth masks (Sec. 4.2), at the ground-truth resolution.

Predictions are matched to ground truths by file stem and resized (nearest neighbour) to the
ground-truth size. Pairs in which either mask is empty are skipped and counted.

Example
    python tools/eval_hd95.py --gt data/polyp/test/masks --pred work_dirs/segmentor_polyp/pred_test
"""
import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))   # repository root

from semcurseg.metrics import hd95  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def read_mask(path, size_wh=None):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if size_wh is not None and (m.shape[1], m.shape[0]) != size_wh:
        m = cv2.resize(m, size_wh, interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description='HD95 between predicted and ground-truth masks')
    p.add_argument('--gt', required=True)
    p.add_argument('--pred', required=True)
    args = p.parse_args()

    preds = {osp.splitext(f)[0]: f for f in os.listdir(args.pred) if f.lower().endswith(IMG_EXTS)}
    values, skipped, missing = [], 0, 0
    for name in sorted(f for f in os.listdir(args.gt) if f.lower().endswith(IMG_EXTS)):
        stem = osp.splitext(name)[0]
        if stem not in preds:
            missing += 1
            continue
        gt = read_mask(osp.join(args.gt, name))
        value = hd95(gt, read_mask(osp.join(args.pred, preds[stem]), (gt.shape[1], gt.shape[0])))
        if value is None:
            skipped += 1
        else:
            values.append(value)
    print(f'HD95 = {np.mean(values):.2f} px over {len(values)} images '
          f'({skipped} skipped with an empty mask, {missing} without prediction)')


if __name__ == '__main__':
    main()

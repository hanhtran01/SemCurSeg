"""Evaluate the Stage 2 network on the test split: mIoU and mDice over images (Sec. 4.2).

Captions and pseudo masks of the test images come from tools/stage1_inference.py; the ground-truth
masks are used only for the metrics. Predictions are also written at the original image size, for
tools/eval_hd95.py.

Example
    python tools/test_segmentor.py --images data/polyp/test/images --masks data/polyp/test/masks \
        --pseudo data/polyp/stage1/test/masks --captions data/polyp/stage1/test/captions.json \
        --checkpoint work_dirs/segmentor_polyp/best_dice.pth --save-dir work_dirs/segmentor_polyp/pred_test
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))   # repository root

from semcurseg.metrics import overlap_metrics  # noqa: E402
from semcurseg.segmentor import SegmentationDataset, SemCurSegNet  # noqa: E402


def get_args():
    p = argparse.ArgumentParser('SemCurSeg Stage 2 evaluation')
    p.add_argument('--images', required=True)
    p.add_argument('--masks', required=True, help='ground-truth masks (metrics only)')
    p.add_argument('--pseudo', required=True, help='pseudo masks from tools/stage1_inference.py')
    p.add_argument('--captions', required=True, help='captions.json from tools/stage1_inference.py')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--no-apfo', dest='use_apfo', action='store_false')
    p.add_argument('--input-size', type=int, default=384)
    p.add_argument('--max-length', type=int, default=64)
    p.add_argument('--text-encoder', default='bert-base-uncased')
    p.add_argument('--vision-encoder', default='facebook/convnext-tiny-224')
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = SegmentationDataset(args.images, args.masks, args.captions, 'test', pseudo_dir=args.pseudo,
                                  size=args.input_size, tokenizer=args.text_encoder, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SemCurSegNet(args.text_encoder, args.vision_encoder, use_apfo=args.use_apfo).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    per_image = {}
    for batch in tqdm(loader, desc='test'):
        name = batch['name'][0]
        text = {k: v.to(device) for k, v in batch['text'].items()}
        prob = model(batch['image'].to(device), text, batch['pseudo'].to(device))[0, 0].cpu().numpy()
        per_image[name] = overlap_metrics(batch['mask'][0, 0].numpy() > 0.5, prob > 0.5)
        h, w = cv2.imread(osp.join(args.images, name), cv2.IMREAD_GRAYSCALE).shape
        pred = cv2.resize((prob > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(osp.join(args.save_dir, osp.splitext(name)[0] + '.png'), pred)

    mean = {k: float(np.mean([m[k] for m in per_image.values()])) for k in next(iter(per_image.values()))}
    with open(osp.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'mean': mean, 'per_image': per_image}, f, indent=1)
    print(f'{len(per_image)} images | mIoU {100 * mean["iou"]:.2f} | mDice {100 * mean["dice"]:.2f} | '
          f'recall {100 * mean["recall"]:.2f} | precision {100 * mean["precision"]:.2f}')


if __name__ == '__main__':
    main()

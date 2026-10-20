"""Stage 1 inference: CG-TVCC boxes -> SAM Adapt pseudo masks and SCGM captions.

CG-TVCC is prompted with the class name only ("polyp" or "tumor"). The retained boxes are segmented
by SAM Adapt, and the caption given to the Stage 2 network is built from the predicted boxes with
the size thresholds calibrated on the training boxes. No annotation of the processed images is read,
so the same command serves the training, validation and test splits.

Outputs
    <out-dir>/masks/<image stem>.png   pseudo mask, union over the retained boxes (0/255)
    <out-dir>/captions.json            {image file name: caption}
    <out-dir>/boxes.json               {image file name: [[x1, y1, x2, y2, score], ...]}

Example
    python tools/stage1_inference.py --config configs/cgtvcc_polyp.py \
        --checkpoint work_dirs/cgtvcc_polyp/best_coco_bbox_mAP_epoch_N.pth \
        --sam-adapt work_dirs/sam_adapt_polyp/best.pth --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --thresholds data/polyp/annotations/size_thresholds.json \
        --image-dir data/polyp/test/images --out-dir data/polyp/stage1/test
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))   # repository root

from mmcv.transforms import Compose  # noqa: E402
from mmdet.apis import init_detector  # noqa: E402
from mmdet.utils import get_test_pipeline_cfg  # noqa: E402
from torchvision.ops import nms  # noqa: E402

from semcurseg.sam_adapt import SAMAdapt  # noqa: E402
from semcurseg.scgm.attributes import TASKS, caption_from_boxes, load_thresholds  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


class Grounder:
    """CG-TVCC prompted with the class name."""

    def __init__(self, config, checkpoint, device):
        self.model = init_detector(config, checkpoint, device=device)
        pipeline = [t for t in get_test_pipeline_cfg(self.model.cfg) if t['type'] != 'LoadAnnotations']
        self.pipeline = Compose(pipeline)

    @torch.no_grad()
    def __call__(self, img_path, prompt):
        data = self.pipeline(dict(img_path=img_path, img_id=0, text=prompt, custom_entities=True,
                                  tokens_positive=None))
        pred = self.model.test_step(dict(inputs=[data['inputs']], data_samples=[data['data_samples']]))[0]
        return pred.pred_instances.bboxes.float().cpu(), pred.pred_instances.scores.float().cpu()


def select_boxes(boxes, scores, score_thr, iou_thr, max_boxes, keep_top1):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0, ))
    keep = nms(boxes, scores, iou_thr)                 # sorted by score
    boxes, scores = boxes[keep], scores[keep]
    sel = scores >= score_thr
    if keep_top1 and not sel.any():
        sel[0] = True
    return boxes[sel][:max_boxes], scores[sel][:max_boxes]


def load_sam(path, sam_checkpoint, sam_type, device):
    """SAM Adapt (pretrained SAM image encoder + the parameters trained by tools/train_sam_adapt.py),
    or the original SAM when no SAM Adapt checkpoint is given."""
    if path is None:
        return SAMAdapt(sam_type, sam_checkpoint, False, False, False, False).to(device).eval()
    ckpt = torch.load(path, map_location='cpu')
    a = ckpt['args']
    model = SAMAdapt(a['sam_type'], sam_checkpoint, a['use_hcae'], a['use_edge_prompt'],
                     a['use_uncertainty'], a['use_nbis'])
    return model.load_trained(ckpt['model_state']).to(device).eval()


def main():
    p = argparse.ArgumentParser(description='Stage 1: pseudo masks and captions from predicted boxes')
    p.add_argument('--config', required=True, help='CG-TVCC config')
    p.add_argument('--checkpoint', required=True, help='CG-TVCC checkpoint')
    p.add_argument('--sam-adapt', help='best.pth written by tools/train_sam_adapt.py; '
                                       'omit to use the original SAM (no pixel-level masks)')
    p.add_argument('--sam-checkpoint', required=True, help='pretrained SAM weights, e.g. sam_vit_h_4b8939.pth')
    p.add_argument('--sam-type', default='vit_h', help='SAM variant when --sam-adapt is omitted')
    p.add_argument('--thresholds', required=True, help='size thresholds written by tools/build_scgm_captions.py')
    p.add_argument('--task', default='polyp', choices=sorted(TASKS))
    p.add_argument('--image-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--score-thr', type=float, default=0.3)
    p.add_argument('--nms-iou', type=float, default=0.5)
    p.add_argument('--max-boxes', type=int, default=3)
    p.add_argument('--keep-top1', action='store_true',
                   help='keep the best box when none passes --score-thr (data in which every image has a lesion)')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    thresholds = load_thresholds(args.thresholds)
    prompt = TASKS[args.task]['object']
    grounder = Grounder(args.config, args.checkpoint, args.device)
    sam = load_sam(args.sam_adapt, args.sam_checkpoint, args.sam_type, args.device)

    mask_dir = osp.join(args.out_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    captions, detections, n_empty = {}, {}, 0
    names = sorted(f for f in os.listdir(args.image_dir) if f.lower().endswith(IMG_EXTS))
    for name in tqdm(names, desc='stage 1'):
        path = osp.join(args.image_dir, name)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        boxes, scores = select_boxes(*grounder(path, prompt), args.score_thr, args.nms_iou, args.max_boxes,
                                     args.keep_top1)
        mask = np.zeros((h, w), np.uint8)
        if len(boxes):
            mask[sam.predict(image, boxes.numpy()).any(dim=0).numpy()] = 255
        else:
            n_empty += 1
        cv2.imwrite(osp.join(mask_dir, osp.splitext(name)[0] + '.png'), mask)
        captions[name] = caption_from_boxes(boxes.tolist(), w, h, thresholds, args.task)
        detections[name] = [b + [s] for b, s in zip(boxes.tolist(), scores.tolist())]

    with open(osp.join(args.out_dir, 'captions.json'), 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=1, ensure_ascii=False)
    with open(osp.join(args.out_dir, 'boxes.json'), 'w', encoding='utf-8') as f:
        json.dump(detections, f)
    print(f'{len(names)} images, {n_empty} without a retained box -> {args.out_dir}')


if __name__ == '__main__':
    main()

"""Train the Stage 2 network (APFO + TVHA) on the Stage 1 pseudo masks (Sec. 3.6-3.8 and 4.3).

Training targets and captions are those written by tools/stage1_inference.py for the training
images. The validation ground-truth masks are used only to select the checkpoint; the validation
captions and pseudo masks also come from tools/stage1_inference.py.

Example
    python tools/train_segmentor.py \
        --train-images data/polyp/train/images --train-pseudo data/polyp/stage1/train/masks \
        --train-captions data/polyp/stage1/train/captions.json \
        --val-images data/polyp/val/images --val-masks data/polyp/val/masks \
        --val-pseudo data/polyp/stage1/val/masks --val-captions data/polyp/stage1/val/captions.json \
        --out-dir work_dirs/segmentor_polyp
"""
import argparse
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))   # repository root

from semcurseg.metrics import overlap_metrics  # noqa: E402
from semcurseg.segmentor import SegmentationDataset, SemCurSegNet, dice_bce_loss  # noqa: E402


def get_args():
    p = argparse.ArgumentParser('SemCurSeg Stage 2 training')
    p.add_argument('--train-images', required=True)
    p.add_argument('--train-pseudo', required=True, help='Stage 1 pseudo masks of the training images (targets)')
    p.add_argument('--train-captions', required=True, help='captions.json of the training images')
    p.add_argument('--val-images', required=True)
    p.add_argument('--val-masks', required=True, help='ground-truth masks of the validation images')
    p.add_argument('--val-pseudo', required=True, help='Stage 1 pseudo masks of the validation images')
    p.add_argument('--val-captions', required=True)
    p.add_argument('--no-apfo', dest='use_apfo', action='store_false')
    p.add_argument('--input-size', type=int, default=384)
    p.add_argument('--max-length', type=int, default=64, help='caption length in tokens')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda-dice', type=float, default=1.0)
    p.add_argument('--lambda-bce', type=float, default=1.0)
    p.add_argument('--text-encoder', default='bert-base-uncased')
    p.add_argument('--vision-encoder', default='facebook/convnext-tiny-224')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--wandb-project', help='log to Weights & Biases under this project')
    return p.parse_args()


def run_epoch(model, loader, args, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, scores = 0.0, []
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc='train' if train else 'val'):
            image, mask, pseudo = (batch[k].to(device) for k in ('image', 'mask', 'pseudo'))
            text = {k: v.to(device) for k, v in batch['text'].items()}
            prob = model(image, text, pseudo)
            loss = dice_bce_loss(prob, mask, args.lambda_dice, args.lambda_bce)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            scores += [overlap_metrics(m > 0.5, p > 0.5)
                       for m, p in zip(mask.cpu().numpy(), prob.detach().cpu().numpy())]
    return total / len(loader), {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(osp.join(args.out_dir, 'train_log.txt'), 'a', encoding='utf-8')

    def report(msg):
        print(msg)
        log.write(msg + '\n')
        log.flush()

    report(str(vars(args)))
    common = dict(size=args.input_size, tokenizer=args.text_encoder, max_length=args.max_length)
    # the training target is the pseudo mask, which is also the APFO input
    train_set = SegmentationDataset(args.train_images, args.train_pseudo, args.train_captions, 'train',
                                    pseudo_dir=args.train_pseudo, **common)
    val_set = SegmentationDataset(args.val_images, args.val_masks, args.val_captions, 'val',
                                  pseudo_dir=args.val_pseudo, **common)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SemCurSegNet(args.text_encoder, args.vision_encoder, use_apfo=args.use_apfo).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    wandb = None
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=osp.basename(osp.normpath(args.out_dir)), config=vars(args))

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_m = run_epoch(model, train_loader, args, device, optimizer)
        val_loss, val_m = run_epoch(model, val_loader, args, device)
        scheduler.step(val_loss)
        if val_m['dice'] > best:
            best = val_m['dice']
            torch.save(model.state_dict(), osp.join(args.out_dir, 'best_dice.pth'))
            report(f'validation Dice improved to {best:.4f}; checkpoint saved')
        report(f'epoch {epoch:03d} ({time.time() - start:.0f}s) | train loss {train_loss:.4f} '
               f'dice {train_m["dice"]:.4f} iou {train_m["iou"]:.4f} | val loss {val_loss:.4f} '
               f'dice {val_m["dice"]:.4f} iou {val_m["iou"]:.4f}')
        if wandb is not None:
            wandb.log({'train/loss': train_loss, 'val/loss': val_loss,
                       **{f'train/{k}': v for k, v in train_m.items()},
                       **{f'val/{k}': v for k, v in val_m.items()}})
    if wandb is not None:
        wandb.finish()
    log.close()


if __name__ == '__main__':
    main()

"""Train SAM Adapt on the training images that have ground-truth masks (Sec. 3.5 and 4.3).

The SAM image encoder and prompt encoder stay frozen; the mask decoder, HCAE, the edge prompt
generator, the uncertainty head and the NBIS hard-region refinement are trained with

    loss = Dice + BCE + 0.1 * uncertainty loss.

Only a random subset of the training images (--mask-ratio, 20% in the paper) is used, of which
--val-ratio (20%) is held out for validation. Each connected component of a mask is one sample with
its own box prompt, as at inference where every predicted box is prompted separately. The images
used are listed in <out-dir>/mask_subset.txt. Checkpoints contain the trained parameters only; the
frozen encoder is restored from the SAM checkpoint.

Example
    python tools/train_sam_adapt.py --image-dir data/polyp/train/images --mask-dir data/polyp/train/masks \
        --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth --mask-ratio 0.2 --out-dir work_dirs/sam_adapt_polyp
"""
import argparse
import json
import os
import os.path as osp
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))   # repository root

from semcurseg.sam_adapt import HFNMTransform, SAMAdapt, UncertaintyEstimator  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def list_pairs(image_dir, mask_dir):
    masks = {osp.splitext(f)[0]: f for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXTS)}
    return [(osp.join(image_dir, f), osp.join(mask_dir, masks[osp.splitext(f)[0]]))
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(IMG_EXTS) and osp.splitext(f)[0] in masks]


def read_mask(path, size):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize((m > 127).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


class LesionDataset(Dataset):
    """One sample per lesion (connected component) of the selected masked images."""

    def __init__(self, pairs, train=True, size=352, hfnm_prob=0.5, min_pixels=16):
        self.train, self.size = train, size
        self.samples = []
        for img_path, mask_path in pairs:
            n, labels = cv2.connectedComponents(read_mask(mask_path, size))
            self.samples += [(img_path, mask_path, k) for k in range(1, n) if (labels == k).sum() >= min_pixels]
        self.hfnm = HFNMTransform(noise_std=0.1, noise_prob=hfnm_prob) if train and hfnm_prob > 0 else None

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _augment(image, mask):
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)
        if random.random() < 0.35:
            h, w = image.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-12, 12), random.uniform(0.95, 1.05))
            image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if random.random() < 0.3:
            image = np.clip(image.astype(np.float32) * random.uniform(0.9, 1.1) + random.uniform(-12, 12),
                            0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path, k = self.samples[idx]
        image = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (self.size, self.size))
        _, labels = cv2.connectedComponents(read_mask(mask_path, self.size))
        mask = (labels == k).astype(np.uint8)
        if self.train:
            image, mask = self._augment(image, mask)
            if self.hfnm is not None:
                image, _ = self.hfnm(image)
        ys, xs = np.nonzero(mask)
        box = (np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32) if len(xs)
               else np.array([0, 0, self.size - 1, self.size - 1], np.float32))
        return {'image': image, 'mask': mask.astype(np.float32), 'box': box}


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0]}


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def enlarge_box(box, ratio, size):
    """Enlarge a box by a random margin of up to `ratio` of its size on each side."""
    x1, y1, x2, y2 = box
    m = np.random.rand(4) * ratio
    return np.array([max(0, x1 - (x2 - x1) * m[0]), max(0, y1 - (y2 - y1) * m[1]),
                     min(size - 1, x2 + (x2 - x1) * m[2]), min(size - 1, y2 + (y2 - y1) * m[3])], np.float32)


def lesion_loss(model, logits, target, unc_weight):
    prob = torch.sigmoid(logits)
    dice = 1 - (2 * (prob * target).sum() + 1e-6) / (prob.sum() + target.sum() + 1e-6)
    loss = dice + F.binary_cross_entropy_with_logits(logits, target)
    if unc_weight > 0 and model.uncertainty is not None:
        _, _, alpha = model.uncertainty.estimator(logits)
        loss = loss + unc_weight * UncertaintyEstimator.loss(alpha, target)
    return loss


def overlap(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter, total = (pred * target).sum(), pred.sum() + target.sum()
    return (2 * inter / (total + 1e-6)).item(), (inter / (total - inter + 1e-6)).item()


def train_one_epoch(model, loader, optimizer, args, device):
    model.train()
    totals = np.zeros(2)
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc='train')
    for step, batch in enumerate(pbar, 1):
        losses, dices = [], []
        for image, mask, box in zip(batch['image'], batch['mask'], batch['box']):
            logits = model.logits_at(image, enlarge_box(box, args.box_jitter, image.shape[0])[None])
            target = torch.from_numpy(mask)[None, None].to(device)
            losses.append(lesion_loss(model, logits, target, args.unc_weight))
            dices.append(overlap(logits.detach(), target)[0])
        loss = torch.stack(losses).mean()
        (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        totals += [loss.item(), np.mean(dices)]
        pbar.set_postfix(loss=f'{totals[0] / step:.4f}', dice=f'{totals[1] / step:.4f}')
    return totals / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = [overlap(model.logits_at(image, box[None]), torch.from_numpy(mask)[None, None].to(device))
              for batch in tqdm(loader, desc='val')
              for image, mask, box in zip(batch['image'], batch['mask'], batch['box'])]
    return np.mean(scores, axis=0)


def main():
    p = argparse.ArgumentParser(description='Train SAM Adapt')
    p.add_argument('--image-dir', required=True)
    p.add_argument('--mask-dir', required=True)
    p.add_argument('--mask-ratio', type=float, default=0.2, help='fraction of training images whose masks are used')
    p.add_argument('--subset-list', help='use the image names in this file instead of a random subset')
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--sam-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    p.add_argument('--sam-checkpoint', required=True)
    p.add_argument('--no-hcae', dest='use_hcae', action='store_false')
    p.add_argument('--no-edge-prompt', dest='use_edge_prompt', action='store_false')
    p.add_argument('--no-uncertainty', dest='use_uncertainty', action='store_false')
    p.add_argument('--no-nbis', dest='use_nbis', action='store_false')
    p.add_argument('--no-hfnm', dest='use_hfnm', action='store_false')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--box-jitter', type=float, default=0.1)
    p.add_argument('--unc-weight', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=6, help='early stopping on validation Dice')
    p.add_argument('--input-size', type=int, default=352)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # masked subset and validation split, by image
    pairs = list_pairs(args.image_dir, args.mask_dir)
    if args.subset_list:
        keep = {line.strip() for line in open(args.subset_list, encoding='utf-8') if line.strip()}
        subset = [pr for pr in pairs if osp.basename(pr[0]) in keep]
    else:
        subset = random.Random(args.seed).sample(pairs, max(1, round(len(pairs) * args.mask_ratio)))
    with open(osp.join(args.out_dir, 'mask_subset.txt'), 'w', encoding='utf-8') as f:
        f.writelines(osp.basename(img) + '\n' for img, _ in sorted(subset))
    order = list(range(len(subset)))
    random.Random(args.seed + 1).shuffle(order)
    n_val = max(1, round(len(subset) * args.val_ratio))
    train_set = LesionDataset([subset[i] for i in order[n_val:]], True, args.input_size,
                              0.5 if args.use_hfnm else 0.0)
    val_set = LesionDataset([subset[i] for i in order[:n_val]], False, args.input_size)
    print(f'masks of {len(subset)}/{len(pairs)} training images: {len(subset) - n_val} for training '
          f'({len(train_set)} lesions), {n_val} for validation ({len(val_set)} lesions)')
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers, collate_fn=collate)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = SAMAdapt(args.sam_type, args.sam_checkpoint, args.use_hcae, args.use_edge_prompt,
                     args.use_uncertainty, args.use_nbis).to(device)
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    print(f'trainable parameters: {sum(p_.numel() for p_ in trainable) / 1e6:.2f} M')
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best, stale = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, args, device)
        val_dice, val_iou = validate(model, val_loader, device)
        scheduler.step(val_dice)
        print(f'epoch {epoch}: train loss {tr_loss:.4f} dice {tr_dice:.4f} | val dice {val_dice:.4f} iou {val_iou:.4f}')
        ckpt = {'epoch': epoch, 'val_dice': float(val_dice), 'model_state': model.trained_state(), 'args': vars(args)}
        torch.save(ckpt, osp.join(args.out_dir, 'latest.pth'))
        if val_dice > best:
            best, stale = val_dice, 0
            torch.save(ckpt, osp.join(args.out_dir, 'best.pth'))
        else:
            stale += 1
            if stale >= args.patience:
                print(f'early stopping after {epoch} epochs')
                break
    with open(osp.join(args.out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'best_val_dice': float(best)}, f, indent=2)
    print(f'best validation Dice {best:.4f}')


if __name__ == '__main__':
    main()

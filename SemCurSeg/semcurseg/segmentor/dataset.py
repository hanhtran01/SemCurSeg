"""Stage 2 dataset: image, target mask, Stage 1 pseudo mask and caption.

    training      target = Stage 1 pseudo mask; APFO receives the same pseudo mask
    val / test    target = ground-truth mask (evaluation only); APFO receives the Stage 1 pseudo mask

Captions come from the captions.json written by tools/stage1_inference.py ({image file name: caption});
an ODVG JSON lines file or a fixed string can also be given. Adapted from SimTxtSeg (Xie et al.,
MICCAI 2024), Apache-2.0; files are matched by name rather than by sorted order.
"""
import json
import os
from functools import lru_cache

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# Sec. 4.3: rotation up to 35 degrees, horizontal and vertical flips, coarse dropout, each p = 0.3
TRAIN_AUG = A.Compose([
    A.Rotate(limit=35, p=0.3),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
], additional_targets={'pseudo': 'mask'})


@lru_cache(maxsize=4)
def get_tokenizer(name):
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def load_captions(source):
    """{file name: caption} from captions.json or an ODVG JSON lines file; None if `source` is not a file."""
    if not os.path.isfile(source):
        return None
    with open(source, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None                                   # JSON lines
    if isinstance(data, dict) and 'grounding' not in data:
        return {k: str(v) for k, v in data.items()}
    records = [data] if isinstance(data, dict) else [json.loads(l) for l in content.splitlines() if l.strip()]
    return {rec['filename']: rec['grounding']['caption'] for rec in records}


def _index(directory):
    return {os.path.splitext(f)[0]: os.path.join(directory, f)
            for f in os.listdir(directory) if f.lower().endswith(IMG_EXTS)}


class SegmentationDataset(Dataset):
    """Returns image (3, S, S) in [0, 1], target (1, S, S), pseudo mask (1, S, S), tokenised caption
    and, for split='test', the image file name."""

    def __init__(self, image_dir, mask_dir, captions, split, pseudo_dir=None, size=384,
                 tokenizer='bert-base-uncased', max_length=64):
        super().__init__()
        masks = _index(mask_dir)
        pseudos = _index(pseudo_dir) if pseudo_dir else None
        self.items = []
        for name in sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)):
            stem = os.path.splitext(name)[0]
            if stem not in masks:
                raise FileNotFoundError(f'no mask for {name} in {mask_dir}')
            if pseudos is not None and stem not in pseudos:
                raise FileNotFoundError(f'no Stage 1 pseudo mask for {name} in {pseudo_dir}')
            self.items.append((name, os.path.join(image_dir, name), masks[stem],
                               pseudos[stem] if pseudos is not None else None))

        self.captions = load_captions(captions)
        self.fixed_text = captions if self.captions is None else None
        if self.captions is not None:
            missing = [n for n, *_ in self.items if n not in self.captions]
            if missing:
                raise KeyError(f'{len(missing)} images have no caption in {captions}, e.g. {missing[:3]}')

        self.split = split
        self.size = (size, size)
        self.max_length = max_length
        self.tokenizer = get_tokenizer(tokenizer)

    def __len__(self):
        return len(self.items)

    def _tokenize(self, text):
        out = self.tokenizer.encode_plus(text, padding='max_length', max_length=self.max_length,
                                         truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': out['input_ids'].squeeze(0), 'attention_mask': out['attention_mask'].squeeze(0)}

    @staticmethod
    def _read_mask(path, hw):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(path)
        if m.shape != hw:
            m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        return (m > 127).astype(np.uint8)

    def __getitem__(self, index):
        name, img_path, mask_path, pseudo_path = self.items[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hw = image.shape[:2]
        mask = self._read_mask(mask_path, hw)
        pseudo = self._read_mask(pseudo_path, hw) if pseudo_path else np.zeros(hw, np.uint8)

        if self.split == 'train':
            aug = TRAIN_AUG(image=image, mask=mask, pseudo=pseudo)
            image, mask, pseudo = aug['image'], aug['mask'], aug['pseudo']

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        pseudo = cv2.resize(pseudo, self.size, interpolation=cv2.INTER_NEAREST)

        sample = {
            'image': np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0,
            'mask': mask[None].astype(np.float32),
            'pseudo': pseudo[None].astype(np.float32),
            'text': self._tokenize(self.captions[name] if self.captions is not None else self.fixed_text),
        }
        if self.split == 'test':
            sample['name'] = name
        return sample

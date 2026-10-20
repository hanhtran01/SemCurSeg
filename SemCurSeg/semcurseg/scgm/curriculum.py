"""SCGM curriculum for CG-TVCC training (Sec. 3.3.2), as MMDetection components.

CurriculumODVGDataset   ODVG grounding data whose captions follow the current phrase level
CurriculumPhraseHook    object-level phrases for epochs 0-9, attribute-level for 10-24, spatial after
"""
import json
import os.path as osp

from mmengine.fileio import get_local_path
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS, HOOKS

from .attributes import compose_caption


@DATASETS.register_module()
class CurriculumODVGDataset(BaseDetDataset):
    """Grounding dataset in ODVG format with SCGM phrase levels.

    A region may carry `phrases`, the object-, attribute- and spatial-level phrase of one lesion,
    each contained in the next. At phrase level L the caption is rebuilt from the level-L phrase of
    every lesion and each box is aligned with the tokens of its level-L phrase, so lesions in the
    same image keep separate positive maps (Sec. 3.3.3). The spans of levels 0..L are also returned
    as `level_tokens_positive` for the phrase-specific loss weighting. Regions with a single
    `phrase` are treated as one-level phrase sets.
    """

    def __init__(self, *args, max_phrase_level=2, **kwargs):
        self.phrase_level = int(max_phrase_level)
        super().__init__(*args, **kwargs)

    def set_phrase_level(self, level):
        self.phrase_level = int(level)

    def load_data_list(self):
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f if line.strip()]

        data_list = []
        for rec in records:
            width, height = rec['width'], rec['height']
            grounding = rec['grounding']
            instances, phrase_sets = [], []
            for label, region in enumerate(grounding.get('regions', [])):
                bboxes = region['bbox'] if isinstance(region['bbox'][0], (list, tuple)) else [region['bbox']]
                for x1, y1, x2, y2 in bboxes:
                    inter_w = max(0, min(x2, width) - max(x1, 0))
                    inter_h = max(0, min(y2, height) - max(y1, 0))
                    if inter_w * inter_h == 0 or (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue
                    instances.append(dict(bbox=[x1, y1, x2, y2], bbox_label=label, ignore_flag=0))
                phrase_sets.append(list(region.get('phrases') or [region['phrase']]))
            data_list.append(dict(
                img_path=osp.join(self.data_prefix['img'], rec['filename']),
                height=height,
                width=width,
                instances=instances,
                phrase_sets=phrase_sets,
                fallback_caption=grounding['caption'],
                dataset_mode='VG',
            ))
        return data_list

    def get_data_info(self, idx):
        info = super().get_data_info(idx)
        phrase_sets = info.pop('phrase_sets')
        fallback = info.pop('fallback_caption')
        if not phrase_sets:                       # image without lesions
            info.update(text=fallback, phrases={}, level_tokens_positive={})
            return info
        caption, spans = compose_caption(phrase_sets, self.phrase_level)
        info['text'] = caption
        info['phrases'] = {
            label: dict(phrase=phrases[min(self.phrase_level, len(phrases) - 1)],
                        tokens_positive=spans[label][-1])
            for label, phrases in enumerate(phrase_sets)
        }
        info['level_tokens_positive'] = {label: spans[label] for label in range(len(phrase_sets))}
        return info


def _curriculum_datasets(dataset):
    if isinstance(dataset, CurriculumODVGDataset):
        yield dataset
    if hasattr(dataset, 'dataset'):
        yield from _curriculum_datasets(dataset.dataset)
    for child in getattr(dataset, 'datasets', []):
        yield from _curriculum_datasets(child)


@HOOKS.register_module()
class CurriculumPhraseHook(Hook):
    """Object-level phrases for epochs [0, ep1), attribute-level for [ep1, ep2), spatial-level after.

    The level is passed to every CurriculumODVGDataset of the training loader and to the detector
    (for the phrase-specific loss weighting). Persistent dataloader workers hold their own copy of
    the dataset, so they are restarted when the level changes.
    """

    priority = 'NORMAL'

    def __init__(self, ep1=10, ep2=25):
        if not 0 <= ep1 <= ep2:
            raise ValueError(f'expected 0 <= ep1 <= ep2, got {ep1}, {ep2}')
        self.ep1, self.ep2 = ep1, ep2
        self._level = None

    def level_at(self, epoch):
        if epoch < self.ep1:
            return 0
        return 1 if epoch < self.ep2 else 2

    def before_train_epoch(self, runner):
        level = self.level_at(runner.epoch)
        if level == self._level:
            return
        datasets = list(_curriculum_datasets(runner.train_dataloader.dataset))
        if not datasets:
            runner.logger.warning('CurriculumPhraseHook: no CurriculumODVGDataset in the training loader')
        for ds in datasets:
            ds.set_phrase_level(level)
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        if hasattr(model, 'set_curriculum_level'):
            model.set_curriculum_level(level)

        loader = runner.train_dataloader
        iterator = getattr(loader, '_iterator', None)
        if getattr(loader, 'persistent_workers', False) and iterator is not None:
            try:
                iterator._shutdown_workers()
            finally:
                loader._iterator = None
        self._level = level
        runner.logger.info(f'SCGM curriculum: phrase level {level} from epoch {runner.epoch}')

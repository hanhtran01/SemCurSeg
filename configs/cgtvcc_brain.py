# CG-TVCC for brain tumour grounding on the LGG MRI dataset; same protocol as the polyp config.
_base_ = ['./cgtvcc_polyp.py']

data_root = 'data/brain/'
class_name = ('tumor', )
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo))
test_dataloader = dict(dataset=dict(data_root=data_root, metainfo=metainfo))
val_evaluator = dict(ann_file=data_root + 'annotations/val_coco.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test_coco.json')

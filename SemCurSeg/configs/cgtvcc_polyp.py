# CG-TVCC (Curriculum-Guided Textual-to-Visual Cue Converter, Sec. 3.4) for polyp grounding:
# Grounding DINO Swin-T trained on SCGM curriculum captions (Sec. 3.3), MMDetection 3.3.0.
# Paths are relative to the repository root, from which all commands are run.
_base_ = [
    '_base_/coco_detection.py',
    '_base_/schedule_1x.py',
    '_base_/default_runtime.py',
]

custom_imports = dict(imports=['semcurseg.scgm.curriculum', 'semcurseg.scgm.phrase_weighting'],
                      allow_failed_imports=False)

# ---- paths -------------------------------------------------------------------------------------
data_root = 'data/polyp/'
train_ann = 'annotations/train_odvg.jsonl'        # tools/build_scgm_captions.py
val_ann = 'annotations/val_coco.json'             # tools/build_coco_boxes.py
test_ann = 'annotations/test_coco.json'
pretrained = 'checkpoints/swin_tiny_patch4_window7_224.pth'
load_from = ('checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_'
             '20231204_095047-b448804b.pth')

lang_model_name = 'bert-base-uncased'
class_name = ('polyp', )
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

# ---- model -------------------------------------------------------------------------------------
model = dict(
    type='CurriculumGroundingDINO',
    use_phrase_weighting=True,
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        fusion_layer_cfg=dict(v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='PhraseWeightedGroundingDINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        loss_cls=dict(
            type='PhraseWeightedFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            num_levels=3,
            acc_momentum=0.99),
        loss_bbox=dict(type='L1Loss', loss_weight=8.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=8.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=4.0)
            ])),
    test_cfg=dict(max_per_img=300))

# ---- data --------------------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        hue_delta=18),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(320, 1333), (480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333), (672, 1333),
                            (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)],
                     keep_ratio=True),
                dict(type='RandomCrop', crop_type='absolute_range', crop_size=(128, 256),
                     allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomSamplingNegPos', tokenizer_name=lang_model_name, num_sample_negative=15,
         max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip',
                   'flip_direction', 'text', 'custom_entities', 'tokens_positive',
                   'level_tokens_positive', 'dataset_mode'))
]

# Validation and test use the class name as the only text prompt and the boxes of the COCO files;
# no caption derived from validation or test annotations reaches the model.
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, imdecode_backend='pillow'),
    dict(type='FixScaleResize', scale=(1024, 1333), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'tokens_positive'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='CurriculumODVGDataset',
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        max_phrase_level=0,
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        return_classes=True,
        ann_file=val_ann,
        data_prefix=dict(img='val/images/'),
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        return_classes=True,
        ann_file=test_ann,
        data_prefix=dict(img='test/images/'),
        pipeline=test_pipeline))

val_evaluator = dict(type='CocoMetric', ann_file=data_root + val_ann, metric='bbox', format_only=False)
test_evaluator = dict(type='CocoMetric', ann_file=data_root + test_ann, metric='bbox', format_only=False)

# ---- optimisation (Sec. 4.3) -------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
    }))

max_epoch = 100
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epoch, by_epoch=True, milestones=[30, 60, 85], gamma=0.1)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=1)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
# Swin with activation checkpointing needs a static graph under DDP.
model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=False, static_graph=True)

auto_scale_lr = dict(base_batch_size=8)

default_hooks = dict(
    visualization=dict(type='GroundingVisualizationHook'),
    checkpoint=dict(max_keep_ckpts=1, save_best='auto'))

custom_hooks = [
    # SCGM curriculum: object-level phrases for epochs 0-9, attribute-level for 10-24, spatial after
    dict(type='CurriculumPhraseHook', ep1=10, ep2=25),
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', rule='greater', min_delta=0.001,
         patience=30, strict=False, check_finite=True),
]

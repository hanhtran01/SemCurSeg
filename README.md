# SemCurSeg: Semantic Curriculum Grounding for Text-Guided Weakly Supervised Segmentation


SemCurSeg trains a text-guided lesion segmentation network from bounding boxes and the masks of a
small fraction of the training images. In Stage 1, the Semantic Curriculum Grounding Module (SCGM)
turns each lesion box into curriculum captions, the Curriculum-Guided Textual-to-Visual Cue
Converter (CG-TVCC) grounds them to boxes, and SAM Adapt converts the boxes into pseudo masks. In
Stage 2, a segmentation network with the Adaptive Prompt Feature Optimizer (APFO) and Text-Vision
Hybrid Attention (TVHA) is trained on the pseudo masks. At inference no annotation is used: CG-TVCC
is prompted with the class name only, and the caption and pseudo mask given to the Stage 2 network
are derived from its predicted boxes.

<img src=images/framework.png width=800 />
```
## Code and paper

| Paper | Code |
|---|---|
| 3.3 SCGM: attribute extraction, curriculum captions, object-aware phrase assignment | `semcurseg/scgm/attributes.py`, `tools/build_scgm_captions.py` |
| 3.3 SCGM: curriculum schedule (epochs 0-9 / 10-24 / 25+) | `semcurseg/scgm/curriculum.py` |
| 3.3 SCGM: phrase-specific loss weighting, w_k proportional to 1/a_k | `semcurseg/scgm/phrase_weighting.py` |
| 3.4 CG-TVCC (Grounding DINO Swin-T) | `configs/cgtvcc_polyp.py`, `configs/cgtvcc_brain.py`, `tools/train_cgtvcc.py` |
| 3.5 SAM Adapt: HCAE, edge prompt, uncertainty, NBIS, HFNM | `semcurseg/sam_adapt/`, `tools/train_sam_adapt.py` |
| Stage 1 inference: pseudo masks and captions from predicted boxes | `tools/stage1_inference.py` |
| 3.6 APFO | `semcurseg/segmentor/apfo.py` |
| 3.7 TVHA and the attention decoder | `semcurseg/segmentor/tvha.py`, `semcurseg/segmentor/model.py` |
| 3.8 Training objective (Dice + BCE) | `semcurseg/segmentor/loss.py`, `tools/train_segmentor.py` |
| 4.2 mIoU, mDice, HD95 | `semcurseg/metrics.py`, `tools/test_segmentor.py`, `tools/eval_hd95.py` |
| 4.3 Implementation details | defaults of the configs and of the scripts in `tools/` |
```
```
configs/           CG-TVCC configs (MMDetection 3.3.0)
semcurseg/         scgm/, sam_adapt/, segmentor/, metrics.py
tools/             one script per step of the pipeline (below)
third_party/       Segment Anything
images/            figures
```

## Installation

The experiments in the paper used PyTorch 1.13.1 and MMDetection 3.3.0 (MMCV 2.1.0, MMEngine 0.10).

```bash
conda create -n semcurseg python=3.10 -y && conda activate semcurseg
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U openmim && mim install mmengine "mmcv==2.1.0" "mmdet==3.3.0"
pip install -r requirements.txt
```

Download into `checkpoints/`:
[swin_tiny_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth),
the [Grounding DINO Swin-T checkpoint](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth)
and [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
BERT-base and ConvNeXt-Tiny are downloaded from the Hugging Face hub on first use.

## Data

- Polyp: Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB (e.g. from the
  [PraNet collection](https://github.com/DengPingFan/PraNet)) and
  [PolypGen](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312).
- Brain tumour: [LGG segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (TCIA).

```
data/polyp/
  train/images  train/masks
  val/images    val/masks
  test/images   test/masks
```

Masks are binary PNG files with the same stem as their image. Training reads only the boxes of the
training masks (one box per connected component) and the masks of the SAM Adapt subset.
Validation masks are used to select checkpoints and test masks only to compute the metrics.

## Training and evaluation

All commands are run from the repository root. They are given for polyps; for brain MRI use
`data/brain`, `--task tumor`, `--class-name tumor` and `configs/cgtvcc_brain.py`.

**1. SCGM captions and size thresholds**

```bash
python tools/build_scgm_captions.py --image-dir data/polyp/train/images --mask-dir data/polyp/train/masks \
    --task polyp --output data/polyp/annotations/train_odvg.jsonl \
    --thresholds-out data/polyp/annotations/size_thresholds.json
python tools/build_coco_boxes.py --image-dir data/polyp/val/images --mask-dir data/polyp/val/masks \
    --class-name polyp --output data/polyp/annotations/val_coco.json
python tools/build_coco_boxes.py --image-dir data/polyp/test/images --mask-dir data/polyp/test/masks \
    --class-name polyp --output data/polyp/annotations/test_coco.json
```

The size thresholds are the 33rd and 66th percentiles of the training box areas. CG-TVCC is
validated and tested on the COCO boxes with the class name as the only prompt.

**2. CG-TVCC**

```bash
bash tools/dist_train_cgtvcc.sh configs/cgtvcc_polyp.py 4      # or: python tools/train_cgtvcc.py configs/cgtvcc_polyp.py
python tools/test_cgtvcc.py configs/cgtvcc_polyp.py work_dirs/cgtvcc_polyp/<best checkpoint>.pth
```

**3. SAM Adapt**

```bash
python tools/train_sam_adapt.py --image-dir data/polyp/train/images --mask-dir data/polyp/train/masks \
    --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth --mask-ratio 0.2 --out-dir work_dirs/sam_adapt_polyp
```

The masks of a random 20% of the training images are used; the list is written to
`work_dirs/sam_adapt_polyp/mask_subset.txt`.

**4. Stage 1 inference** (every split, from predicted boxes only)

```bash
for split in train val test; do
  python tools/stage1_inference.py --config configs/cgtvcc_polyp.py \
      --checkpoint work_dirs/cgtvcc_polyp/<best checkpoint>.pth \
      --sam-adapt work_dirs/sam_adapt_polyp/best.pth --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
      --thresholds data/polyp/annotations/size_thresholds.json \
      --image-dir data/polyp/$split/images --out-dir data/polyp/stage1/$split
done
```

**5. Stage 2**

```bash
python tools/train_segmentor.py \
    --train-images data/polyp/train/images --train-pseudo data/polyp/stage1/train/masks \
    --train-captions data/polyp/stage1/train/captions.json \
    --val-images data/polyp/val/images --val-masks data/polyp/val/masks \
    --val-pseudo data/polyp/stage1/val/masks --val-captions data/polyp/stage1/val/captions.json \
    --out-dir work_dirs/segmentor_polyp
```

**6. Evaluation**

```bash
python tools/test_segmentor.py --images data/polyp/test/images --masks data/polyp/test/masks \
    --pseudo data/polyp/stage1/test/masks --captions data/polyp/stage1/test/captions.json \
    --checkpoint work_dirs/segmentor_polyp/best_dice.pth --save-dir work_dirs/segmentor_polyp/pred_test
python tools/eval_hd95.py --gt data/polyp/test/masks --pred work_dirs/segmentor_polyp/pred_test
```

`test_segmentor.py` reports mIoU and mDice averaged over images; `eval_hd95.py` computes HD95
between mask boundaries at the ground-truth resolution.

## Ablations

| Experiment | Setting |
|---|---|
| Table 5, without APFO | `--no-apfo` in `train_segmentor.py` and `test_segmentor.py` |
| Table 5, "+SCGM" (no pixel-level masks) | omit `--sam-adapt` in `stage1_inference.py` (original SAM), then train Stage 2 with `--no-apfo` |
| Table 6, mask budget | `--mask-ratio 0.2 / 0.4 / 0.6 / 0.8 / 1.0` in `train_sam_adapt.py` |
| SAM Adapt modules | `--no-hcae`, `--no-edge-prompt`, `--no-uncertainty`, `--no-nbis`, `--no-hfnm` |
| without curriculum | remove `CurriculumPhraseHook` from the config and set `max_phrase_level=2` |
| without phrase weighting | `use_phrase_weighting=False` in the config |

## License

Apache License 2.0 (see `LICENSE`). Code taken from SimTxtSeg, MMDetection and Segment Anything,
and the changes made to it, are listed in `NOTICE`.

## Acknowledgements

This code builds on [SimTxtSeg](https://github.com/xyx1024/SimTxtSeg),
[MMDetection](https://github.com/open-mmlab/mmdetection),
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and
[Segment Anything](https://github.com/facebookresearch/segment-anything).

## Citation

The citation will be added after the review process.

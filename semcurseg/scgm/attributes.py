"""SCGM attribute extraction (Sec. 3.3.1), curriculum captions (Sec. 3.3.2) and object-aware phrase
assignment (Sec. 3.3.3).

Every attribute is computed from a lesion bounding box (x1, y1, x2, y2) and the image size, so the
same code serves training (ground-truth boxes) and inference (boxes predicted by CG-TVCC).

    size      small / medium-sized / large, from the box area relative to the image, with the
              33rd and 66th percentiles of the training boxes as thresholds
    shape     round / oval / elongated, from the aspect ratio of the box (for a rectangle the
              compactness 4*pi*A/P^2 is a function of the aspect ratio only)
    location  cell of the box centre in a 3x3 grid over the image

The attributes fill three caption levels (object, attribute, spatial), e.g.
    "polyp"
    "small round polyp"
    "small round polyp is visible in the upper part of the colon"
Each lesion keeps its own phrase set and character spans, so lesions of one image are grounded
separately.
"""
import json

import numpy as np

# Object name and the organ used in location phrases, per task.
TASKS = {
    'polyp': dict(object='polyp', organ='colon'),
    'tumor': dict(object='tumor', organ='brain'),
}

SIZE_WORDS = ('small', 'medium-sized', 'large')
NUM_LEVELS = 3


# ----------------------------------------------------------------------------------------------
# Size
# ----------------------------------------------------------------------------------------------

def box_area_ratio(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(img_w * img_h)


def calibrate_size_thresholds(area_ratios, p_small=33.0, p_medium=66.0):
    """Percentile thresholds of the training-box area ratios."""
    ratios = np.asarray(area_ratios, dtype=np.float64)
    if ratios.size == 0:
        raise ValueError('no training boxes to calibrate the size thresholds')
    return {
        'small_max': float(np.percentile(ratios, p_small)),
        'medium_max': float(np.percentile(ratios, p_medium)),
        'p_small': p_small,
        'p_medium': p_medium,
        'num_boxes': int(ratios.size),
    }


def save_thresholds(thresholds, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds(path):
    with open(path, 'r', encoding='utf-8') as f:
        thr = json.load(f)
    for key in ('small_max', 'medium_max'):
        if key not in thr:
            raise KeyError(f'{path} has no "{key}"')
    return thr


def size_word(box, img_w, img_h, thresholds):
    ratio = box_area_ratio(box, img_w, img_h)
    if ratio < thresholds['small_max']:
        return SIZE_WORDS[0]
    if ratio < thresholds['medium_max']:
        return SIZE_WORDS[1]
    return SIZE_WORDS[2]


# ----------------------------------------------------------------------------------------------
# Shape
# ----------------------------------------------------------------------------------------------

def shape_word(box, oval_ratio=1.25, elongated_ratio=1.6):
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    aspect = max(w, h) / min(w, h)
    if aspect >= elongated_ratio:
        return 'elongated'
    if aspect >= oval_ratio:
        return 'oval'
    return 'round'


# ----------------------------------------------------------------------------------------------
# Location
# ----------------------------------------------------------------------------------------------

def grid_cell(box, img_w, img_h):
    """(row, col) of the box centre in a 3x3 grid."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    col = 'left' if cx < img_w / 3 else ('right' if cx > 2 * img_w / 3 else 'central')
    row = 'upper' if cy < img_h / 3 else ('lower' if cy > 2 * img_h / 3 else 'mid')
    return row, col


def location_phrase(box, img_w, img_h, organ='colon'):
    row, col = grid_cell(box, img_w, img_h)
    if row == 'mid' and col == 'central':
        return f'in the center of the {organ}'
    if row == 'mid':
        side = 'wall' if organ == 'colon' else 'side'
        return f'on the {col} {side} of the {organ}'
    if col == 'central':
        return f'in the {row} part of the {organ}'
    return f'in the {row} {col} region of the {organ}'


# ----------------------------------------------------------------------------------------------
# Captions
# ----------------------------------------------------------------------------------------------

def lesion_phrases(box, img_w, img_h, thresholds, task='polyp'):
    """Object-, attribute- and spatial-level phrases of one lesion; each contains the previous."""
    obj = TASKS[task]['object']
    organ = TASKS[task]['organ']
    attr = f'{size_word(box, img_w, img_h, thresholds)} {shape_word(box)} {obj}'
    spatial = f'{attr} is visible {location_phrase(box, img_w, img_h, organ)}'
    return [obj, attr, spatial]


def negative_caption(task='polyp'):
    return f'No {TASKS[task]["object"]} is visible in the {TASKS[task]["organ"]}.'


def find_span(text, phrase, start=0):
    idx = text.lower().find(phrase.lower(), start)
    if idx < 0:
        raise ValueError(f'"{phrase}" not found in "{text}"')
    return [idx, idx + len(phrase)]


def compose_caption(phrase_sets, level):
    """Caption for a curriculum level and, per lesion, the character spans of levels 0..level.

    Each lesion contributes the phrase of the requested level, followed by a full stop, so lesions
    never share tokens. Returns (caption, spans) with spans[i][k] = [[start, end]] for lesion i and
    level k <= level.
    """
    parts, spans, offset = [], [], 0
    for phrases in phrase_sets:
        top = min(level, len(phrases) - 1)
        part = phrases[top]
        lesion_spans = []
        for k in range(top + 1):
            beg, end = find_span(part, phrases[k])
            lesion_spans.append([[offset + beg, offset + end]])
        parts.append(part + '.')
        spans.append(lesion_spans)
        offset += len(part) + 2          # phrase + '.' + ' '
    return ' '.join(parts), spans


def odvg_record(filename, img_w, img_h, boxes, thresholds, task='polyp'):
    """ODVG (JSON lines) record with the spatial-level caption and the phrases of every level."""
    if not boxes:
        return {
            'filename': filename, 'height': int(img_h), 'width': int(img_w),
            'grounding': {'caption': negative_caption(task), 'regions': []},
        }
    phrase_sets = [lesion_phrases(b, img_w, img_h, thresholds, task) for b in boxes]
    caption, spans = compose_caption(phrase_sets, NUM_LEVELS - 1)
    regions = []
    for box, phrases, lesion_spans in zip(boxes, phrase_sets, spans):
        regions.append({
            'bbox': [[int(round(v)) for v in box]],
            'phrases': phrases,
            'all_tokens_positive': lesion_spans,
            # the usual ODVG keys, set to the spatial level
            'phrase': phrases[-1],
            'tokens_positive': lesion_spans[-1],
        })
    return {
        'filename': filename, 'height': int(img_h), 'width': int(img_w),
        'grounding': {'caption': caption, 'regions': regions},
    }


def caption_from_boxes(boxes, img_w, img_h, thresholds, task='polyp'):
    """Spatial-level caption used as the text input of the Stage 2 network."""
    if not boxes:
        return negative_caption(task)
    phrase_sets = [lesion_phrases(b, img_w, img_h, thresholds, task) for b in boxes]
    return compose_caption(phrase_sets, NUM_LEVELS - 1)[0]

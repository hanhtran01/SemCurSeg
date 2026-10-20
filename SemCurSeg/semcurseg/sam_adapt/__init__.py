"""SAM Adapt (Sec. 3.5): SAM ViT with a frozen encoder, HCAE, edge prompts, uncertainty and NBIS refinement.

    sam_adapt.py    SAMAdapt model (forward, logits at image resolution, prediction)
    hcae.py         Hierarchical Cross-scale Attention Enhancement
    edge_prompt.py  edge prompt generator fused with the box prompt
    refinement.py   uncertainty head, NBIS and hard-region refinement
    hfnm.py         High-Frequency Noise Masking augmentation
"""
import os.path as osp
import sys

# Segment Anything ships in third_party/; an installed `segment_anything` takes precedence.
_THIRD_PARTY = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'third_party')
if _THIRD_PARTY not in sys.path:
    sys.path.append(_THIRD_PARTY)

from .hfnm import HFNMTransform  # noqa: E402
from .refinement import UncertaintyEstimator  # noqa: E402
from .sam_adapt import SAMAdapt  # noqa: E402

__all__ = ['SAMAdapt', 'HFNMTransform', 'UncertaintyEstimator']

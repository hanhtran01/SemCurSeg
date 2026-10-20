"""Semantic Curriculum Grounding Module (Sec. 3.3).

    attributes.py         attribute extraction, curriculum captions and per-lesion phrase spans
                          (Sec. 3.3.1-3.3.3); plain Python, used for training and inference captions
    curriculum.py         curriculum ODVG dataset and phrase-level hook for MMDetection (Sec. 3.3.2)
    phrase_weighting.py   phrase-specific loss weighting for Grounding DINO (Sec. 3.3.4)

The MMDetection modules are imported by the CG-TVCC configs (custom_imports) and are not imported
here, so that attributes.py can be used without MMDetection.
"""

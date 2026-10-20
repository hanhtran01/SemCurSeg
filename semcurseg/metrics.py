"""Evaluation metrics (Sec. 4.2): per-image Dice and IoU (averaged into mDice and mIoU) and HD95.

HD95 is the 95th percentile of the distances between the boundary pixels of the predicted and the
ground-truth mask, taken in both directions.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree


def overlap_metrics(gt, pred):
    """gt, pred: binary arrays of the same shape. Returns Dice, IoU, recall and precision."""
    gt = gt.astype(bool).ravel()
    pred = pred.astype(bool).ravel()
    tp = float(np.logical_and(gt, pred).sum())
    g, p = float(gt.sum()), float(pred.sum())
    eps = 1e-15
    return {
        'dice': (2 * tp + eps) / (g + p + eps),
        'iou': (tp + eps) / (g + p - tp + eps),
        'recall': (tp + eps) / (g + eps),
        'precision': (tp + eps) / (p + eps),
    }


def boundary_points(mask):
    mask = mask.astype(np.uint8)
    return np.column_stack(np.nonzero(mask - cv2.erode(mask, np.ones((3, 3), np.uint8))))


def hd95(gt, pred):
    """HD95 in pixels between two binary masks of the same size; None if either mask is empty."""
    a, b = boundary_points(pred), boundary_points(gt)
    if len(a) == 0 or len(b) == 0:
        return None
    da, _ = cKDTree(b).query(a, k=1)
    db, _ = cKDTree(a).query(b, k=1)
    return float(np.percentile(np.concatenate([da, db]), 95))

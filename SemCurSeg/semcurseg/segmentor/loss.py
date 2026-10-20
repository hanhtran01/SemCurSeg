"""Training objective of the Stage 2 network (Sec. 3.8): L = lambda_1 * Dice + lambda_2 * BCE."""
import torch.nn.functional as F


def dice_bce_loss(prob, target, lambda_dice=1.0, lambda_bce=1.0, eps=1.0):
    """prob: predicted probabilities (B, 1, H, W); target: pseudo mask in {0, 1}."""
    prob = prob.clamp(1e-6, 1 - 1e-6)
    inter = (prob * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return lambda_dice * dice.mean() + lambda_bce * F.binary_cross_entropy(prob, target)

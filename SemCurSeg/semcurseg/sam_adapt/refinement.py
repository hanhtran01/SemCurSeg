"""Uncertainty head and hard-region refinement of SAM Adapt.

UncertaintyHead: evidential (Dirichlet) estimate of a per-pixel uncertainty map from the decoder
logits, followed by a residual refinement of the logits guided by that map.
NBIS: noisy boundary importance sampling; the variance of a boundary predictor under feature noise
marks hard regions, and HardRegionRefinement applies a further residual correction there.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator(nn.Module):

    def __init__(self, hidden_channels=32):
        super().__init__()
        self.evidence_net = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1))

    def forward(self, logits):
        """logits [B, 1, H, W] -> uncertainty [B, 1, H, W], evidence and Dirichlet alpha [B, 2, H, W]."""
        evidence = F.softplus(self.evidence_net(torch.cat([logits, -logits], dim=1)), beta=5)
        alpha = evidence + 1.0
        uncertainty = 2.0 / alpha.sum(dim=1, keepdim=True)
        return uncertainty, evidence, alpha

    @staticmethod
    def loss(alpha, target, eps=1e-8):
        """Uncertainty term of the SAM Adapt loss; target [B, 1, H, W] in {0, 1}."""
        one_hot = torch.cat([1 - target, target], dim=1)
        prob = alpha / (alpha.sum(dim=1, keepdim=True) + eps)
        return (one_hot * (torch.log(prob + eps) - torch.log(alpha / (alpha + eps) + eps))).sum(dim=1).mean()


class UncertaintyHead(nn.Module):

    def __init__(self, hidden_channels=32):
        super().__init__()
        self.estimator = UncertaintyEstimator(hidden_channels)
        self.refine_net = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))

    def forward(self, logits):
        uncertainty, _, _ = self.estimator(logits)
        refined = logits + 0.5 * self.refine_net(torch.cat([logits, uncertainty], dim=1))
        return refined, uncertainty


class NBIS(nn.Module):

    def __init__(self, feature_dim=256, hidden_dim=128, noise_std=0.1, num_samples=5):
        super().__init__()
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1), nn.Sigmoid())

    def forward(self, features):
        """features [B, C, h, w] -> variance map [B, 1, h, w]."""
        preds = torch.stack([self.boundary_conv(features + torch.randn_like(features) * self.noise_std)
                             for _ in range(self.num_samples)])
        return preds.var(dim=0, unbiased=False)


class HardRegionRefinement(nn.Module):

    def __init__(self, hidden_channels=32):
        super().__init__()
        self.refine_conv = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1))

    def forward(self, logits, variance):
        variance = F.interpolate(variance, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        variance = variance.expand(logits.shape[0], -1, -1, -1)
        residual = self.refine_conv(torch.cat([logits, variance], dim=1))
        return logits + residual * torch.sigmoid(variance * 5)

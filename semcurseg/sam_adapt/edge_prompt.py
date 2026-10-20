"""Edge prompt generator of SAM Adapt.

Structural edges (Canny and local entropy) inside the box are fused with channel-attended SAM
features into a one-channel prompt, which SAM Adapt passes to the prompt encoder as a dense (mask)
prompt together with the box.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_uint8_hwc(t):
    """[C, H, W] tensor of any range (e.g. SAM-normalised) -> [H, W, C] uint8, min-max rescaled."""
    img = t.detach().float().cpu().numpy().transpose(1, 2, 0)
    lo, hi = img.min(), img.max()
    return ((img - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)


def edge_map(images, canny_low=50, canny_high=150, entropy_window=15, n_bins=32):
    """0.5 * Canny + 0.5 * normalised local entropy, [B, 1, H, W] in [0, 1]."""
    out = torch.zeros(images.shape[0], 1, *images.shape[2:], dtype=torch.float32)
    edges_bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(images.shape[0]):
        gray = cv2.cvtColor(_to_uint8_hwc(images[i]), cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, canny_low, canny_high).astype(np.float32) / 255.0
        g = gray.astype(np.float32) / 255.0
        entropy = np.zeros_like(g)
        for b in range(n_bins):
            ind = ((g >= edges_bins[b]) & (g < edges_bins[b + 1])).astype(np.float32)
            p = cv2.boxFilter(ind, -1, (entropy_window, entropy_window), normalize=True,
                              borderType=cv2.BORDER_REFLECT)
            entropy -= p * np.log(p + 1e-8)
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        out[i, 0] = torch.from_numpy(0.5 * canny + 0.5 * entropy)
    return out.to(images.device)


class EdgePromptGenerator(nn.Module):
    """edges [1, 1, H, W] (edge_map of the SAM input), features [1, C, h, w], box [1, 4] in the
    SAM input frame -> prompt [1, 1, h, w] in [0, 1]."""

    def __init__(self, feature_channels=256, hidden_dim=128, reduction=4):
        super().__init__()
        self.ca_fc1 = nn.Linear(feature_channels, feature_channels // reduction, bias=False)
        self.ca_fc2 = nn.Linear(feature_channels // reduction, feature_channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_channels + 1, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, edges, features, box):
        _, _, h_img, w_img = edges.shape
        x1, y1, x2, y2 = (int(v) for v in box[0].round().tolist())
        inside = torch.zeros_like(edges)
        inside[..., max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)] = 1.0
        edges = F.interpolate(edges * inside, size=features.shape[-2:], mode='bilinear', align_corners=False)

        gap = features.mean(dim=(2, 3))
        attn = torch.sigmoid(self.ca_fc2(F.relu(self.ca_fc1(gap))))[..., None, None]
        return self.fuse(torch.cat([features * attn, edges], dim=1))

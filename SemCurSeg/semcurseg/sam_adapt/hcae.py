"""Hierarchical Cross-scale Attention Enhancement (HCAE) of SAM Adapt.

Features of four evenly spaced blocks of the frozen SAM ViT encoder are projected to a common width,
reweighted by pairwise cross-scale attention and fused; the result is added to the SAM image
embedding with a learnable weight.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HCAEModule(nn.Module):

    def __init__(self, in_channels, num_scales=4, hidden_dim=128, out_channels=256):
        super().__init__()
        self.num_scales = num_scales
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                          nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
            for _ in range(num_scales)])
        self.attn_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim * num_scales, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.output_proj = nn.Conv2d(hidden_dim, out_channels, 1)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, features):
        size = features[-1].shape[-2:]
        w = torch.softmax(self.scale_weights, dim=0)
        proj = []
        for i, (f, p) in enumerate(zip(features, self.projections)):
            x = p(f)
            if x.shape[-2:] != size:
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            proj.append(x * w[i])
        attn = torch.stack([self.attn_conv(torch.cat([proj[i], proj[j]], dim=1))
                            for i in range(self.num_scales) for j in range(i + 1, self.num_scales)]).mean(0)
        return self.output_proj(self.refine(torch.cat([x * attn for x in proj], dim=1)))


class HCAE(nn.Module):
    """Runs the frozen SAM image encoder and enhances its embedding with HCAE."""

    def __init__(self, sam_encoder, hidden_dim=128):
        super().__init__()
        # keep the encoder out of this module's parameters; it belongs to SAMAdapt.sam
        self._encoder = [sam_encoder]
        embed_dim = sam_encoder.patch_embed.proj.out_channels
        out_channels = sam_encoder.neck[0].out_channels
        n = len(sam_encoder.blocks)
        self.block_ids = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        self.hcae = HCAEModule(embed_dim, num_scales=4, hidden_dim=hidden_dim, out_channels=out_channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))   # small residual weight at the start
        self._features = {}
        for k, idx in enumerate(self.block_ids):
            sam_encoder.blocks[idx].register_forward_hook(self._hook(k))

    def _hook(self, key):
        def fn(module, inputs, output):
            self._features[key] = output
        return fn

    def forward(self, images):
        self._features.clear()
        with torch.no_grad():
            embedding = self._encoder[0](images)                         # [B, 256, 64, 64]
        feats = [self._features[k].permute(0, 3, 1, 2).contiguous()     # ViT blocks are NHWC
                 for k in range(len(self.block_ids))]
        enhanced = self.hcae(feats)
        if enhanced.shape[-2:] != embedding.shape[-2:]:
            enhanced = F.interpolate(enhanced, size=embedding.shape[-2:], mode='bilinear', align_corners=False)
        return embedding + self.alpha * enhanced

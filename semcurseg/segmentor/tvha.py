"""Text-Vision Hybrid Attention (TVHA, Sec. 3.7) and the three-stage attention decoder.

Adapted from SimTxtSeg (Xie et al., MICCAI 2024), Apache-2.0. Changes: attention pooling of the
caption tokens; positional encodings sized to the tokens used and not stored in checkpoints; adaptive
global pooling in the channel attention; dropout registered as a module (SimTxtSeg created it inside
forward(), so it stayed active at test time); padding in UpCat created on the input device.
"""
import math

import torch
import torch.nn as nn
from einops import rearrange
from monai.networks.blocks.upsample import SubpixelUpsample


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TextVisionFusion(nn.Module):
    """f_v' = LN(MHSA(f_v)) + f_v;  f_t' = LN(MHCA_tv(f_t, f_v')) + f_t;  f_mix = LN(MHCA_vt(f_v', f_t')) + f_v'."""

    def __init__(self, in_channels, output_text_len, spatial_size, input_text_len=24, embed_dim=768):
        super().__init__()
        self.spatial_size = spatial_size
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm1 = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(in_channels, num_heads=1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)    # vision -> text
        self.cross_attn1 = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)   # text -> vision
        self.text_project = nn.Sequential(                                                      # Phi_proj
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels, max_len=spatial_size * spatial_size)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, txt):
        """x: [B, C, H, W] visual features; txt: [B, L, 768] text tokens."""
        x = rearrange(x, 'b c h w -> b (h w) c')
        txt = self.text_project(txt)

        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis = x + self.self_attn_norm(self.self_attn(q, k, value=vis2)[0])

        txt2, _ = self.cross_attn1(query=self.txt_pos(txt), key=self.vis_pos(vis), value=vis)
        txt = txt + self.cross_attn_norm1(txt2)

        vis2 = self.norm2(vis)
        vis2, _ = self.cross_attn(query=self.vis_pos(vis2), key=self.txt_pos(txt), value=txt)
        vis = vis + self.scale * self.cross_attn_norm(vis2)
        return rearrange(vis, 'b (h w) c -> b c h w', h=self.spatial_size, w=self.spatial_size)


class ChannelAttentionBlock(nn.Module):
    """W_c = sigmoid(MLP(AvgPool(f)) + MLP(MaxPool(f))) applied to the fused map, then a 3x3 conv."""

    def __init__(self, inplanes, planes, drop_out=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes * 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(planes * 2, round(planes / 2))
        self.fc2 = nn.Linear(round(planes / 2), planes * 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(0.5) if drop_out else None
        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, 1, bias=False),
                                             nn.BatchNorm2d(planes * 2))

    def _weights(self, pooled):
        w = self.sigmoid(self.fc2(self.relu(self.fc1(pooled.flatten(1)))))
        return w.view(w.size(0), w.size(1), 1, 1)

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        residual = self.downchannel(x) if self.downchannel is not None else x
        w_avg = self._weights(self.globalAvgPool(out))
        w_max = self._weights(self.globalMaxPool(out))
        out = w_avg * out + w_max * out + residual
        out = self.relu(self.bn3(self.conv3(self.relu(out))))
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class UpCat(nn.Module):
    """Up-samples the deeper features and concatenates them with the skip connection."""

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, skip, deep):
        up = self.up(deep)
        offset = skip.size(3) - up.size(3)
        if offset > 0:     # odd sizes: pad the width as in SimTxtSeg
            up = torch.cat([up, torch.rand(*up.shape[:3], offset, device=up.device, dtype=up.dtype)], dim=3)
        return torch.cat([skip, up], dim=1)


class AttentionTokenPool(nn.Module):
    """Pools the caption tokens to `tgt_len` tokens with learnable queries (cross-attention)."""

    def __init__(self, d_model=768, tgt_len=24, n_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, tgt_len, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, key_padding_mask=None):
        q = self.queries.expand(tokens.size(0), -1, -1)
        return self.norm(self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)[0])


class TVHADecoder(nn.Module):
    """Three decoder stages, each: up-sample and concatenate, text-vision fusion, channel attention.

    Stage 1 starts from the deepest features (global lesion structure), stage 2 adds intermediate
    encoder features (shape) and stage 3 shallow features (texture and boundary).
    """

    def __init__(self, text_len=24):
        super().__init__()
        filters = [96, 192, 384, 768]                   # ConvNeXt-Tiny stage widths
        self.txt_pool = AttentionTokenPool(d_model=768, tgt_len=text_len)
        self.up_concat4 = UpCat(filters[3], filters[2])
        self.up_concat3 = UpCat(filters[2], filters[1])
        self.up_concat2 = UpCat(filters[1], filters[0])
        self.fusion1 = TextVisionFusion(filters[3], 24, 24, input_text_len=text_len)
        self.fusion2 = TextVisionFusion(filters[2], 12, 48, input_text_len=text_len)
        self.fusion3 = TextVisionFusion(filters[1], 9, 96, input_text_len=text_len)
        self.up4 = ChannelAttentionBlock(filters[3], filters[2], drop_out=True)
        self.up3 = ChannelAttentionBlock(filters[2], filters[1])
        self.up2 = ChannelAttentionBlock(filters[1], filters[0])
        self.up1 = SubpixelUpsample(2, filters[0], 24, 4)

    def forward(self, vis, txt, txt_padding_mask=None):
        """vis: ConvNeXt features [(B,96,96,96), (B,192,48,48), (B,384,24,24), (B,768,12,12)];
        txt: (B, L, 768); txt_padding_mask: (B, L), True at padding."""
        txt = self.txt_pool(txt, key_padding_mask=txt_padding_mask)
        x = self.up4(self.fusion1(self.up_concat4(vis[2], vis[3]), txt))
        x = self.up3(self.fusion2(self.up_concat3(vis[1], x), txt))
        x = self.up2(self.fusion3(self.up_concat2(vis[0], x), txt))
        return self.up1(x)

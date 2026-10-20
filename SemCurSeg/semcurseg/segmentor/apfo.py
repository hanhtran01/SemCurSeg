"""Adaptive Prompt Feature Optimizer (APFO, Sec. 3.6)."""
import torch
import torch.nn as nn


class APFO(nn.Module):
    """Z = F(E_t, F_v, E_m), alpha = sigmoid(MLP(Z)), E_t' = alpha * E_t.

    Z concatenates the mean text embedding over the valid tokens (E_t), the globally pooled deepest
    visual features (F_v) and a pooled embedding of the Stage 1 pseudo mask (E_m); alpha gates the
    channels of every token. Only a global vector of the pseudo mask enters, so the network cannot
    copy the mask spatially.
    """

    def __init__(self, text_dim=768, vis_dim=768, mask_dim=64, hidden_dim=512):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.mask_dim = mask_dim
        self.mask_encoder = nn.Sequential(block(1, 16), block(16, 32), block(32, mask_dim),
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(text_dim + vis_dim + mask_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, text_dim))
        # start close to the identity (alpha ~ 0.88) so that the text is not suppressed early on
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def forward(self, txt, txt_mask, vis, pseudo_mask=None):
        """txt (B, L, 768), txt_mask (B, L) with 1 for valid tokens, vis (B, 768, h, w),
        pseudo_mask (B, 1, H, W) in [0, 1] or None. Returns E_t' and alpha (B, 768)."""
        valid = txt_mask.unsqueeze(-1).to(txt.dtype)
        e_t = (txt * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        f_v = vis.mean(dim=(2, 3))
        if pseudo_mask is None:
            e_m = txt.new_zeros(txt.size(0), self.mask_dim)
        else:
            e_m = self.mask_encoder(pseudo_mask.to(txt.dtype))
        alpha = torch.sigmoid(self.mlp(torch.cat([e_t, f_v, e_m], dim=1)))
        return txt * alpha.unsqueeze(1), alpha

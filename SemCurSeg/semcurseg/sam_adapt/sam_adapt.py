"""SAM Adapt (Sec. 3.5): SAM with a frozen image encoder, adapted with a small set of lesion masks.

Trained parts: the SAM mask decoder, HCAE, the edge prompt generator, the uncertainty head and the
hard-region refinement (NBIS). The image encoder and the prompt encoder stay frozen.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from .edge_prompt import EdgePromptGenerator, edge_map
from .hcae import HCAE
from .refinement import NBIS, HardRegionRefinement, UncertaintyHead


class SAMAdapt(nn.Module):

    def __init__(self, sam_type='vit_h', sam_checkpoint=None, use_hcae=True, use_edge_prompt=True,
                 use_uncertainty=True, use_nbis=True):
        super().__init__()
        self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        encoder = self.sam.image_encoder
        embed_dim = encoder.patch_embed.proj.out_channels
        prompt_dim = self.sam.prompt_encoder.embed_dim
        self.hcae = HCAE(encoder, hidden_dim=max(64, min(256, embed_dim // 10))) if use_hcae else None
        self.edge_prompt = EdgePromptGenerator(prompt_dim) if use_edge_prompt else None
        self.uncertainty = UncertaintyHead() if use_uncertainty else None
        self.nbis = NBIS(prompt_dim) if use_nbis else None
        self.hard_region = HardRegionRefinement() if use_nbis else None
        for p in list(self.sam.image_encoder.parameters()) + list(self.sam.prompt_encoder.parameters()):
            p.requires_grad = False
        self.transform = ResizeLongestSide(encoder.img_size)

    def train(self, mode=True):
        super().train(mode)
        self.sam.image_encoder.eval()       # frozen backbone
        return self

    def embed(self, images):
        if self.hcae is not None:
            return self.hcae(images)
        with torch.no_grad():
            return self.sam.image_encoder(images)

    def forward(self, images, boxes, return_all=False):
        """images: one SAM-preprocessed image [1, 3, 1024, 1024]; boxes: [N, 4] in the same frame.
        Returns low-resolution mask logits [N, 1, 256, 256]."""
        assert images.shape[0] == 1, 'one image with N boxes per call'
        emb = self.embed(images)

        dense_input = None
        if self.edge_prompt is not None:
            edges = edge_map(images)
            edge = torch.cat([self.edge_prompt(edges, emb, boxes[j:j + 1]) for j in range(boxes.shape[0])])
            edge = F.interpolate(edge, size=self.sam.prompt_encoder.mask_input_size, mode='bilinear',
                                 align_corners=False)
            dense_input = torch.logit(edge.clamp(1e-4, 1.0 - 1e-4))     # SAM mask prompts are logits

        sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes, masks=dense_input)
        logits, _ = self.sam.mask_decoder(image_embeddings=emb, image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                          sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                          multimask_output=False)
        uncertainty = variance = None
        if self.uncertainty is not None:
            logits, uncertainty = self.uncertainty(logits)
        if self.nbis is not None:
            variance = self.nbis(emb)
            logits = self.hard_region(logits, variance)
        if return_all:
            return dict(logits=logits, uncertainty=uncertainty, variance=variance)
        return logits

    def load_trained(self, state):
        """Load parameters saved by trained_state(); the frozen encoder keeps the SAM checkpoint."""
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith('sam.image_encoder.')]
        if missing or unexpected:
            raise RuntimeError(f'SAM Adapt checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}')
        return self

    def trained_state(self):
        """State dict without the frozen SAM image encoder."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith('sam.image_encoder.')}

    def logits_at(self, image, boxes):
        """Mask logits [N, 1, H, W] at the resolution of `image` ([H, W, 3] uint8 RGB, boxes [N, 4] in pixels)."""
        device = next(self.parameters()).device
        h, w = image.shape[:2]
        resized = self.transform.apply_image(image)
        x = torch.from_numpy(resized).permute(2, 0, 1).float().to(device)
        x = self.sam.preprocess(x.unsqueeze(0))
        b = torch.from_numpy(self.transform.apply_boxes(np.asarray(boxes, np.float32), (h, w))).float().to(device)
        return self.sam.postprocess_masks(self(x, b), input_size=resized.shape[:2], original_size=(h, w))

    @torch.no_grad()
    def predict(self, image, boxes, threshold=0.5):
        """Binary masks [N, H, W] (bool, CPU) for the boxes of one image."""
        self.eval()
        return (torch.sigmoid(self.logits_at(image, boxes)) > threshold)[:, 0].cpu()

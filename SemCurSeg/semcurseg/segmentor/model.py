"""Stage 2 segmentation network: ConvNeXt-Tiny and frozen BERT-base encoders, APFO and TVHA."""
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from transformers import AutoModel

from .apfo import APFO
from .tvha import TVHADecoder


class TextEncoder(nn.Module):
    """Frozen BERT-base; returns the hidden states of the last layer."""

    def __init__(self, name='bert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(name, output_hidden_states=True, trust_remote_code=True)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                         return_dict=True)
        return out['hidden_states'][-1]


class VisionEncoder(nn.Module):
    """ConvNeXt-Tiny (ImageNet weights); returns the feature maps of its four stages."""

    def __init__(self, name='facebook/convnext-tiny-224'):
        super().__init__()
        self.model = AutoModel.from_pretrained(name, output_hidden_states=True, trust_remote_code=True)

    def forward(self, x):
        return self.model(x, output_hidden_states=True)['hidden_states'][1:]


class SemCurSegNet(nn.Module):
    """Text-guided segmentation network of SemCurSeg (Fig. 1, Stage 2)."""

    def __init__(self, text_encoder='bert-base-uncased', vision_encoder='facebook/convnext-tiny-224',
                 use_apfo=True):
        super().__init__()
        self.text_encoder = TextEncoder(text_encoder)
        self.vision_encoder = VisionEncoder(vision_encoder)
        self.apfo = APFO() if use_apfo else None
        self.decoder = TVHADecoder()
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, image, text, pseudo_mask=None):
        """image (B, 3, H, W), text {'input_ids', 'attention_mask'}, pseudo_mask (B, 1, H, W) or None.
        Returns lesion probabilities (B, 1, H, W)."""
        vis = self.vision_encoder(image)
        txt = self.text_encoder(text['input_ids'], text['attention_mask'])
        if self.apfo is not None:
            txt, _ = self.apfo(txt, text['attention_mask'], vis[-1], pseudo_mask)
        out = self.decoder(vis, txt, txt_padding_mask=(text['attention_mask'] == 0))
        return self.out(out).sigmoid()

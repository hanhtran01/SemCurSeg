"""SCGM phrase-specific loss weighting (Sec. 3.3.4) for Grounding DINO (MMDetection 3.3).

The grounding (classification) loss of Grounding DINO is a sigmoid focal loss over query-token
pairs. SCGM splits the caption tokens into levels: object tokens (and tokens outside any lesion
phrase) belong to level 0, the attribute words of a lesion to level 1, and the remaining words of
its spatial phrase to level 2. With L_k the loss over the tokens of level k,

    L_SCGM = sum_k w_k L_k,    w_k proportional to 1 / a_k,

where a_k is a running estimate of the grounding accuracy of level k: the fraction of positive
query-token pairs of that level whose predicted probability exceeds 0.5. Only the levels already
introduced by the curriculum are weighted, and the weights of the active levels average to one,
so a level gains weight only after it has been introduced and while it is grounded less
accurately than the others.

Implementation: the detector computes the level of every token and the per-token weights before
the head loss is evaluated; the head flattens them in the same order as its masked logits; the
focal loss multiplies its element weights by them and updates a_k.
"""
import torch

from mmdet.models.dense_heads import GroundingDINOHead
from mmdet.models.detectors import GroundingDINO
from mmdet.models.losses import FocalLoss
from mmdet.registry import MODELS


@MODELS.register_module()
class PhraseWeightedFocalLoss(FocalLoss):
    """Sigmoid focal loss with per-token weights and running per-level grounding accuracy."""

    def __init__(self, *args, num_levels=3, acc_momentum=0.99, init_acc=1.0, min_acc=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_levels = num_levels
        self.acc_momentum = acc_momentum
        self.min_acc = min_acc
        # a_k; kept as a buffer so that it is saved with the checkpoint. Under DDP the buffers of
        # rank 0 are broadcast at every forward pass, so all ranks use the same estimate.
        self.register_buffer('level_acc', torch.full((num_levels,), float(init_acc)))
        self.pending_weight = None
        self.pending_level = None

    def level_weights(self, num_active):
        acc = self.level_acc[:num_active].clamp(min=self.min_acc)
        inv = 1.0 / acc
        return inv * num_active / inv.sum()

    def set_pending(self, weight, level):
        self.pending_weight, self.pending_level = weight, level

    def clear_pending(self):
        self.pending_weight, self.pending_level = None, None

    @torch.no_grad()
    def _update_accuracy(self, pred, target, level):
        positive = target > 0.5
        hit = pred.sigmoid() > 0.5
        for k in range(self.num_levels):
            sel = positive & (level == k)
            n = sel.sum()
            if n > 0:
                acc = (hit & sel).sum().float() / n.float()
                self.level_acc[k] = self.acc_momentum * self.level_acc[k] + (1 - self.acc_momentum) * acc

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        token_weight = self.pending_weight
        if token_weight is not None and token_weight.numel() == pred.numel():
            token_weight = token_weight.view_as(pred).to(pred.dtype)
            weight = token_weight if weight is None else weight * token_weight
            if self.training and self.pending_level is not None:
                self._update_accuracy(pred.detach(), target, self.pending_level.view_as(pred))
        return super().forward(pred, target, weight=weight, avg_factor=avg_factor,
                               reduction_override=reduction_override)


@MODELS.register_module()
class PhraseWeightedGroundingDINOHead(GroundingDINOHead):
    """Grounding DINO head that passes per-token weights to PhraseWeightedFocalLoss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_weights = None   # (batch, max_text_len), set by CurriculumGroundingDINO.loss
        self.token_levels = None

    def _flatten_like_logits(self, per_token, num_queries):
        # Same masking as GroundingDINOHead.loss_by_feat_single: padded text positions are dropped.
        text_masks = self.text_masks.new_zeros((self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1).repeat(1, num_queries, 1)
        values = per_token.unsqueeze(1).repeat(1, num_queries, 1)
        return torch.masked_select(values, text_mask)

    def _arm(self, num_queries):
        if self.token_weights is None or not isinstance(self.loss_cls, PhraseWeightedFocalLoss):
            return
        self.loss_cls.set_pending(
            self._flatten_like_logits(self.token_weights, num_queries),
            self._flatten_like_logits(self.token_levels, num_queries))

    def _disarm(self):
        if isinstance(self.loss_cls, PhraseWeightedFocalLoss):
            self.loss_cls.clear_pending()

    def loss_by_feat_single(self, cls_scores, *args, **kwargs):
        self._arm(cls_scores.size(1))
        try:
            return super().loss_by_feat_single(cls_scores, *args, **kwargs)
        finally:
            self._disarm()

    def _loss_dn_single(self, dn_cls_scores, *args, **kwargs):
        self._arm(dn_cls_scores.size(1))
        try:
            return super()._loss_dn_single(dn_cls_scores, *args, **kwargs)
        finally:
            self._disarm()


@MODELS.register_module()
class CurriculumGroundingDINO(GroundingDINO):
    """Grounding DINO trained with SCGM curriculum captions and phrase-specific loss weighting."""

    def __init__(self, *args, use_phrase_weighting=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_phrase_weighting = use_phrase_weighting
        self.curriculum_level = 2   # CurriculumPhraseHook sets the level of the current epoch

    def set_curriculum_level(self, level):
        self.curriculum_level = int(level)

    def _token_levels(self, data_sample):
        """Level of every token position of the caption (0 outside lesion phrases)."""
        max_len = self.bbox_head.max_text_len
        levels = torch.zeros(max_len, dtype=torch.long)
        level_spans = data_sample.metainfo.get('level_tokens_positive') or {}
        if not level_spans:
            return levels
        tokenized = self.language_model.tokenizer(
            [data_sample.text],
            padding='max_length' if self.language_model.pad_to_max else 'longest',
            return_tensors='pt')
        top = max(len(v) for v in level_spans.values()) - 1
        # phrases are nested, so assign from the outermost level inwards
        for k in range(top, -1, -1):
            spans = [v[k] for v in level_spans.values() if len(v) > k]
            if not spans:
                continue
            _, positive_map = self.get_positive_map(tokenized, spans)
            covered = (positive_map > 0).any(dim=0)[:max_len]
            levels[:covered.numel()][covered] = k
        return levels

    def loss(self, batch_inputs, batch_data_samples):
        head = self.bbox_head
        armed = (self.use_phrase_weighting
                 and isinstance(head, PhraseWeightedGroundingDINOHead)
                 and isinstance(head.loss_cls, PhraseWeightedFocalLoss))
        if armed:
            num_active = min(self.curriculum_level, head.loss_cls.num_levels - 1) + 1
            levels = torch.stack([self._token_levels(ds) for ds in batch_data_samples])
            levels = levels.clamp(max=num_active - 1).to(batch_inputs.device)
            weights = head.loss_cls.level_weights(num_active).to(batch_inputs.device)
            head.token_levels = levels
            head.token_weights = weights[levels]
        try:
            losses = super().loss(batch_inputs, batch_data_samples)
        finally:
            head.token_levels = None
            head.token_weights = None
        if armed:
            for k in range(head.loss_cls.num_levels):
                losses[f'scgm_acc_l{k}'] = head.loss_cls.level_acc[k].detach().clone()
        return losses

"""Main and explicitly supervised category losses, shared by training and tests."""

import torch
from torch.nn import functional as F


def segmentation_loss(output, target, weight=None, category_weight=0.4,
                      pan_weight=0.2, decoder_weight=0.0, ignore_index=6):
    if min(category_weight, pan_weight, decoder_weight) < 0:
        raise ValueError("Auxiliary loss weights must be nonnegative.")
    if isinstance(output, tuple):
        logits, auxiliary = output
        if torch.is_tensor(auxiliary):
            auxiliary = {"decoder": auxiliary}
    else:
        logits, auxiliary = output, {}

    def cross_entropy(prediction):
        prediction = F.interpolate(prediction, size=target.shape[-2:],
                                   mode="bilinear", align_corners=False)
        # Avoid NaN for a crop containing only ignored pixels.
        if not (target != ignore_index).any():
            return prediction.sum() * 0
        return F.cross_entropy(prediction, target, weight=weight,
                               ignore_index=ignore_index)

    def mean_loss(predictions):
        if not predictions:
            return logits.new_zeros(())
        return torch.stack([cross_entropy(p) for p in predictions]).mean()

    losses = {"main": cross_entropy(logits)}
    losses["rgb"] = mean_loss(auxiliary.get("rgb", [])) if category_weight else logits.new_zeros(())
    losses["dsm"] = mean_loss(auxiliary.get("dsm", [])) if category_weight else logits.new_zeros(())
    losses["pan"] = mean_loss(auxiliary.get("pan", [])) if pan_weight else logits.new_zeros(())
    decoder = auxiliary.get("decoder")
    losses["decoder"] = cross_entropy(decoder) if decoder_weight and decoder is not None else logits.new_zeros(())
    total = (losses["main"] + category_weight * (losses["rgb"] + losses["dsm"])
             + pan_weight * losses["pan"] + decoder_weight * losses["decoder"])
    return total, losses

import torch


def charbonnier_loss(
    pred, target, weight=None, reduction="mean", sample_wise=False, eps=1e-12
):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target) ** 2 + eps).mean()

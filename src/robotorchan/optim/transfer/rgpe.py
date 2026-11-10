"""Ranking-based source model weights for transfer Bayesian optimization."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models.model import Model
from torch import Tensor


def rgpe_weights(
    source_models: Sequence[Model],
    target_x: Tensor,
    target_y: Tensor,
    target_loo_predictions: Tensor,
    *,
    n_bootstrap: int = 256,
    seed: int = 0,
) -> Tensor:
    """Estimate RGPE weights from pairwise ranking loss on target observations.

    The target predictions must be leave-one-out (LOO) predictions, not
    in-sample fitted predictions, to avoid favoring the target model.
    Source predictions are posterior means at target_x. Bootstrap samples
    pairs of target observations and assigns each replicate to the model
    with the lowest ranking loss. Ties split weight equally.

    Returns weights ordered as [*source_models, target_model]. A source
    with worse full-data ranking loss than the target is excluded to
    reduce negative transfer. All tensors must use compatible devices.
    """
    if target_x.ndim != 2 or target_x.shape[0] < 2:
        raise ValueError("target_x must have shape (n, d) with n >= 2")
    n = target_x.shape[0]
    y = target_y.reshape(-1)
    loo = target_loo_predictions.reshape(-1)
    if y.numel() != n or loo.numel() != n:
        raise ValueError("target_y and target_loo_predictions need n values")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not bool(torch.isfinite(y).all() and torch.isfinite(loo).all()):
        raise ValueError("target values and predictions must be finite")

    predictions = []
    for model in source_models:
        with torch.no_grad():
            mean = model.posterior(target_x).mean.reshape(-1)
        if mean.numel() != n or not bool(torch.isfinite(mean).all()):
            raise ValueError("source model must predict finite scalar means")
        predictions.append(mean)
    predictions.append(loo)
    pred = torch.stack(predictions)

    i, j = torch.triu_indices(n, n, offset=1, device=y.device)
    # Ignore equal observed outcomes: they have no ranking preference.
    informative = y[i] != y[j]
    i, j = i[informative], j[informative]
    if i.numel() == 0:
        weights = y.new_zeros(len(predictions))
        weights[-1] = 1
        return weights

    actual = torch.sign(y[i] - y[j])
    predicted = torch.sign(pred[:, i] - pred[:, j])
    # A predicted tie is counted as an incorrect ranking.
    losses = (predicted != actual).to(dtype=y.dtype)
    full_loss = losses.mean(dim=-1)
    eligible = full_loss <= full_loss[-1]
    eligible[-1] = True

    generator = torch.Generator(device=y.device).manual_seed(seed)
    counts = y.new_zeros(len(predictions))
    for _ in range(n_bootstrap):
        indices = torch.randint(
            i.numel(), (i.numel(),), generator=generator, device=y.device
        )
        batch_loss = losses[:, indices].mean(dim=-1)
        best = torch.where(
            eligible, batch_loss, torch.full_like(batch_loss, float("inf"))
        )
        winners = best == best.min()
        counts += winners.to(y.dtype) / winners.sum()
    return counts / n_bootstrap

"""Sequential mixed-variable preference learning with pairwise feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from torch import Tensor

from robotorchan.models.preference.pairwise import MixedPairwiseGP
from robotorchan.optim.preference import optimize_mixed_preference_pair


def run_mixed_preference_loop(
    datapoints: Tensor,
    comparisons: Tensor,
    *,
    cat_dims: list[int],
    bounds: Tensor,
    fixed_features_list: list[dict[int, float]],
    feedback: Callable[[Tensor], int],
    steps: int = 1,
    fit_optimizer_kwargs: dict[str, Any] | None = None,
    num_restarts: int = 2,
    raw_samples: int = 16,
    options: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Fit, propose a mixed q=2 pair, and append winner/loser feedback.

    feedback(pair) returns the index (0 or 1) of the preferred candidate.
    Comparison indices refer to rows in the returned datapoints tensor.
    The caller owns persistence and may resume with the returned history.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    points = datapoints.detach().clone()
    preferences = comparisons.detach().clone()
    proposed_pairs: list[Tensor] = []
    for _ in range(steps):
        model = MixedPairwiseGP(points, preferences, cat_dims=cat_dims)
        model.fit(optimizer_kwargs=fit_optimizer_kwargs)
        model.eval()
        acquisition = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        pair, _ = optimize_mixed_preference_pair(
            acquisition,
            bounds,
            fixed_features_list,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
        )
        pair = pair.detach()
        winner = feedback(pair.clone())
        if winner not in (0, 1) or isinstance(winner, bool):
            raise ValueError("feedback must return the winner index 0 or 1.")
        offset = points.shape[0]
        points = torch.cat((points, pair), dim=0)
        comparison = torch.tensor(
            [[offset + winner, offset + (1 - winner)]],
            dtype=preferences.dtype,
            device=preferences.device,
        )
        preferences = torch.cat((preferences, comparison), dim=0)
        proposed_pairs.append(pair)
    return points, preferences, proposed_pairs


def run_mixed_preference_batch_loop(
    datapoints: Tensor,
    comparisons: Tensor,
    *,
    cat_dims: list[int],
    bounds: Tensor,
    fixed_features_list: list[dict[int, float]],
    feedback: Callable[[Tensor], list[int]],
    pairs_per_step: int,
    steps: int = 1,
    fit_optimizer_kwargs: dict[str, Any] | None = None,
    num_restarts: int = 2,
    raw_samples: int = 16,
    options: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Propose independent q=2 pairs per round and collect pairwise feedback.

    The model is fitted once per round. Each pair is optimized independently
    against the same posterior; no joint q>2 acquisition or listwise ranking
    is assumed. Diversity and duplicate handling are separate concerns.
    feedback(pairs) receives shape (pairs_per_step, 2, d) and returns one
    winner index (0 or 1) for each pair.
    """
    if pairs_per_step < 1:
        raise ValueError("pairs_per_step must be at least 1.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    points = datapoints.detach().clone()
    preferences = comparisons.detach().clone()
    proposed_batches: list[Tensor] = []
    for _ in range(steps):
        model = MixedPairwiseGP(points, preferences, cat_dims=cat_dims)
        model.fit(optimizer_kwargs=fit_optimizer_kwargs)
        model.eval()
        acquisition = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        pairs = []
        for _ in range(pairs_per_step):
            pair, _ = optimize_mixed_preference_pair(
                acquisition,
                bounds,
                fixed_features_list,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
            pairs.append(pair.detach())
        batch = torch.stack(pairs)
        winners = feedback(batch.clone())
        if len(winners) != pairs_per_step or any(
            type(winner) is not int or winner not in (0, 1) for winner in winners
        ):
            raise ValueError("feedback must return one winner index (0 or 1) per pair.")
        offset = points.shape[0]
        points = torch.cat((points, batch.reshape(-1, points.shape[-1])), dim=0)
        new_comparisons = torch.tensor(
            [
                [offset + 2 * i + winner, offset + 2 * i + 1 - winner]
                for i, winner in enumerate(winners)
            ],
            dtype=preferences.dtype,
            device=preferences.device,
        )
        preferences = torch.cat((preferences, new_comparisons), dim=0)
        proposed_batches.append(batch)
    return points, preferences, proposed_batches

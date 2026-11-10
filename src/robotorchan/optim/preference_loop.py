"""Sequential mixed-variable preference learning with pairwise feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from torch import Tensor

from robotorchan.models import MixedPairwiseGP
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

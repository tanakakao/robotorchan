"""Select target-task initial evaluations using source-task knowledge."""

from __future__ import annotations

import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples


def select_warm_start_points(
    bounds: Tensor,
    n: int,
    *,
    source_x: Tensor | None = None,
    source_datasets: list[tuple[Tensor, Tensor]] | None = None,
    source_y: Tensor | None = None,
    source_model: Model | None = None,
    maximize: bool = True,
    seed: int | None = None,
) -> Tensor:
    """Select unique feasible source points, filling the remainder with Sobol.

    Source objectives or predictions only rank candidates; they are never
    returned as target observations. Evaluate the returned points on the
    target task before fitting a target model.

    Args:
        bounds: Search bounds of shape (2, d).
        n: Number of target-task initial evaluations.
        source_x: Source candidates of shape (m, d) for a single source.
        source_datasets: Independent (X, Y) source tasks, interleaved by rank.
        source_y: Optional observed source objectives of shape (m,) or (m, 1).
        source_model: Optional fitted source model for posterior-mean ranking.
        maximize: Whether higher source values are preferred.
        seed: Sobol seed for filling the remaining slots.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape (2, d)")
    if n < 1:
        raise ValueError("n must be positive")
    if not bool(torch.all(bounds[0] < bounds[1])):
        raise ValueError("bounds must have strictly increasing limits")
    if source_datasets is not None:
        if source_x is not None or source_y is not None or source_model is not None:
            raise ValueError("source_datasets cannot be combined with single-source inputs")
        if not source_datasets:
            raise ValueError("source_datasets must not be empty")
        datasets = source_datasets
    else:
        if source_x is None or (source_y is None) == (source_model is None):
            raise ValueError("Provide source_x and exactly one of source_y and source_model")
        datasets = [(source_x, source_y)]

    ranked: list[Tensor] = []
    for x, y in datasets:
        if x.ndim != 2 or x.shape[1] != bounds.shape[1]:
            raise ValueError("source candidates must have shape (m, d)")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("source candidates must be finite")
        if y is not None:
            if y.ndim not in (1, 2) or (y.ndim == 2 and y.shape[1] != 1):
                raise ValueError("source objectives must have shape (m,) or (m, 1)")
            values = y.reshape(-1)
            if values.numel() != x.shape[0]:
                raise ValueError("source objectives must have one value per point")
        else:
            with torch.no_grad():
                values = source_model.posterior(x).mean.reshape(-1)
            if values.numel() != x.shape[0]:
                raise ValueError("source_model must have a scalar posterior mean")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("source rankings must be finite")
        inside = ((x >= bounds[0]) & (x <= bounds[1])).all(dim=-1)
        eligible = torch.nonzero(inside, as_tuple=False).flatten()
        order = eligible[torch.argsort(values[eligible], descending=maximize, stable=True)]
        ranked.append(x[order])

    # Interleave per-task ranks, rather than comparing raw objective scales.
    ordered_candidates = []
    for rank in range(max(len(x) for x in ranked)):
        for task_x in ranked:
            if rank < len(task_x):
                ordered_candidates.append(task_x[rank])
    selected: list[Tensor] = []
    for candidate in ordered_candidates:
        if not any(bool(torch.equal(candidate, existing)) for existing in selected):
            selected.append(candidate)
            if len(selected) == n:
                break

    if len(selected) < n:
        # Generate a deterministic, independent candidate pool. Exact duplicates
        # are rejected, including duplicates against source candidates.
        generated = 0
        while len(selected) < n:
            samples = draw_sobol_samples(
                bounds=bounds,
                n=max(16, 2 * (n - len(selected))),
                q=1,
                seed=None if seed is None else seed + generated,
            ).squeeze(-2)
            for candidate in samples:
                if not any(bool(torch.equal(candidate, existing)) for existing in selected):
                    selected.append(candidate)
                    if len(selected) == n:
                        break
            generated += 1
    return torch.stack(selected)

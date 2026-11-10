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
    source_x: Tensor,
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
        source_x: Source candidates of shape (m, d).
        source_y: Optional observed source objectives of shape (m,) or (m, 1).
        source_model: Optional fitted source model for posterior-mean ranking.
        maximize: Whether higher source values are preferred.
        seed: Sobol seed for filling the remaining slots.
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape (2, d)")
    if source_x.ndim != 2 or source_x.shape[1] != bounds.shape[1]:
        raise ValueError("source_x must have shape (m, d)")
    if n < 1:
        raise ValueError("n must be positive")
    if (source_y is None) == (source_model is None):
        raise ValueError("Provide exactly one of source_y and source_model")
    if not bool(torch.all(bounds[0] < bounds[1])):
        raise ValueError("bounds must have strictly increasing limits")
    if not bool(torch.isfinite(source_x).all()):
        raise ValueError("source_x must be finite")

    if source_y is not None:
        values = source_y.reshape(-1)
        if values.numel() != source_x.shape[0]:
            raise ValueError("source_y must have one value per source point")
    else:
        with torch.no_grad():
            values = source_model.posterior(source_x).mean.reshape(-1)
        if values.numel() != source_x.shape[0]:
            raise ValueError("source_model must have a scalar posterior mean")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("source rankings must be finite")

    inside = ((source_x >= bounds[0]) & (source_x <= bounds[1])).all(dim=-1)
    eligible = torch.nonzero(inside, as_tuple=False).flatten()
    ordered = eligible[torch.argsort(values[eligible], descending=maximize, stable=True)]
    selected: list[Tensor] = []
    for index in ordered.tolist():
        candidate = source_x[index]
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

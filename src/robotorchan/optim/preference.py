"""Joint mixed-variable optimization of an analytic pairwise acquisition."""

from __future__ import annotations

from itertools import product

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from torch import Tensor


class _FixedPairAcquisition(AcquisitionFunction):
    """Optimize continuous coordinates while fixing each candidate's categories."""

    def __init__(
        self,
        acquisition: AcquisitionFunction,
        dimension: int,
        continuous_dims: list[int],
        assignments: tuple[dict[int, float], dict[int, float]],
    ) -> None:
        super().__init__(model=acquisition.model)
        self.acquisition = acquisition
        self.dimension = dimension
        self.continuous_dims = continuous_dims
        self.assignments = assignments

    def forward(self, X: Tensor) -> Tensor:
        full = X.new_zeros(*X.shape[:-1], self.dimension)
        full[..., self.continuous_dims] = X
        for index, features in enumerate(self.assignments):
            for dim, value in features.items():
                full[..., index, dim] = value
        return self.acquisition(full)


def optimize_mixed_preference_pair(
    acquisition: AcquisitionFunction,
    bounds: Tensor,
    fixed_features_list: list[dict[int, float]],
    *,
    num_restarts: int = 2,
    raw_samples: int = 16,
    options: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize a q=2 preference pair jointly across categorical assignments.

    Enumerate assignments independently for both candidates; BoTorch's
    sequential q>1 mixed optimizer cannot evaluate analytic EUBO at q=1.
    This helper is intended for modest categorical spaces.
    """
    if not fixed_features_list:
        raise ValueError("fixed_features_list must not be empty.")
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError("bounds must have shape (2, d).")
    categorical_dims = set(fixed_features_list[0])
    if any(set(features) != categorical_dims for features in fixed_features_list):
        raise ValueError("All assignments must fix the same categorical dimensions.")
    continuous_dims = [i for i in range(bounds.shape[1]) if i not in categorical_dims]
    best_pair: Tensor | None = None
    best_value: Tensor | None = None
    for assignments in product(fixed_features_list, repeat=2):
        if continuous_dims:
            reduced = _FixedPairAcquisition(
                acquisition, bounds.shape[1], continuous_dims, assignments
            )
            x, _ = optimize_acqf(
                acq_function=reduced,
                bounds=bounds[:, continuous_dims],
                q=2,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
            )
            pair = bounds.new_zeros(2, bounds.shape[1])
            pair[:, continuous_dims] = x
        else:
            pair = bounds.new_zeros(2, bounds.shape[1])
        for index, features in enumerate(assignments):
            for dim, value in features.items():
                pair[index, dim] = value
        with torch.no_grad():
            value = acquisition(pair.unsqueeze(0)).reshape(-1)[0]
        if best_value is None or value.item() > best_value.item():
            best_pair, best_value = pair.detach(), value.detach()
    assert best_pair is not None and best_value is not None
    return best_pair, best_value

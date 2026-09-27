"""PyTorch acquisition optimizer backend.

This module keeps BoTorch acquisition optimization semantics and swaps only
the numerical candidate generator for ``gen_candidates_torch``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.generation.gen import gen_candidates_torch
from botorch.optim import optimize_acqf as botorch_optimize_acqf
from torch import Tensor

from robotorchan.optim.constraints import CandidateConstraints

TorchOptimizerName = Literal["adam", "adamw", "sgd"]

_TORCH_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def optimize_acqf_torch(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int | None,
    *,
    optimizer: TorchOptimizerName | type[torch.optim.Optimizer] = "adam",
    options: dict[str, Any] | None = None,
    constraints: CandidateConstraints | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
    batch_initial_conditions: Tensor | None = None,
    timeout_sec: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimize a BoTorch acquisition function with a ``torch.optim`` backend.

    BoTorch remains responsible for restart initialization and best-candidate
    selection. ``gen_candidates_torch`` performs differentiable local optimization.

    Candidate constraints are rejected explicitly for now. Constraint handling
    belongs to the dedicated cross-optimizer constraint phase.
    """
    if q < 1:
        raise ValueError("q must be at least 1.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least 1.")
    candidate_constraints = constraints or CandidateConstraints()
    if candidate_constraints.has_constraints:
        raise ValueError(
            "The PyTorch optimizer backend does not support candidate constraints yet."
        )

    optimizer_class = _resolve_torch_optimizer(optimizer)
    gen_candidates = partial(
        gen_candidates_torch,
        optimizer=optimizer_class,
        timeout_sec=timeout_sec,
    )
    return botorch_optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=None if options is None else dict(options),
        fixed_features=fixed_features,
        batch_initial_conditions=batch_initial_conditions,
        gen_candidates=gen_candidates,
    )


def _resolve_torch_optimizer(
    optimizer: TorchOptimizerName | type[torch.optim.Optimizer],
) -> type[torch.optim.Optimizer]:
    if isinstance(optimizer, str):
        try:
            return _TORCH_OPTIMIZERS[optimizer.lower()]
        except KeyError as error:
            supported = ", ".join(sorted(_TORCH_OPTIMIZERS))
            raise ValueError(
                f"Unknown torch optimizer {optimizer!r}. Supported optimizers: {supported}."
            ) from error
    if isinstance(optimizer, type) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer
    raise TypeError("optimizer must be a supported name or torch.optim.Optimizer subclass.")

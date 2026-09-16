"""Common interfaces for acquisition-function search strategies.

Search strategies are deliberately separate from surrogate models. A surrogate
models the objective and remains compatible with BoTorch acquisition functions;
a search strategy decides how an acquisition function is optimized over a
candidate domain.

The initial contract is intentionally limited to continuous, box-constrained
search. Later strategies may add latent embeddings, trust regions, or adaptive
subspaces without changing model posterior APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor


@dataclass(frozen=True)
class SearchResult:
    """Result returned by a search strategy.

    Args:
        candidates: Selected candidates in the original/public input space.
        acquisition_value: Acquisition value associated with the selected
            candidates when the strategy computes one. Strategies that do not
            expose a meaningful acquisition value may return ``None``.
        optimization_time: Wall-clock time spent selecting candidates in
            seconds. This excludes surrogate construction and fitting.
        metadata: Strategy-specific diagnostics. Stable public information
            belongs in explicit fields; this mapping is reserved for optional
            diagnostics such as restart counts or latent reconstruction error.
    """

    candidates: Tensor
    acquisition_value: Tensor | None
    optimization_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidates.ndim < 2:
            raise ValueError("candidates must have shape [..., q, d].")
        if self.optimization_time < 0:
            raise ValueError("optimization_time must be non-negative.")


class SearchStrategy(ABC):
    """Base class for acquisition-function search strategies.

    A strategy consumes an already constructed BoTorch acquisition function and
    returns candidates in the surrogate's public/original input space. The
    strategy must not mutate the surrogate's posterior contract or reducer
    lifecycle.

    Phase 1 defines only continuous box-constrained optimization. Constraint,
    mixed-variable, and stateful strategy APIs are intentionally deferred until
    their requirements are concrete.
    """

    def __init__(self, bounds: Tensor) -> None:
        self._validate_bounds(bounds)
        self.bounds = bounds.detach().clone()

    @property
    def input_dim(self) -> int:
        """Dimension of the public candidate space."""
        return self.bounds.shape[-1]

    @abstractmethod
    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize ``acq_function`` and return public-space candidates."""

    @staticmethod
    def _validate_bounds(bounds: Tensor) -> None:
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must have shape [2, d].")
        if bounds.shape[1] < 1:
            raise ValueError("bounds must contain at least one input dimension.")
        if not (bounds[0] < bounds[1]).all():
            raise ValueError("Every lower bound must be strictly below its upper bound.")

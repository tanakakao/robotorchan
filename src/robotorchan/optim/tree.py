"""Gradient-free acquisition search for tree-ensemble surrogates."""

from __future__ import annotations

from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.acquisition.non_gp import validate_non_gp_acquisition
from robotorchan.optim.base import SearchResult
from robotorchan.optim.random import RandomSearchStrategy


class TreeEnsembleSearchStrategy(RandomSearchStrategy):
    """Random-search optimizer specialized for non-GP tree surrogates.

    The strategy intentionally reuses the generic random-search implementation
    instead of pretending that sklearn tree acquisitions are differentiable.
    It adds capability validation so a tree-search path cannot silently accept
    an analytic Gaussian acquisition or a gradient-based surrogate contract.
    """

    def __init__(
        self,
        bounds: Tensor,
        *,
        num_samples: int = 4096,
        seed: int | None = None,
    ) -> None:
        super().__init__(bounds, num_samples=num_samples, seed=seed)

    def optimize(
        self,
        acq_function: AcquisitionFunction,
        *,
        q: int = 1,
    ) -> SearchResult:
        """Optimize a compatible MC acquisition without candidate gradients."""
        model = acq_function.model
        validate_non_gp_acquisition(model, acq_function)
        if getattr(model, "supports_input_gradients", True):
            raise TypeError(
                "TreeEnsembleSearchStrategy expects a surrogate without candidate-input gradients."
            )
        result = super().optimize(acq_function, q=q)
        metadata = dict(result.metadata)
        metadata["strategy"] = "tree_ensemble_random_search"
        return SearchResult(
            candidates=result.candidates,
            acquisition_value=result.acquisition_value,
            metadata=metadata,
        )

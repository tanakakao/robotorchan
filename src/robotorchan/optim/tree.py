"""Gradient-free acquisition search for tree-ensemble surrogates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
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
        integer_dims: Sequence[int] = (),
        categorical_values: Mapping[int, Sequence[float]] | None = None,
    ) -> None:
        super().__init__(bounds, num_samples=num_samples, seed=seed)
        self.integer_dims = self._normalize_dims(integer_dims)
        raw_categorical = dict(categorical_values or {})
        normalized_categorical = self._normalize_dims(raw_categorical)
        self.categorical_values = {
            dim: raw_categorical[raw_dim]
            for dim, raw_dim in zip(normalized_categorical, raw_categorical, strict=True)
        }
        structured = set(self.integer_dims) | set(self.categorical_values)
        if set(self.integer_dims) & set(self.categorical_values):
            raise ValueError("A dimension cannot be both integer and categorical.")
        if any(len(values) == 0 for values in self.categorical_values.values()):
            raise ValueError("Each categorical dimension must provide at least one value.")
        for dim, values in self.categorical_values.items():
            tensor = torch.as_tensor(values, dtype=self.bounds.dtype, device=self.bounds.device)
            if not torch.isfinite(tensor).all() or tensor.unique().numel() != tensor.numel():
                raise ValueError("Categorical values must be finite and unique.")
            if (tensor < self.bounds[0, dim]).any() or (tensor > self.bounds[1, dim]).any():
                raise ValueError("Categorical values must lie within bounds.")
        for dim in self.integer_dims:
            if torch.ceil(self.bounds[0, dim]) > torch.floor(self.bounds[1, dim]):
                raise ValueError("Integer dimensions must contain at least one feasible integer.")

    def _normalize_dims(self, dims: Sequence[int] | Mapping[int, object]) -> tuple[int, ...]:
        normalized = tuple(dim % self.input_dim if dim < 0 else dim for dim in dims)
        if any(dim < 0 or dim >= self.input_dim for dim in normalized):
            raise ValueError("Structured search dimensions contain an invalid index.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Structured search dimensions must not contain duplicates.")
        return normalized

    def _sample_candidate_batches(self, q: int) -> Tensor:
        samples = super()._sample_candidate_batches(q)
        for dim in self.integer_dims:
            lower = torch.ceil(self.bounds[0, dim]).to(torch.long)
            upper = torch.floor(self.bounds[1, dim]).to(torch.long)
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=samples.device)
                generator.manual_seed(self.seed + dim + 1)
            samples[..., dim] = torch.randint(
                int(lower),
                int(upper) + 1,
                samples.shape[:-1],
                device=samples.device,
                generator=generator,
            ).to(samples.dtype)
        for dim, allowed in self.categorical_values.items():
            values = torch.as_tensor(allowed, dtype=samples.dtype, device=samples.device)
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=samples.device)
                generator.manual_seed(self.seed + dim + 1)
            indices = torch.randint(
                len(allowed),
                samples.shape[:-1],
                device=samples.device,
                generator=generator,
            )
            samples[..., dim] = values[indices]
        return samples

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
        if self.integer_dims:
            metadata["integer_dims"] = self.integer_dims
        if self.categorical_values:
            metadata["categorical_dims"] = tuple(sorted(self.categorical_values))
        return SearchResult(
            candidates=result.candidates,
            acquisition_value=result.acquisition_value,
            metadata=metadata,
        )

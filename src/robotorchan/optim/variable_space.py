"""Mixed-variable search-space contract for candidate optimizers.

The contract describes public/raw candidate coordinates and is independent of
surrogate-model categorical kernels and numerical optimizer backends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MixedVariableSpace:
    """Describe continuous, integer, and categorical candidate dimensions."""

    bounds: Tensor
    integer_dims: tuple[int, ...] = ()
    categorical_values: Mapping[int, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        bounds = self.bounds
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] < 1:
            raise ValueError("bounds must have shape [2, d] with d >= 1.")
        if not bounds.is_floating_point():
            raise TypeError("bounds must use a floating dtype.")
        if not torch.isfinite(bounds).all():
            raise ValueError("bounds must contain only finite values.")
        if not torch.all(bounds[0] < bounds[1]):
            raise ValueError("Every lower bound must be strictly below its upper bound.")

        integer_dims = tuple(self.integer_dims)
        categories = dict(self.categorical_values or {})
        d = bounds.shape[-1]
        structured_dims = integer_dims + tuple(categories)
        if len(structured_dims) != len(set(structured_dims)):
            raise ValueError(
                "Each dimension must have exactly one continuous, integer, or "
                "categorical owner."
            )
        if any(dim < 0 or dim >= d for dim in structured_dims):
            raise ValueError("Variable dimensions must lie within the input dimension.")

        for dim in integer_dims:
            lower = torch.ceil(bounds[0, dim])
            upper = torch.floor(bounds[1, dim])
            if lower > upper:
                raise ValueError(
                    f"Integer dimension {dim} has no legal value within its bounds."
                )

        normalized_categories: dict[int, tuple[float, ...]] = {}
        for dim, values in categories.items():
            values_tuple = tuple(float(value) for value in values)
            if not values_tuple:
                raise ValueError(f"categorical_values[{dim}] must not be empty.")
            if len(values_tuple) != len(set(values_tuple)):
                raise ValueError(f"categorical_values[{dim}] must contain unique values.")
            values_tensor = torch.as_tensor(
                values_tuple,
                dtype=bounds.dtype,
                device=bounds.device,
            )
            if not torch.isfinite(values_tensor).all():
                raise ValueError(f"categorical_values[{dim}] must contain finite values.")
            if torch.any(values_tensor < bounds[0, dim]) or torch.any(
                values_tensor > bounds[1, dim]
            ):
                raise ValueError(f"categorical_values[{dim}] must lie within bounds.")
            normalized_categories[dim] = values_tuple

        object.__setattr__(self, "bounds", bounds.detach().clone())
        object.__setattr__(self, "integer_dims", integer_dims)
        object.__setattr__(self, "categorical_values", normalized_categories)

    @property
    def input_dim(self) -> int:
        """Number of public candidate dimensions."""
        return self.bounds.shape[-1]

    @property
    def categorical_dims(self) -> tuple[int, ...]:
        """Categorical dimensions in insertion order."""
        return tuple(self.categorical_values or {})

    @property
    def continuous_dims(self) -> tuple[int, ...]:
        """Dimensions with ordinary continuous semantics."""
        structured = set(self.integer_dims) | set(self.categorical_dims)
        return tuple(dim for dim in range(self.input_dim) if dim not in structured)

    @property
    def is_mixed(self) -> bool:
        """Whether more than one variable kind is present."""
        kinds = (
            bool(self.continuous_dims),
            bool(self.integer_dims),
            bool(self.categorical_dims),
        )
        return sum(kinds) > 1

    def validate_fixed_features(
        self,
        fixed_features: Mapping[int, float | Tensor] | None,
    ) -> None:
        """Validate fixed values against variable-space semantics."""
        if not fixed_features:
            return
        for dim, raw_value in fixed_features.items():
            if dim < 0 or dim >= self.input_dim:
                raise ValueError(f"fixed_features dimension {dim} is out of range.")
            value = torch.as_tensor(
                raw_value,
                dtype=self.bounds.dtype,
                device=self.bounds.device,
            )
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError(f"fixed_features[{dim}] must be one finite scalar.")
            scalar = value.reshape(())
            if scalar < self.bounds[0, dim] or scalar > self.bounds[1, dim]:
                raise ValueError(f"fixed_features[{dim}] must lie within bounds.")
            if dim in self.integer_dims and scalar != torch.round(scalar):
                raise ValueError(f"fixed_features[{dim}] must be an integer value.")
            if dim in self.categorical_dims:
                allowed = torch.as_tensor(
                    self.categorical_values[dim],
                    dtype=self.bounds.dtype,
                    device=self.bounds.device,
                )
                if not torch.any(scalar == allowed):
                    raise ValueError(
                        f"fixed_features[{dim}] must be one of the categorical values."
                    )

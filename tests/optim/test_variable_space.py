"""Tests for the mixed-variable search-space contract."""

import pytest
import torch

from robotorchan.optim import MixedVariableSpace


def test_variable_space_partitions_dimensions() -> None:
    space = MixedVariableSpace(
        torch.tensor([[0.0, 1.2, 0.0], [1.0, 4.8, 2.0]], dtype=torch.double),
        integer_dims=(1,),
        categorical_values={2: (0.0, 2.0)},
    )

    assert space.continuous_dims == (0,)
    assert space.integer_dims == (1,)
    assert space.categorical_dims == (2,)
    assert space.is_mixed


def test_variable_space_rejects_overlapping_dimension_ownership() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="exactly one"):
        MixedVariableSpace(
            bounds,
            integer_dims=(1,),
            categorical_values={1: (0.0, 1.0)},
        )


def test_variable_space_rejects_integer_domain_without_legal_value() -> None:
    bounds = torch.tensor([[0.0, 0.1], [1.0, 0.9]], dtype=torch.double)

    with pytest.raises(ValueError, match="no legal value"):
        MixedVariableSpace(bounds, integer_dims=(1,))


def test_variable_space_rejects_invalid_categorical_domain() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="unique"):
        MixedVariableSpace(bounds, categorical_values={1: (0.0, 0.0)})
    with pytest.raises(ValueError, match="within bounds"):
        MixedVariableSpace(bounds, categorical_values={1: (0.0, 3.0)})


def test_variable_space_validates_fixed_features() -> None:
    space = MixedVariableSpace(
        torch.tensor([[0.0, 1.2, 0.0], [1.0, 4.8, 2.0]], dtype=torch.double),
        integer_dims=(1,),
        categorical_values={2: (0.0, 2.0)},
    )

    space.validate_fixed_features({0: 0.5, 1: 3.0, 2: 2.0})

    with pytest.raises(ValueError, match="integer value"):
        space.validate_fixed_features({1: 2.5})
    with pytest.raises(ValueError, match="categorical values"):
        space.validate_fixed_features({2: 1.0})


def test_variable_space_clones_bounds() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds)
    bounds[0, 0] = -1.0

    assert space.bounds[0, 0] == 0.0

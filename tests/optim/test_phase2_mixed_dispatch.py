"""Regression tests for mixed-space optimizer dispatch."""

from unittest.mock import Mock

import pytest
import torch

from robotorchan.optim import dispatch
from robotorchan.optim.domains.variable_space import MixedVariableSpace


@pytest.fixture
def categorical_space():
    bounds = torch.tensor([[0.0], [2.0]], dtype=torch.double)
    return bounds, MixedVariableSpace(bounds, categorical_values={0: (0.0, 1.0, 2.0)})


def test_categorical_only_ga_uses_mixed_backend(monkeypatch, categorical_space):
    bounds, space = categorical_space
    backend = Mock(return_value=(torch.zeros(1, 1), torch.zeros(1)))
    monkeypatch.setattr(dispatch, "optimize_acqf_mixed_ga", backend)
    dispatch.optimize_acqf(None, bounds, 1, optimizer="ga", variable_space=space)
    assert backend.call_count == 1
    assert backend.call_args.kwargs["variable_space"] is space


def test_categorical_only_botorch_uses_mixed_backend(monkeypatch, categorical_space):
    bounds, space = categorical_space
    backend = Mock(return_value=(torch.zeros(1, 1), torch.zeros(1)))
    monkeypatch.setattr(dispatch, "optimize_acqf_mixed_botorch", backend)
    dispatch.optimize_acqf(
        None,
        bounds,
        1,
        optimizer="botorch",
        variable_space=space,
        optimizer_options={"fixed_features_list": [{0: 0.0}, {0: 2.0}]},
    )
    assert backend.call_count == 1
    assert backend.call_args.args[4] == [{0: 0.0}, {0: 2.0}]


def test_mixed_botorch_requires_explicit_discrete_configs(categorical_space):
    bounds, space = categorical_space
    with pytest.raises(ValueError, match="fixed_features_list"):
        dispatch.optimize_acqf(None, bounds, 1, variable_space=space)


def test_mixed_botorch_rejects_incomplete_configs(categorical_space):
    bounds, space = categorical_space
    with pytest.raises(ValueError, match="every discrete dimension"):
        dispatch.optimize_acqf(
            None,
            bounds,
            1,
            variable_space=space,
            optimizer_options={"fixed_features_list": [{}]},
        )


def test_mixed_botorch_rejects_invalid_category(categorical_space):
    bounds, space = categorical_space
    with pytest.raises(ValueError, match="categorical values"):
        dispatch.optimize_acqf(
            None,
            bounds,
            1,
            variable_space=space,
            optimizer_options={"fixed_features_list": [{0: 0.5}]},
        )

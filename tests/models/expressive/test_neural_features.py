"""Tests for shared expressive neural-feature utilities."""

import torch
from torch import nn

from robotorchan.models.expressive.neural_features import (
    make_feature_network,
    validate_feature_output,
    validate_neural_feature_config,
)


def test_make_feature_network_respects_shape_dtype_and_activation() -> None:
    network = make_feature_network(
        3,
        5,
        (7, 6),
        "silu",
        device=torch.device("cpu"),
        dtype=torch.double,
    )
    X = torch.randn(4, 3, dtype=torch.double)

    output = network(X)

    assert output.shape == (4, 5)
    assert output.dtype == torch.double
    assert any(isinstance(module, nn.SiLU) for module in network)


def test_reverse_feature_network_reverses_hidden_widths() -> None:
    network = make_feature_network(
        2,
        4,
        (8, 6),
        "relu",
        reverse=True,
        device=torch.device("cpu"),
        dtype=torch.double,
    )
    linear_layers = [module for module in network if isinstance(module, nn.Linear)]

    assert linear_layers[0].out_features == 6
    assert linear_layers[1].out_features == 8
    assert linear_layers[-1].out_features == 4


def test_neural_feature_config_and_output_validation() -> None:
    validate_neural_feature_config(2, (4,), "gelu", 1e-8)
    source = torch.randn(3, 5, dtype=torch.double)
    validate_feature_output(torch.randn(3, 2, dtype=torch.double), source, 2)

    invalid_configs = [
        (0, (4,), "gelu", 1e-8),
        (2, (0,), "gelu", 1e-8),
        (2, (4,), "unknown", 1e-8),
        (2, (4,), "gelu", 0.0),
    ]
    for args in invalid_configs:
        try:
            validate_neural_feature_config(*args)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {args}.")

    try:
        validate_feature_output(torch.randn(3, 3), source, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected output-dimension validation to fail.")

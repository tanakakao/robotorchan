"""Shared neural-feature utilities for expressive GP models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def validate_neural_feature_config(
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    eps: float,
) -> None:
    """Validate the common configuration for trainable neural features."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")
    if any(width <= 0 for width in hidden_dims):
        raise ValueError("hidden_dims must contain only positive integers.")
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {activation!r}. Choose from {sorted(_ACTIVATIONS)}."
        )
    if eps <= 0:
        raise ValueError("eps must be positive.")


def make_feature_network(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    *,
    reverse: bool = False,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Sequential:
    """Build the default MLP used by jointly trained neural GP models."""
    widths = tuple(reversed(hidden_dims)) if reverse else hidden_dims
    layers: list[nn.Module] = []
    previous = input_dim
    for width in widths:
        layers.extend([nn.Linear(previous, width), _ACTIVATIONS[activation]()])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers).to(device=device, dtype=dtype)


def validate_feature_output(
    feature_X: Tensor,
    source_X: Tensor,
    output_dim: int,
    *,
    name: str = "feature_extractor",
) -> None:
    """Validate that a feature module preserves batches and returns its contract."""
    if feature_X.shape[:-1] != source_X.shape[:-1]:
        raise ValueError(f"{name} must preserve all non-feature input dimensions.")
    if feature_X.shape[-1] != output_dim:
        raise ValueError(
            f"{name} output dimension must equal latent_dim; "
            f"expected {output_dim}, got {feature_X.shape[-1]}."
        )

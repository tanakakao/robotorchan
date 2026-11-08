"""Tests for optimizer capability metadata and runtime validation."""

import torch

from robotorchan.optim import get_optimizer_capabilities
from robotorchan.optim.backend_support.runtime import make_generator, validate_bounds


def test_public_optimizer_names_have_capabilities() -> None:
    for name in (
        "botorch",
        "torch_adam",
        "torch_adamw",
        "torch_sgd",
        "random",
        "sobol",
        "de",
        "cmaes",
        "ga",
        "pso",
        "hybrid",
    ):
        assert get_optimizer_capabilities(name).continuous


def test_torch_capabilities_advertise_nonlinear_penalty() -> None:
    from robotorchan.optim.capabilities import ConstraintHandling

    for name in ("torch_adam", "torch_adamw", "torch_sgd"):
        capabilities = get_optimizer_capabilities(name)
        assert capabilities.nonlinear_inequality_constraints
        assert capabilities.interpoint_nonlinear_constraints
        assert capabilities.constraint_handling.nonlinear_inequality == ConstraintHandling.PENALTY


def test_unseeded_generator_uses_global_rng() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    assert make_generator(bounds, None) is None


def test_validate_bounds_rejects_zero_dimensional_domain() -> None:
    try:
        validate_bounds(torch.empty(2, 0))
    except ValueError as exc:
        assert "at least one input dimension" in str(exc)
    else:
        raise AssertionError("Zero-dimensional bounds should be rejected.")

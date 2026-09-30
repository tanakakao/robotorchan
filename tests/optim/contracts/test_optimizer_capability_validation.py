"""Tests for optimizer capability metadata and runtime validation."""

import torch

from robotorchan.optim import get_optimizer_capabilities, optimize_acqf
from robotorchan.optim.backend_support.runtime import make_generator, validate_bounds
from robotorchan.optim.constraints import CandidateConstraints


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


def test_torch_dispatch_rejects_constraints_from_capabilities() -> None:
    bounds = torch.tensor([[0.0], [1.0]])
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.2),)
    )
    try:
        optimize_acqf(
            object(),  # type: ignore[arg-type]
            bounds,
            q=1,
            optimizer="torch_adam",
            constraints=constraints,
        )
    except NotImplementedError as exc:
        assert "linear inequalities" in str(exc)
    else:
        raise AssertionError("Unsupported constraints should fail before backend execution.")


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

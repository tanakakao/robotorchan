"""Sampling baselines for acquisition-function optimization."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor
from torch.quasirandom import SobolEngine

from robotorchan.optim.runtime import make_generator, validate_bounds
from robotorchan.optim.variable_space import MixedVariableSpace


def optimize_acqf_sampling(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    num_samples: int = 4096,
    method: str = "sobol",
    seed: int | None = None,
    variable_space: MixedVariableSpace | None = None,
    fixed_features: dict[int, float | Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the best acquisition-valued q-batch from sampled candidates."""
    validate_bounds(bounds)
    if q < 1:
        raise ValueError("q must be at least 1.")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    method = method.lower()
    if method not in {"random", "sobol"}:
        raise ValueError("method must be either 'random' or 'sobol'.")

    if variable_space is not None:
        if not torch.equal(variable_space.bounds, bounds):
            raise ValueError("variable_space bounds must match bounds.")
        variable_space.validate_fixed_features(fixed_features)
        if method != "random" and (variable_space.integer_dims or variable_space.categorical_dims):
            raise NotImplementedError(
                "Mixed-variable Sobol sampling is not defined in Phase 4; use method='random'."
            )

    if method == "sobol":
        samples = _draw_sobol(bounds, num_samples, q, seed)
    else:
        samples = _draw_random(bounds, num_samples, q, seed)
        if variable_space is not None:
            samples = _apply_mixed_random_semantics(samples, variable_space, seed)
    samples = _apply_fixed_features(samples, fixed_features)

    with torch.no_grad():
        values = acq_function(samples)
    if values.numel() != num_samples:
        raise ValueError(
            "Sampling optimization requires one scalar acquisition value per sampled q-batch."
        )
    scores = values.reshape(num_samples)
    selected = scores.argmax()
    return samples[selected], scores[selected]


def _draw_random(bounds: Tensor, num_samples: int, q: int, seed: int | None) -> Tensor:
    generator = make_generator(bounds, seed)
    unit = torch.rand(
        num_samples,
        q,
        bounds.shape[-1],
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    return bounds[0] + (bounds[1] - bounds[0]) * unit


def _draw_sobol(bounds: Tensor, num_samples: int, q: int, seed: int | None) -> Tensor:
    dimension = q * bounds.shape[-1]
    engine = SobolEngine(dimension=dimension, scramble=True, seed=seed)
    unit = engine.draw(num_samples, dtype=bounds.dtype).to(device=bounds.device)
    unit = unit.reshape(num_samples, q, bounds.shape[-1])
    return bounds[0] + (bounds[1] - bounds[0]) * unit


def _apply_mixed_random_semantics(
    samples: Tensor,
    variable_space: MixedVariableSpace,
    seed: int | None,
) -> Tensor:
    """Draw integer and categorical coordinates uniformly from their legal sets."""
    generator = make_generator(variable_space.bounds, seed)
    for dim in variable_space.integer_dims:
        lower = int(torch.ceil(variable_space.bounds[0, dim]).item())
        upper = int(torch.floor(variable_space.bounds[1, dim]).item())
        samples[..., dim] = torch.randint(
            lower,
            upper + 1,
            samples.shape[:-1],
            device=samples.device,
            generator=generator,
        ).to(dtype=samples.dtype)
    for dim in variable_space.categorical_dims:
        values = torch.as_tensor(
            variable_space.categorical_values[dim],
            dtype=samples.dtype,
            device=samples.device,
        )
        indices = torch.randint(
            0,
            values.numel(),
            samples.shape[:-1],
            device=samples.device,
            generator=generator,
        )
        samples[..., dim] = values[indices]
    return samples


def _apply_fixed_features(
    samples: Tensor,
    fixed_features: dict[int, float | Tensor] | None,
) -> Tensor:
    if not fixed_features:
        return samples
    for dim, value in fixed_features.items():
        if dim < 0 or dim >= samples.shape[-1]:
            raise ValueError(f"fixed_features dimension {dim} is out of range.")
        samples[..., dim] = torch.as_tensor(value, dtype=samples.dtype, device=samples.device)
    return samples

"""Sampling baselines for acquisition-function optimization."""

from __future__ import annotations

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor
from torch.quasirandom import SobolEngine

from robotorchan.optim.runtime import make_generator, validate_bounds


def optimize_acqf_sampling(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    *,
    num_samples: int = 4096,
    method: str = "sobol",
    seed: int | None = None,
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

    if method == "sobol":
        samples = _draw_sobol(bounds, num_samples, q, seed)
    else:
        samples = _draw_random(bounds, num_samples, q, seed)

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

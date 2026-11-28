"""Construct native GPyTorch covariance modules for temporal inputs."""

import math
from typing import Literal

import torch
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
    SpectralMixtureKernel,
)

TemporalKernelKind = Literal[
    "rbf",
    "matern",
    "exponential",
    "rq",
    "periodic",
    "locally_periodic",
    "linear",
    "spectral_mixture",
]


_EMPTY_BATCH_SHAPE = torch.Size()


def make_temporal_kernel(
    kind: TemporalKernelKind,
    *,
    time_dim: int = 0,
    batch_shape: torch.Size = _EMPTY_BATCH_SHAPE,
    period_length: float | None = None,
    lengthscale: float | None = None,
    local_lengthscale: float | None = None,
    nu: float = 2.5,
    num_mixtures: int = 4,
) -> Kernel:
    """Build a native covariance module on a single time coordinate.

    Lengths use the input time units, except the periodic kernel lengthscale,
    which controls smoothness within a cycle. Parameters remain trainable.
    Locally periodic covariance is the product of periodic and RBF kernels.
    Spectral mixture parameters use native initialization by default; callers
    may initialize them from training data using GPyTorch's public API.
    """
    if type(time_dim) is not int or time_dim < 0:
        raise ValueError("time_dim must be a nonnegative column index")
    for name, value in (
        ("period_length", period_length),
        ("lengthscale", lengthscale),
        ("local_lengthscale", local_lengthscale),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if period_length is not None and kind not in ("periodic", "locally_periodic"):
        raise ValueError("period_length requires a periodic kernel")
    if local_lengthscale is not None and kind != "locally_periodic":
        raise ValueError("local_lengthscale requires a locally periodic kernel")
    if lengthscale is not None and kind in ("linear", "spectral_mixture"):
        raise ValueError("this kernel does not expose a single lengthscale")
    kwargs = {"active_dims": (time_dim,), "ard_num_dims": 1, "batch_shape": batch_shape}
    if kind == "rbf":
        kernel = RBFKernel(**kwargs)
    elif kind in ("matern", "exponential"):
        kernel = MaternKernel(nu=0.5 if kind == "exponential" else nu, **kwargs)
    elif kind == "rq":
        kernel = RQKernel(**kwargs)
    elif kind in ("periodic", "locally_periodic"):
        kernel = PeriodicKernel(**kwargs)
        if period_length is not None:
            kernel.period_length = period_length
        if kind == "locally_periodic":
            envelope = RBFKernel(**kwargs)
            if local_lengthscale is not None:
                envelope.lengthscale = local_lengthscale
            if lengthscale is not None:
                kernel.lengthscale = lengthscale
            return ScaleKernel(kernel * envelope, batch_shape=batch_shape)
    elif kind == "linear":
        return LinearKernel(**kwargs)
    elif kind == "spectral_mixture":
        if type(num_mixtures) is not int or num_mixtures < 1:
            raise ValueError("num_mixtures must be a positive integer")
        return SpectralMixtureKernel(num_mixtures=num_mixtures, **kwargs)
    else:
        raise ValueError(f"unknown temporal kernel: {kind}")
    if lengthscale is not None:
        kernel.lengthscale = lengthscale
    return ScaleKernel(kernel, batch_shape=batch_shape)

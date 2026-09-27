"""Capability contract for acquisition-function optimizer backends.

The backend contract extends BoTorch acquisition optimization without replacing
BoTorch's public concepts. Backends consume an already constructed BoTorch
AcquisitionFunction and operate on Tensor bounds and q-batches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerCapabilities:
    """Capabilities exposed by an acquisition optimizer backend.

    These flags describe backend behavior rather than surrogate-model
    capabilities. Unsupported combinations should be rejected before numerical
    optimization starts instead of being approximated silently.
    """

    continuous: bool = True
    integer: bool = False
    categorical: bool = False
    mixed: bool = False
    requires_grad: bool = False
    linear_inequality_constraints: bool = False
    linear_equality_constraints: bool = False
    nonlinear_inequality_constraints: bool = False
    interpoint_nonlinear_constraints: bool = False
    q_batch: bool = True
    sequential: bool = False
    fixed_features: bool = False
    gpu: bool = False
    batch_evaluation: bool = False


BOTORCH_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    continuous=True,
    requires_grad=True,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    sequential=True,
    fixed_features=True,
    gpu=False,
)

BOTORCH_MIXED_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    continuous=True,
    integer=True,
    categorical=True,
    mixed=True,
    requires_grad=True,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=False,
    q_batch=True,
    fixed_features=True,
    gpu=False,
)

"""Capability contract for acquisition-function optimizer backends.

The backend contract extends BoTorch acquisition optimization without replacing
BoTorch's public concepts. Backends consume an already constructed BoTorch
AcquisitionFunction and operate on Tensor bounds and q-batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConstraintHandling(StrEnum):
    """How an optimizer enforces a supported candidate constraint."""

    NATIVE = "native"
    PENALTY = "penalty"
    FEASIBILITY_FIRST = "feasibility_first"
    REPAIR = "repair"
    PROJECTION = "projection"
    REJECTION = "rejection"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ConstraintHandlingCapabilities:
    """Constraint enforcement semantics for one optimizer backend."""

    linear_inequality: ConstraintHandling = ConstraintHandling.UNSUPPORTED
    linear_equality: ConstraintHandling = ConstraintHandling.UNSUPPORTED
    nonlinear_inequality: ConstraintHandling = ConstraintHandling.UNSUPPORTED
    interpoint_nonlinear: ConstraintHandling = ConstraintHandling.UNSUPPORTED


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
    constraint_handling: ConstraintHandlingCapabilities = ConstraintHandlingCapabilities()


BOTORCH_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.NATIVE,
        linear_equality=ConstraintHandling.NATIVE,
        nonlinear_inequality=ConstraintHandling.NATIVE,
        interpoint_nonlinear=ConstraintHandling.NATIVE,
    ),
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
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.NATIVE,
        linear_equality=ConstraintHandling.NATIVE,
        nonlinear_inequality=ConstraintHandling.NATIVE,
    ),
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


TORCH_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    continuous=True,
    requires_grad=True,
    q_batch=True,
    fixed_features=True,
    gpu=True,
    batch_evaluation=True,
)


SAMPLING_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    continuous=True,
    integer=True,
    categorical=True,
    mixed=True,
    requires_grad=False,
    q_batch=True,
    fixed_features=True,
    gpu=True,
    batch_evaluation=True,
)


DIFFERENTIAL_EVOLUTION_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.FEASIBILITY_FIRST,
        linear_equality=ConstraintHandling.FEASIBILITY_FIRST,
        nonlinear_inequality=ConstraintHandling.FEASIBILITY_FIRST,
        interpoint_nonlinear=ConstraintHandling.FEASIBILITY_FIRST,
    ),
    continuous=True,
    integer=True,
    categorical=False,
    mixed=True,
    requires_grad=False,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    gpu=False,
    batch_evaluation=False,
    fixed_features=True,
)


CMAES_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.PENALTY,
        linear_equality=ConstraintHandling.PENALTY,
        nonlinear_inequality=ConstraintHandling.PENALTY,
        interpoint_nonlinear=ConstraintHandling.PENALTY,
    ),
    continuous=True,
    requires_grad=False,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    gpu=False,
    batch_evaluation=False,
)


GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.FEASIBILITY_FIRST,
        linear_equality=ConstraintHandling.FEASIBILITY_FIRST,
        nonlinear_inequality=ConstraintHandling.FEASIBILITY_FIRST,
        interpoint_nonlinear=ConstraintHandling.FEASIBILITY_FIRST,
    ),
    continuous=True,
    requires_grad=False,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    gpu=True,
    batch_evaluation=True,
    fixed_features=True,
)


MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.FEASIBILITY_FIRST,
        linear_equality=ConstraintHandling.FEASIBILITY_FIRST,
        nonlinear_inequality=ConstraintHandling.FEASIBILITY_FIRST,
        interpoint_nonlinear=ConstraintHandling.FEASIBILITY_FIRST,
    ),
    continuous=True,
    integer=True,
    categorical=True,
    mixed=True,
    requires_grad=False,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    gpu=True,
    batch_evaluation=True,
)


HYBRID_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.PENALTY,
        linear_equality=ConstraintHandling.PENALTY,
        nonlinear_inequality=ConstraintHandling.PENALTY,
        interpoint_nonlinear=ConstraintHandling.PENALTY,
    ),
    continuous=True,
    requires_grad=True,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    sequential=False,
    fixed_features=False,
    gpu=False,
    batch_evaluation=False,
)


NSGA2_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    continuous=True,
    requires_grad=False,
    q_batch=False,
    gpu=True,
    batch_evaluation=True,
)


PSO_OPTIMIZER_CAPABILITIES = OptimizerCapabilities(
    constraint_handling=ConstraintHandlingCapabilities(
        linear_inequality=ConstraintHandling.PENALTY,
        linear_equality=ConstraintHandling.PENALTY,
        nonlinear_inequality=ConstraintHandling.PENALTY,
        interpoint_nonlinear=ConstraintHandling.PENALTY,
    ),
    continuous=True,
    integer=True,
    requires_grad=False,
    linear_inequality_constraints=True,
    linear_equality_constraints=True,
    nonlinear_inequality_constraints=True,
    interpoint_nonlinear_constraints=True,
    q_batch=True,
    gpu=True,
    batch_evaluation=True,
    fixed_features=True,
)


OPTIMIZER_CAPABILITIES: dict[str, OptimizerCapabilities] = {
    "botorch": BOTORCH_OPTIMIZER_CAPABILITIES,
    "torch_adam": TORCH_OPTIMIZER_CAPABILITIES,
    "torch_adamw": TORCH_OPTIMIZER_CAPABILITIES,
    "torch_sgd": TORCH_OPTIMIZER_CAPABILITIES,
    "random": SAMPLING_OPTIMIZER_CAPABILITIES,
    "sobol": SAMPLING_OPTIMIZER_CAPABILITIES,
    "de": DIFFERENTIAL_EVOLUTION_OPTIMIZER_CAPABILITIES,
    "cmaes": CMAES_OPTIMIZER_CAPABILITIES,
    "ga": GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
    "pso": PSO_OPTIMIZER_CAPABILITIES,
    "hybrid": HYBRID_OPTIMIZER_CAPABILITIES,
}


def get_optimizer_capabilities(name: str) -> OptimizerCapabilities:
    """Return capabilities for a public scalar optimizer name."""
    try:
        return OPTIMIZER_CAPABILITIES[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(OPTIMIZER_CAPABILITIES))
        raise ValueError(
            f"Unknown optimizer {name!r}. Supported optimizers: {supported}."
        ) from error

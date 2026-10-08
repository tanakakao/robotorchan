"""Resolved acquisition-composition plans for heterogeneous problem semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective, MCAcquisitionObjective
from botorch.models.model import Model as BoTorchModel
from botorch.sampling.base import MCSampler
from torch import Tensor

from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.models.classification.registry import CLASSIFICATION_MODEL_REGISTRY
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics.feasibility import (
    FeasibilityRepresentation,
    ProbabilityOfFeasibility,
    SampleProbabilityOfFeasibility,
    SampleResidualFeasibility,
)
from robotorchan.semantics.objectives import RegressionObjective, SemanticObjective
from robotorchan.semantics.probability import ClassificationProbabilityOfFeasibility
from robotorchan.semantics.problem import ProblemSemantics


@dataclass(frozen=True, slots=True)
class SampleShapeContract:
    """Resolved sample/posterior-batch/q shape contract for composition."""

    sample_shape: torch.Size
    batch_shape: torch.Size
    q: int

    @property
    def value_shape(self) -> torch.Size:
        """Return the required shape of one scalar sample-wise value."""
        return self.sample_shape + self.batch_shape + torch.Size([self.q])

    @classmethod
    def from_resolved_shapes(
        cls,
        *,
        sample_shape: torch.Size,
        batch_shape: torch.Size,
        q: int,
    ) -> SampleShapeContract:
        """Build a contract from already-resolved sampling and posterior shapes."""
        if q < 1:
            raise ValueError("q must be at least 1.")
        return cls(
            sample_shape=sample_shape,
            batch_shape=batch_shape,
            q=q,
        )


@dataclass(frozen=True, slots=True)
class ObjectiveBinding:
    """Bind one semantic objective to its owning heterogeneous model entry."""

    objective: SemanticObjective
    output_index: int
    entry_index: int
    local_output_index: int


@dataclass(frozen=True, slots=True)
class FeasibilityBinding:
    """Bind one feasibility representation to its owning model entry."""

    representation: FeasibilityRepresentation
    output_index: int
    entry_index: int
    local_output_index: int


@dataclass(frozen=True, slots=True)
class AcquisitionCompositionPlan:
    """Resolved semantic inputs for later BoTorch acquisition construction.

    The plan records ownership without constructing a shared posterior, sampler,
    objective, constraint, or acquisition function. Later composition policies
    decide how compatible bindings are mapped to native BoTorch interfaces.
    """

    objectives: tuple[ObjectiveBinding, ...]
    feasibility: tuple[FeasibilityBinding, ...]


def resolve_acquisition_composition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
) -> AcquisitionCompositionPlan:
    """Resolve heterogeneous problem semantics into acquisition-layer bindings."""
    semantics.validate(model)

    objective_bindings = []
    for objective, output_index in zip(
        semantics.objectives,
        semantics.resolve_objective_outputs(model),
        strict=True,
    ):
        entry_index, local_output_index = model.output_owner(output_index)
        objective_bindings.append(
            ObjectiveBinding(
                objective=objective,
                output_index=output_index,
                entry_index=entry_index,
                local_output_index=local_output_index,
            )
        )

    feasibility_bindings = []
    for representation, output_index in zip(
        semantics.feasibility_representations(model),
        semantics.resolve_constraint_outputs(model),
        strict=True,
    ):
        entry_index, local_output_index = model.output_owner(output_index)
        feasibility_bindings.append(
            FeasibilityBinding(
                representation=representation,
                output_index=output_index,
                entry_index=entry_index,
                local_output_index=local_output_index,
            )
        )

    return AcquisitionCompositionPlan(
        objectives=tuple(objective_bindings),
        feasibility=tuple(feasibility_bindings),
    )


@dataclass(frozen=True, slots=True)
class BoTorchObjectiveBridge:
    """Native BoTorch model and MC objective for one regression binding."""

    model: BoTorchModel
    objective: MCAcquisitionObjective


def make_botorch_objective_bridge(
    model: HeterogeneousModel,
    binding: ObjectiveBinding,
) -> BoTorchObjectiveBridge:
    """Adapt one regression objective binding to native BoTorch interfaces."""
    objective = binding.objective
    if not isinstance(objective, RegressionObjective):
        raise TypeError("BoTorch objective bridge currently supports RegressionObjective only.")

    output_index = objective.resolve_output(model)
    entry_index, local_output_index = model.output_owner(output_index)
    if (
        output_index != binding.output_index
        or entry_index != binding.entry_index
        or local_output_index != binding.local_output_index
    ):
        raise ValueError("ObjectiveBinding does not match the current heterogeneous model.")

    entry_model = model[entry_index]
    if not isinstance(entry_model, BoTorchModel):
        raise TypeError("Regression objective owner must be a BoTorch Model.")

    return BoTorchObjectiveBridge(
        model=entry_model,
        objective=objective.to_botorch(model),
    )


def make_deterministic_pof_acquisition(
    objective_acquisition: AcquisitionFunction,
    binding: FeasibilityBinding,
    *,
    q_reduction: str = "product",
) -> FeasibilityWeightedAcquisition:
    """Weight an acquisition by one deterministic posterior-predictive PoF."""
    representation = binding.representation
    if not isinstance(representation, ProbabilityOfFeasibility):
        raise TypeError(
            "Deterministic PoF weighting requires a ProbabilityOfFeasibility representation."
        )
    return FeasibilityWeightedAcquisition(
        objective_acquisition,
        representation.probability,
        q_reduction=q_reduction,
    )


def make_classification_feasibility_bridge(
    model: HeterogeneousModel,
    binding: FeasibilityBinding,
) -> ClassificationProbabilityOfFeasibility:
    """Expose one classification feasibility binding as posterior-predictive PoF."""
    representation = binding.representation
    if not isinstance(representation, ProbabilityOfFeasibility):
        raise TypeError(
            "Classification feasibility bridge requires a ProbabilityOfFeasibility representation."
        )

    probability = representation.probability
    entry_model = model[binding.entry_index]
    if probability.model is not entry_model:
        raise ValueError("FeasibilityBinding does not match the current heterogeneous model.")

    output_index = model.resolve_output(binding.output_index)
    entry_index, local_output_index = model.output_owner(output_index)
    if entry_index != binding.entry_index or local_output_index != binding.local_output_index:
        raise ValueError("FeasibilityBinding does not match the current heterogeneous model.")

    return probability


def make_continuous_constraint_bridge(
    model: HeterogeneousModel,
    binding: FeasibilityBinding,
) -> Callable[[Tensor], Tensor]:
    """Expose one continuous feasibility binding as a BoTorch sample residual."""
    representation = binding.representation
    if not isinstance(representation, SampleResidualFeasibility):
        raise TypeError(
            "Continuous constraint bridge requires a SampleResidualFeasibility representation."
        )

    output_index = model.resolve_output(binding.output_index)
    entry_index, local_output_index = model.output_owner(output_index)
    if entry_index != binding.entry_index or local_output_index != binding.local_output_index:
        raise ValueError("FeasibilityBinding does not match the current heterogeneous model.")

    return representation.constraint


def make_sample_classification_feasibility_bridge(
    model: HeterogeneousModel,
    binding: FeasibilityBinding,
) -> SampleProbabilityOfFeasibility:
    """Build sample-wise classifier feasibility without mean-PoF fallback."""
    representation = binding.representation
    if not isinstance(representation, ProbabilityOfFeasibility):
        raise TypeError(
            "Sample classification feasibility requires a ProbabilityOfFeasibility binding."
        )

    probability = make_classification_feasibility_bridge(model, binding)
    classifier = probability.model
    registry_entry = next(
        (
            entry
            for entry in CLASSIFICATION_MODEL_REGISTRY.values()
            if isinstance(classifier, entry.model_class)
        ),
        None,
    )
    if registry_entry is None or not registry_entry.capabilities.supports_probability_samples:
        raise TypeError("Classifier must support epistemic class-probability samples.")

    sample_class_probabilities = classifier.sample_class_probabilities
    feasible_class = probability.feasible_class

    def sample_probability(
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        probabilities = sample_class_probabilities(
            X,
            sample_shape=sample_shape,
            **kwargs,
        )
        return probabilities[..., feasible_class]

    return SampleProbabilityOfFeasibility(sample_probability)


def make_qei_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    best_f: float | Tensor,
    sampler: MCSampler | None = None,
) -> qExpectedImprovement:
    """Build native BoTorch qEI for one unconstrained regression objective."""
    plan = resolve_acquisition_composition(model, semantics)
    if len(plan.objectives) != 1:
        raise ValueError("qEI integration requires exactly one objective.")
    if plan.feasibility:
        raise ValueError(
            "Phase 11 qEI integration is unconstrained; use constraint composition instead."
        )

    binding = plan.objectives[0]
    objective = binding.objective
    if not isinstance(objective, RegressionObjective):
        raise TypeError("qEI integration currently supports RegressionObjective only.")

    bridge = make_botorch_objective_bridge(model, binding)
    directed_best_f = objective.direction.apply(torch.as_tensor(best_f))
    return qExpectedImprovement(
        model=bridge.model,
        best_f=directed_best_f,
        sampler=sampler,
        objective=bridge.objective,
    )


def make_continuous_constrained_qei_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    best_f: float | Tensor,
    sampler: MCSampler | None = None,
    infeasible_cost: float | Tensor = 0.0,
) -> qExpectedImprovement:
    """Build native qEI with continuous constraints from one shared posterior."""
    plan = resolve_acquisition_composition(model, semantics)
    if len(plan.objectives) != 1:
        raise ValueError("Constrained qEI requires exactly one objective.")
    if not plan.feasibility:
        raise ValueError("Constrained qEI requires at least one continuous constraint.")

    objective_binding = plan.objectives[0]
    objective = objective_binding.objective
    if not isinstance(objective, RegressionObjective):
        raise TypeError("Constrained qEI currently supports RegressionObjective only.")
    if any(
        not isinstance(binding.representation, SampleResidualFeasibility)
        for binding in plan.feasibility
    ):
        raise TypeError("Phase 12 constrained qEI supports continuous constraints only.")
    if any(binding.entry_index != objective_binding.entry_index for binding in plan.feasibility):
        raise ValueError(
            "Continuous constrained qEI requires objective and constraints to share "
            "one BoTorch posterior."
        )

    bridge = make_botorch_objective_bridge(model, objective_binding)
    constraints = [
        make_continuous_constraint_bridge(model, binding) for binding in plan.feasibility
    ]
    constrained_objective = ConstrainedMCObjective(
        objective=bridge.objective,
        constraints=constraints,
        infeasible_cost=infeasible_cost,
    )
    directed_best_f = objective.direction.apply(torch.as_tensor(best_f))
    return qExpectedImprovement(
        model=bridge.model,
        best_f=directed_best_f,
        sampler=sampler,
        objective=constrained_objective,
    )


def make_mixed_constrained_qei_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    best_f: float | Tensor,
    sampler: MCSampler | None = None,
    infeasible_cost: float | Tensor = 0.0,
    q_reduction: str = "product",
) -> AcquisitionFunction:
    """Compose qEI with continuous residuals and deterministic classifier PoF."""
    plan = resolve_acquisition_composition(model, semantics)
    if len(plan.objectives) != 1:
        raise ValueError("Mixed constrained qEI requires exactly one objective.")

    continuous = tuple(
        binding
        for binding in plan.feasibility
        if isinstance(binding.representation, SampleResidualFeasibility)
    )
    classification = tuple(
        binding
        for binding in plan.feasibility
        if isinstance(binding.representation, ProbabilityOfFeasibility)
    )
    if not continuous or not classification:
        raise ValueError(
            "Mixed constrained qEI requires both continuous and classification constraints."
        )
    if len(continuous) + len(classification) != len(plan.feasibility):
        raise TypeError("Mixed constrained qEI received an unsupported feasibility representation.")

    objective_binding = plan.objectives[0]
    if any(binding.entry_index != objective_binding.entry_index for binding in continuous):
        raise ValueError("Continuous constraints must share the objective's BoTorch posterior.")

    objective = objective_binding.objective
    if not isinstance(objective, RegressionObjective):
        raise TypeError("Mixed constrained qEI currently supports RegressionObjective only.")
    bridge = make_botorch_objective_bridge(model, objective_binding)
    constrained_objective = ConstrainedMCObjective(
        objective=bridge.objective,
        constraints=[make_continuous_constraint_bridge(model, binding) for binding in continuous],
        infeasible_cost=infeasible_cost,
    )
    acquisition: AcquisitionFunction = qExpectedImprovement(
        model=bridge.model,
        best_f=objective.direction.apply(torch.as_tensor(best_f)),
        sampler=sampler,
        objective=constrained_objective,
    )
    for binding in classification:
        acquisition = make_deterministic_pof_acquisition(
            acquisition,
            binding,
            q_reduction=q_reduction,
        )
    return acquisition

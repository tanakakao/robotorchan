"""Resolved acquisition-composition plans for heterogeneous problem semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import (
    GenericMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.acquisition.objective import ConstrainedMCObjective, MCAcquisitionObjective
from botorch.models.model import Model as BoTorchModel
from botorch.sampling.base import MCSampler
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from torch import Tensor

from robotorchan.acquisition.classification_constraints import (
    FeasibilityWeightedAcquisition,
    IndependentFeasibilityAggregator,
)
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

    classifier_outputs = [binding.output_index for binding in classification]
    if len(set(classifier_outputs)) != len(classifier_outputs):
        raise ValueError("Repeated classification output constraints are not independent.")

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
    if classification:
        acquisition = FeasibilityWeightedAcquisition(
            acquisition,
            IndependentFeasibilityAggregator(
                [binding.representation.probability for binding in classification]
            ),
            q_reduction=q_reduction,
        )
    return acquisition


def make_qnei_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    X_baseline: Tensor,
    sampler: MCSampler | None = None,
    prune_baseline: bool = False,
    cache_root: bool = True,
) -> qNoisyExpectedImprovement:
    """Build native BoTorch qNEI for one unconstrained regression objective."""
    plan = resolve_acquisition_composition(model, semantics)
    if len(plan.objectives) != 1:
        raise ValueError("qNEI integration requires exactly one objective.")
    if plan.feasibility:
        raise ValueError("Phase 14 qNEI integration supports unconstrained objectives only.")

    binding = plan.objectives[0]
    if not isinstance(binding.objective, RegressionObjective):
        raise TypeError("qNEI integration currently supports RegressionObjective only.")
    if X_baseline.ndim != 2 or X_baseline.shape[0] == 0:
        raise ValueError("X_baseline must be a nonempty n x d tensor.")

    bridge = make_botorch_objective_bridge(model, binding)
    return qNoisyExpectedImprovement(
        model=bridge.model,
        X_baseline=X_baseline,
        sampler=sampler,
        objective=bridge.objective,
        prune_baseline=prune_baseline,
        cache_root=cache_root,
    )


def make_continuous_constrained_qnei_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    X_baseline: Tensor,
    sampler: MCSampler | None = None,
    infeasible_cost: float | Tensor = 0.0,
    prune_baseline: bool = False,
    cache_root: bool = True,
) -> qNoisyExpectedImprovement:
    """Build native qNEI with same-posterior continuous sample constraints."""
    plan = resolve_acquisition_composition(model, semantics)
    if len(plan.objectives) != 1:
        raise ValueError("Constrained qNEI requires exactly one objective.")
    if not plan.feasibility:
        raise ValueError("Constrained qNEI requires continuous constraints.")
    if X_baseline.ndim != 2 or X_baseline.shape[0] == 0:
        raise ValueError("X_baseline must be a nonempty n x d tensor.")

    binding = plan.objectives[0]
    if not isinstance(binding.objective, RegressionObjective):
        raise TypeError("Constrained qNEI currently supports RegressionObjective only.")
    if any(
        not isinstance(feasibility.representation, SampleResidualFeasibility)
        for feasibility in plan.feasibility
    ):
        raise TypeError("Phase 15 constrained qNEI supports continuous constraints only.")
    if any(feasibility.entry_index != binding.entry_index for feasibility in plan.feasibility):
        raise ValueError("Constrained qNEI requires one shared BoTorch posterior.")

    bridge = make_botorch_objective_bridge(model, binding)
    objective = ConstrainedMCObjective(
        objective=bridge.objective,
        constraints=[
            make_continuous_constraint_bridge(model, feasibility)
            for feasibility in plan.feasibility
        ],
        infeasible_cost=infeasible_cost,
    )
    return qNoisyExpectedImprovement(
        model=bridge.model,
        X_baseline=X_baseline,
        sampler=sampler,
        objective=objective,
        prune_baseline=prune_baseline,
        cache_root=cache_root,
    )


def make_multiple_learned_constrained_qnei_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    X_baseline: Tensor,
    sampler: MCSampler | None = None,
    infeasible_cost: float | Tensor = 0.0,
    prune_baseline: bool = False,
    cache_root: bool = True,
    q_reduction: str = "product",
) -> AcquisitionFunction:
    """Compose native continuous-constrained qNEI with classifier PoF factors.

    Classification probabilities are independent marginal factors, not a
    sampled joint posterior or baseline-feasibility correction.
    """
    plan = resolve_acquisition_composition(model, semantics)
    if len(plan.objectives) != 1:
        raise ValueError("Multiple constrained qNEI requires exactly one objective.")
    objective_binding = plan.objectives[0]
    if not isinstance(objective_binding.objective, RegressionObjective):
        raise TypeError("Multiple constrained qNEI requires RegressionObjective.")
    if not plan.feasibility:
        raise ValueError("Multiple constrained qNEI requires at least one constraint.")
    if X_baseline.ndim != 2 or X_baseline.shape[0] == 0:
        raise ValueError("X_baseline must be a nonempty n x d tensor.")

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
    if len(continuous) + len(classification) != len(plan.feasibility):
        raise TypeError("Unsupported feasibility representation for multiple constrained qNEI.")
    classifier_outputs = [binding.output_index for binding in classification]
    if len(set(classifier_outputs)) != len(classifier_outputs):
        raise ValueError("Repeated classification output constraints are not independent.")
    classification_outputs = [binding.output_index for binding in classification]
    if len(set(classification_outputs)) != len(classification_outputs):
        raise ValueError("Repeated classification constraint outputs are not supported.")
    if any(binding.entry_index != objective_binding.entry_index for binding in continuous):
        raise ValueError("Continuous constraints must share the objective's BoTorch posterior.")

    bridge = make_botorch_objective_bridge(model, objective_binding)
    objective: MCAcquisitionObjective = bridge.objective
    if continuous:
        objective = ConstrainedMCObjective(
            objective=bridge.objective,
            constraints=[
                make_continuous_constraint_bridge(model, binding) for binding in continuous
            ],
            infeasible_cost=infeasible_cost,
        )
    acquisition: AcquisitionFunction = qNoisyExpectedImprovement(
        model=bridge.model,
        X_baseline=X_baseline,
        sampler=sampler,
        objective=objective,
        prune_baseline=prune_baseline,
        cache_root=cache_root,
    )
    if classification:
        acquisition = FeasibilityWeightedAcquisition(
            acquisition,
            IndependentFeasibilityAggregator(
                [binding.representation.probability for binding in classification]
            ),
            q_reduction=q_reduction,
        )
    return acquisition


@dataclass(frozen=True, slots=True)
class BoTorchMultiObjectiveBridge:
    """One shared BoTorch posterior with directed multi-output MC objectives."""

    model: BoTorchModel
    objective: MCMultiOutputObjective
    output_indices: tuple[int, ...]


def make_botorch_multiobjective_bridge(
    model: HeterogeneousModel,
    bindings: tuple[ObjectiveBinding, ...],
) -> BoTorchMultiObjectiveBridge:
    """Adapt multiple regression objectives owned by one BoTorch model entry.

    Cross-entry objectives are deliberately unsupported: heterogeneous model
    entries do not expose a joint posterior or coupled MC samples.
    """
    if len(bindings) < 2:
        raise ValueError("Multi-objective bridge requires at least two objectives.")
    if len({binding.output_index for binding in bindings}) != len(bindings):
        raise ValueError("Multi-objective outputs must be distinct.")
    if len({binding.entry_index for binding in bindings}) != 1:
        raise ValueError("Multi-objective bridge requires one shared posterior entry.")
    for binding in bindings:
        if not isinstance(binding.objective, RegressionObjective):
            raise TypeError("Multi-objective bridge supports RegressionObjective only.")
        if binding.objective.resolve_output(model) != binding.output_index:
            raise ValueError("Objective binding does not match its resolved output.")
        if model.output_owner(binding.output_index) != (
            binding.entry_index,
            binding.local_output_index,
        ):
            raise ValueError("Objective binding does not match its model owner.")
    entry_model = model[bindings[0].entry_index]
    if not isinstance(entry_model, BoTorchModel):
        raise TypeError("Multi-objective bridge requires a BoTorch model.")

    def objective(samples: Tensor, X: Tensor | None = None) -> Tensor:
        del X
        return torch.stack(
            [
                binding.objective.direction.apply(samples[..., binding.local_output_index])
                for binding in bindings
            ],
            dim=-1,
        )

    return BoTorchMultiObjectiveBridge(
        model=entry_model,
        objective=GenericMCMultiOutputObjective(objective),
        output_indices=tuple(binding.output_index for binding in bindings),
    )


def make_qehvi_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    ref_point: Tensor | list[float],
    partitioning: BoxDecomposition,
    sampler: MCSampler | None = None,
) -> qExpectedHypervolumeImprovement:
    """Build native unconstrained qEHVI from one shared regression posterior.

    The reference point and partitioning must already use the directed
    objective space. No cross-entry posterior or partitioning is inferred.
    """
    plan = resolve_acquisition_composition(model, semantics)
    if plan.feasibility:
        raise ValueError("Phase 20 qEHVI supports unconstrained objectives only.")
    bridge = make_botorch_multiobjective_bridge(model, plan.objectives)
    reference = torch.as_tensor(ref_point)
    if reference.ndim != 1 or reference.numel() != len(bridge.output_indices):
        raise ValueError("ref_point must have one value per objective.")
    if not torch.isfinite(reference).all():
        raise ValueError("ref_point must contain finite values.")
    if not isinstance(partitioning, BoxDecomposition):
        raise TypeError("partitioning must be a BoTorch BoxDecomposition.")
    if partitioning.num_outcomes != reference.numel():
        raise ValueError("partitioning and ref_point objective dimensions must match.")
    partition_ref = partitioning.ref_point.to(device=reference.device, dtype=reference.dtype)
    if not torch.allclose(reference, partition_ref, rtol=0, atol=0):
        raise ValueError("ref_point must match partitioning.ref_point.")
    return qExpectedHypervolumeImprovement(
        model=bridge.model,
        ref_point=reference.tolist(),
        partitioning=partitioning,
        sampler=sampler,
        objective=bridge.objective,
    )


class _NoPendingFeasibilityWeightedAcquisition(FeasibilityWeightedAcquisition):
    """Reject pending points until classification feasibility can include them."""

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        """Prevent inconsistent pending-point objective and PoF batches."""
        if X_pending is not None and X_pending.numel() > 0:
            raise NotImplementedError(
                "Classification-weighted qEHVI does not support X_pending."
            )
        super().set_X_pending(X_pending)


def make_constrained_qehvi_acquisition(
    model: HeterogeneousModel,
    semantics: ProblemSemantics,
    *,
    ref_point: Tensor | list[float],
    partitioning: BoxDecomposition,
    sampler: MCSampler | None = None,
    eta: float = 1e-3,
    q_reduction: str = "product",
) -> AcquisitionFunction:
    """Compose native constrained qEHVI with heterogeneous feasibility.

    Same-entry continuous residuals use native qEHVI sample-wise constraints.
    Independent classification marginal PoFs weight the resulting acquisition.
    """
    plan = resolve_acquisition_composition(model, semantics)
    if not plan.feasibility:
        raise ValueError("Constrained qEHVI requires at least one constraint.")
    bridge = make_botorch_multiobjective_bridge(model, plan.objectives)
    continuous = tuple(
        binding for binding in plan.feasibility
        if isinstance(binding.representation, SampleResidualFeasibility)
    )
    classification = tuple(
        binding for binding in plan.feasibility
        if isinstance(binding.representation, ProbabilityOfFeasibility)
    )
    if len(continuous) + len(classification) != len(plan.feasibility):
        raise TypeError("Unsupported feasibility representation for constrained qEHVI.")
    if any(binding.entry_index != plan.objectives[0].entry_index for binding in continuous):
        raise ValueError("Continuous constraints require the shared objective posterior.")
    outputs = [binding.output_index for binding in classification]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Repeated classification output constraints are not independent.")
    reference = torch.as_tensor(ref_point)
    if reference.ndim != 1 or reference.numel() != len(bridge.output_indices):
        raise ValueError("ref_point must have one value per objective.")
    if not torch.isfinite(reference).all():
        raise ValueError("ref_point must contain finite values.")
    if not isinstance(partitioning, BoxDecomposition):
        raise TypeError("partitioning must be a BoTorch BoxDecomposition.")
    if partitioning.num_outcomes != reference.numel():
        raise ValueError("partitioning and ref_point objective dimensions must match.")
    partition_ref = partitioning.ref_point.to(device=reference.device, dtype=reference.dtype)
    if not torch.allclose(reference, partition_ref, rtol=0, atol=0):
        raise ValueError("ref_point must match partitioning.ref_point.")
    if eta <= 0:
        raise ValueError("eta must be positive.")
    acquisition: AcquisitionFunction = qExpectedHypervolumeImprovement(
        model=bridge.model,
        ref_point=reference.tolist(),
        partitioning=partitioning,
        sampler=sampler,
        objective=bridge.objective,
        constraints=[make_continuous_constraint_bridge(model, b) for b in continuous]
        if continuous else None,
        eta=eta,
    )
    if classification:
        acquisition = _NoPendingFeasibilityWeightedAcquisition(
            acquisition,
            IndependentFeasibilityAggregator(
                [binding.representation.probability for binding in classification]
            ),
            q_reduction=q_reduction,
        )
    return acquisition

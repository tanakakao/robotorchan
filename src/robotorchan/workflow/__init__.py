"""Capability-aware problem selection and recommendation workflows."""

from robotorchan.workflow.components import (
    ComponentSelection,
    ComponentSelections,
    SelectionStatus,
    evaluate_component_inventory,
)
from robotorchan.workflow.cost_budget import CostBudget
from robotorchan.workflow.problem import (
    ObjectiveType,
    ObservationKind,
    OutputType,
    ProblemPurpose,
    ProblemSpec,
)
from robotorchan.workflow.recommendation import (
    Recommendation,
    recommend_compatible_workflows,
)
from robotorchan.workflow.selector import (
    ModelSelectionResult,
    evaluate_models,
    select_compatible_models,
)

__all__ = [
    "ComponentSelection",
    "ComponentSelections",
    "CostBudget",
    "ModelSelectionResult",
    "ObjectiveType",
    "ObservationKind",
    "OutputType",
    "ProblemPurpose",
    "ProblemSpec",
    "Recommendation",
    "SelectionStatus",
    "evaluate_components",
    "evaluate_models",
    "recommend_compatible_workflows",
    "select_compatible_models",
]


def evaluate_components(spec: ProblemSpec) -> ComponentSelections:
    """Evaluate all factory-visible components against the declared problem."""
    from robotorchan.acquisition import list_acquisitions
    from robotorchan.models import list_models
    from robotorchan.optim import list_optimizers, list_search_strategies

    return evaluate_component_inventory(
        spec,
        models=list_models(),
        acquisitions=list_acquisitions(),
        optimizers=list_optimizers(),
        search_strategies=list_search_strategies(),
    )

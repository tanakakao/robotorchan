"""Machine-readable capability metadata for public surrogate models."""

from dataclasses import dataclass
from enum import StrEnum


class InputType(StrEnum):
    CONTINUOUS = "continuous"
    MIXED = "mixed"


class TaskType(StrEnum):
    SINGLE = "single"
    MULTITASK = "multitask"


class InferenceType(StrEnum):
    EXACT = "exact"
    VARIATIONAL = "variational"
    FULLY_BAYESIAN = "fully_bayesian"
    NOT_APPLICABLE = "not_applicable"


class HighDimensionalStrategy(StrEnum):
    NONE = "none"
    REDUCTION = "reduction"
    NEURAL_REDUCTION = "neural_reduction"
    SAAS = "saas"
    MAP_SAAS = "map_saas"
    ADDITIVE = "additive"
    RANDOM_EMBEDDING = "random_embedding"
    DEEP = "deep"


class RobustnessType(StrEnum):
    CONTAMINATION = "contamination"
    HETEROSKEDASTIC = "heteroskedastic"
    NONSTATIONARY = "nonstationary"
    RELEVANCE_PURSUIT = "relevance_pursuit"
    STUDENT_T = "student_t"
    UNCERTAIN_INPUT = "uncertain_input"


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    input_type: InputType = InputType.CONTINUOUS
    task_type: TaskType = TaskType.SINGLE
    inference: InferenceType = InferenceType.EXACT
    high_dimensional: HighDimensionalStrategy = HighDimensionalStrategy.NONE
    robustness: frozenset[RobustnessType] = frozenset()
    multi_fidelity: bool = False
    structured_output: bool = False
    preference: bool = False
    non_gp: bool = False
    ensemble_posterior: bool = False
    supports_multi_output: bool = False
    supports_posterior_samples: bool = False
    supports_fantasize: bool = True


@dataclass(frozen=True, slots=True)
class DocumentationLinks:
    guide: str
    theory: str
    notebook: str


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
    model_name: str
    capabilities: ModelCapabilities
    documentation: DocumentationLinks
    implementation_strategy: str
    limitations: tuple[str, ...] = ()

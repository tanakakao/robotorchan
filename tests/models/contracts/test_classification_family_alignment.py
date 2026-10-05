"""Cross-layer family-alignment contracts for classification models."""

from robotorchan.models.capabilities import HighDimensionalStrategy
from robotorchan.models.classification import CLASSIFICATION_MODEL_REGISTRY

_BINARY_MODULE_PREFIX = "robotorchan.models.classification.binary."


def _filesystem_family(module_name: str) -> str:
    """Return the model family encoded by a binary classifier module path."""
    assert module_name.startswith(_BINARY_MODULE_PREFIX)
    relative_module = module_name.removeprefix(_BINARY_MODULE_PREFIX)
    return relative_module.split(".", maxsplit=1)[0]


def test_binary_registry_family_matches_implementation_family() -> None:
    """Every binary registry family must match its owning filesystem family."""
    for model_id, entry in CLASSIFICATION_MODEL_REGISTRY.items():
        if not model_id.startswith("binary."):
            continue

        assert entry.family == _filesystem_family(entry.model_class.__module__)


def test_deep_gp_family_and_high_dimensional_capability_are_orthogonal() -> None:
    """DeepGP is expressive while still advertising its deep-HD capability."""
    entry = CLASSIFICATION_MODEL_REGISTRY["binary.deep_gp"]

    assert entry.family == "expressive"
    assert entry.capabilities.high_dimensional is HighDimensionalStrategy.DEEP


def test_joint_encoder_keeps_reduced_hierarchy_and_hd_family() -> None:
    """JointEncoder is reduced-space internally but owned by the HD family."""
    entry = CLASSIFICATION_MODEL_REGISTRY["binary.joint_encoder"]

    assert ".high_dimensional.reduced." in entry.model_class.__module__
    assert entry.family == "high_dimensional"
    assert entry.capabilities.high_dimensional is HighDimensionalStrategy.NEURAL_REDUCTION


def test_standard_structural_variants_use_capabilities_not_fake_families() -> None:
    """Mixed and multitask are capability axes inside the standard family."""
    registry = CLASSIFICATION_MODEL_REGISTRY

    assert registry["binary.mixed"].family == "standard"
    assert registry["binary.multitask"].family == "standard"
    assert registry["binary.kronecker_multitask"].family == "standard"

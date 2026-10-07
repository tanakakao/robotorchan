"""Class-selection helpers shared by classification semantics."""

from __future__ import annotations

from robotorchan.models.classification.base import ClassificationMetadata


def resolve_class(
    selector: object,
    metadata: ClassificationMetadata,
    *,
    argument_name: str,
) -> int:
    """Resolve a probability class index or declared class label."""
    if isinstance(selector, bool):
        raise ValueError(f"{argument_name} must not be a boolean.")
    if isinstance(selector, int):
        if not 0 <= selector < metadata.num_classes:
            raise ValueError(f"{argument_name} is outside the classifier class range.")
        return selector
    try:
        return metadata.class_labels.index(selector)
    except ValueError as error:
        raise ValueError(
            f"{argument_name} is not present in classifier class_labels."
        ) from error

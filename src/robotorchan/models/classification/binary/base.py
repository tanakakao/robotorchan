"""Binary-classification contracts."""

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)


class BinaryClassificationMixin(ClassificationModelMixin):
    """Binary specialization without defining a likelihood or link function."""

    num_classes: ClassVar[int] = 2
    class_labels: ClassVar[tuple[int, int]] = (0, 1)
    likelihood_family = ClassificationLikelihoodFamily.BERNOULLI
    latent_output_structure = LatentOutputStructure.SINGLE

    @abstractmethod
    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary predictions using a model-specific probability path."""

"""ALEBO Mahalanobis-kernel multiclass and ordinal GP classifiers.

Inputs are ALEBO embedded coordinates, not original ambient coordinates.
Use ALEBOStrategy for the ambient-to-embedded search-space mapping.
"""

from __future__ import annotations

from gpytorch.kernels import ScaleKernel
from torch import Tensor

from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.high_dimensional.alebo import MahalanobisRBFKernel


def _alebo_kernel(train_X: Tensor, projection: Tensor | None) -> ScaleKernel:
    if train_X.ndim < 2:
        raise ValueError("train_X must have at least two dimensions.")
    if projection is not None and projection.shape[0] != train_X.shape[-1]:
        raise ValueError("projection rows must equal the embedded input dimension.")
    return ScaleKernel(
        MahalanobisRBFKernel(
            ard_num_dims=train_X.shape[-1],
            projection=projection,
        )
    ).to(train_X)


class ALEBOMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Class-wise variational GP with an ALEBO Mahalanobis covariance."""

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
        projection: Tensor | None = None,
    ) -> None:
        super().__init__(
            train_X, train_Y, num_classes=num_classes,
            covar_module=_alebo_kernel(train_X, projection),
        )

    @property
    def mahalanobis_kernel(self) -> MahalanobisRBFKernel:
        kernel = self.class_models[0].model.covar_module
        if not isinstance(kernel, ScaleKernel) or not isinstance(
            kernel.base_kernel, MahalanobisRBFKernel
        ):
            raise TypeError("Expected ALEBO Mahalanobis kernel.")
        return kernel.base_kernel

    @property
    def metric(self) -> Tensor:
        """Metric of the first class-specific GP (each class learns its own)."""
        return self.mahalanobis_kernel.metric


class ALEBOOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordered-probit GP with ALEBO Mahalanobis geometry."""

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
        projection: Tensor | None = None,
    ) -> None:
        super().__init__(
            train_X, train_Y, num_classes=num_classes,
            covar_module=_alebo_kernel(train_X, projection),
        )

    @property
    def mahalanobis_kernel(self) -> MahalanobisRBFKernel:
        kernel = self.model.covar_module
        if not isinstance(kernel, ScaleKernel) or not isinstance(
            kernel.base_kernel, MahalanobisRBFKernel
        ):
            raise TypeError("Expected ALEBO Mahalanobis kernel.")
        return kernel.base_kernel

    @property
    def metric(self) -> Tensor:
        return self.mahalanobis_kernel.metric

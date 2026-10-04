"""ALEBO binary GP classification in the embedded search space."""

from __future__ import annotations

from gpytorch.kernels import ScaleKernel
from torch import Tensor

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.high_dimensional.alebo import MahalanobisRBFKernel


class ALEBOBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier using ALEBO Mahalanobis geometry.

    The public model input is the ALEBO embedded coordinate ``Z``. Use
    :class:`robotorchan.optim.embedding.ALEBOStrategy` to construct the
    embedding, project embedded candidates back to the ambient search space,
    and supply its ``embedding`` matrix here as ``projection``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        projection: Tensor | None = None,
    ) -> None:
        """Construct the classifier in ALEBO embedded coordinates."""
        if train_X.ndim < 2:
            raise ValueError("train_X must have at least two dimensions.")
        if projection is not None and projection.shape[0] != train_X.shape[-1]:
            raise ValueError("projection rows must equal the embedded input dimension.")
        covar_module = ScaleKernel(
            MahalanobisRBFKernel(
                ard_num_dims=train_X.shape[-1],
                projection=projection,
            )
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
        )

    @property
    def mahalanobis_kernel(self) -> MahalanobisRBFKernel:
        """Return the ALEBO Mahalanobis base kernel."""
        covar_module = self.model.covar_module
        if not isinstance(covar_module, ScaleKernel) or not isinstance(
            covar_module.base_kernel, MahalanobisRBFKernel
        ):
            raise TypeError(
                "ALEBOBinarySingleTaskGPClassifier requires the ALEBO Mahalanobis kernel."
            )
        return covar_module.base_kernel

    @property
    def metric(self) -> Tensor:
        """Return the current learned Mahalanobis metric."""
        return self.mahalanobis_kernel.metric

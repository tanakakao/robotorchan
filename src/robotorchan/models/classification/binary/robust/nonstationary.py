"""Binary GP classification with input-dependent latent smoothness."""

from __future__ import annotations

from gpytorch.kernels import ScaleKernel
from torch import Tensor

from robotorchan.models.classification.binary.robust.base import (
    ClassificationRobustnessType,
    RobustBinaryClassificationMixin,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.robust.nonstationary import GibbsKernel


class NonstationaryBinarySingleTaskGPClassifier(
    RobustBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Binary variational GP classifier with input-dependent latent lengthscales."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        lengthscale_floor: float = 1e-3,
        **kwargs: object,
    ) -> None:
        input_dim = train_X.shape[-1]
        gibbs_kernel = GibbsKernel(input_dim, lengthscale_floor=lengthscale_floor)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=ScaleKernel(gibbs_kernel),
            **kwargs,
        )

    @property
    def gibbs_kernel(self) -> GibbsKernel:
        """Return the Gibbs kernel controlling latent decision-boundary smoothness."""
        kernel = self.covar_module.base_kernel
        if not isinstance(kernel, GibbsKernel):
            raise RuntimeError("Expected GibbsKernel as the base covariance module.")
        return kernel

    def local_lengthscale(self, X: Tensor) -> Tensor:
        """Return learned local lengthscales at each input location."""
        return self.gibbs_kernel.local_lengthscale(X)

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return nonstationary latent-process robustness metadata."""
        return frozenset({ClassificationRobustnessType.NONSTATIONARY})

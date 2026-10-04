"""SAAS priors for high-dimensional binary GP classification."""

from __future__ import annotations

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import HalfCauchyPrior
from torch import Tensor

from robotorchan.models.classification.binary.standard.single_task import BinarySingleTaskGPClassifier


class SaasBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier with sparse inverse-lengthscale shrinkage."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        global_scale: float = 0.1,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a SAAS-style prior over inverse ARD lengthscales."""
        if global_scale <= 0.0:
            raise ValueError("global_scale must be positive.")
        covar_module = self._make_saas_covar_module(train_X, global_scale)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            input_transform=input_transform,
        )
        self.global_scale = float(global_scale)

    @staticmethod
    def _make_saas_covar_module(train_X: Tensor, global_scale: float) -> ScaleKernel:
        """Construct an ARD Matérn kernel with shrinkage on inverse lengthscales."""
        batch_shape = train_X.shape[:-2]
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=train_X.shape[-1],
            batch_shape=batch_shape,
        )
        prior = HalfCauchyPrior(
            torch.as_tensor(global_scale, dtype=train_X.dtype, device=train_X.device)
        )
        base_kernel.register_prior(
            "saas_inv_lengthscale_prior",
            prior,
            lambda module: module.lengthscale.reciprocal(),
            _set_inverse_lengthscale,
        )
        return ScaleKernel(base_kernel, batch_shape=batch_shape)


def _set_inverse_lengthscale(module: MaternKernel, value: Tensor) -> None:
    """Set kernel lengthscales from inverse-lengthscale prior samples."""
    module.lengthscale = value.reciprocal()

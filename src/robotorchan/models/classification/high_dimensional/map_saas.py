"""Sparse high-dimensional binary GP classifiers."""

from __future__ import annotations

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.priors import HalfCauchyPrior
from torch import Tensor

from robotorchan.models.classification.standard.binary import BinarySingleTaskGPClassifier


class MapSaasBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier with a MAP-SAAS-style sparse ARD prior."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        tau: float = 0.1,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a sparse high-dimensional binary classifier."""
        if tau <= 0.0:
            raise ValueError("tau must be positive.")
        covar_module = self._make_sparse_covar_module(train_X, tau)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            input_transform=input_transform,
        )
        self.tau = float(tau)

    @staticmethod
    def _make_sparse_covar_module(train_X: Tensor, tau: float) -> Kernel:
        """Construct ARD covariance and register a global shrinkage prior."""
        covar_module = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=train_X.shape[-1],
            batch_shape=train_X.shape[:-2],
        )
        base_kernel = covar_module.base_kernel if isinstance(covar_module, ScaleKernel) else covar_module
        prior = HalfCauchyPrior(torch.as_tensor(tau, dtype=train_X.dtype, device=train_X.device))
        base_kernel.register_prior("saas_inv_lengthscale_prior", prior, "raw_lengthscale")
        return covar_module

"""SAAS priors for high-dimensional binary GP classification."""

from __future__ import annotations

import torch
from botorch.models.transforms.input import InputTransform
from gpytorch.constraints import Positive
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import HalfCauchyPrior
from torch import Tensor

from robotorchan.models.classification.standard.binary import BinarySingleTaskGPClassifier


class SaasBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier with a SAAS global-local shrinkage prior."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        global_scale: float = 0.1,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize SAAS shrinkage over inverse ARD lengthscales."""
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
        """Construct an ARD Matérn kernel parameterized by inverse lengthscale."""
        batch_shape = train_X.shape[:-2]
        input_dim = train_X.shape[-1]
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=input_dim,
            batch_shape=batch_shape,
            lengthscale_constraint=Positive(transform=None, inv_transform=None),
        )
        base_kernel.register_parameter(
            name="raw_inv_lengthscale",
            parameter=torch.nn.Parameter(
                torch.ones(*batch_shape, 1, input_dim, dtype=train_X.dtype, device=train_X.device)
            ),
        )
        prior = HalfCauchyPrior(
            torch.as_tensor(global_scale, dtype=train_X.dtype, device=train_X.device)
        )
        base_kernel.register_prior(
            "saas_inv_lengthscale_prior",
            prior,
            lambda module: module.raw_inv_lengthscale,
            lambda module, value: module.raw_inv_lengthscale.data.copy_(value),
        )
        base_kernel.register_forward_pre_hook(_sync_inverse_lengthscale)
        return ScaleKernel(base_kernel, batch_shape=batch_shape)


def _sync_inverse_lengthscale(module: MaternKernel, _inputs: tuple[Tensor, ...]) -> None:
    """Map sparse inverse lengthscales to the kernel lengthscale before evaluation."""
    eps = torch.finfo(module.raw_inv_lengthscale.dtype).eps
    module.raw_lengthscale.data.copy_(module.raw_inv_lengthscale.clamp_min(eps).reciprocal())

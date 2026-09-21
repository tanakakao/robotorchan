"""Exact GP for training inputs with uncertain categorical observations."""

from __future__ import annotations

import torch
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import Kernel, ScaleKernel
from torch import Tensor, nn

from robotorchan.models.standard.single_task import SingleTaskGP


def _validate_category_probabilities(probabilities: Tensor, *, num_categories: int) -> None:
    if probabilities.shape[-1] != num_categories:
        raise ValueError("Category probabilities have an incompatible category dimension.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Category probabilities must be finite.")
    if torch.any(probabilities < 0):
        raise ValueError("Category probabilities must be nonnegative.")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("Category probabilities must sum to one along the last dimension.")


class ExpectedCategoricalKernel(Kernel):
    """PSD categorical covariance marginalized over category probabilities."""

    has_lengthscale = False

    def __init__(self, num_categories: int, *, jitter: float = 1e-6) -> None:
        super().__init__()
        if num_categories < 2:
            raise ValueError("num_categories must be at least two.")
        if jitter <= 0:
            raise ValueError("jitter must be positive.")
        self.num_categories = int(num_categories)
        self.jitter = float(jitter)
        self.category_factor = nn.Parameter(torch.eye(num_categories))

    def category_covar(self) -> Tensor:
        """Return the learned positive-definite category covariance matrix."""
        factor = self.category_factor
        identity = torch.eye(
            self.num_categories,
            device=factor.device,
            dtype=factor.dtype,
        )
        return factor @ factor.transpose(-1, -2) + self.jitter * identity

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        """Evaluate expected covariance between category distributions."""
        _validate_category_probabilities(x1, num_categories=self.num_categories)
        _validate_category_probabilities(x2, num_categories=self.num_categories)
        category_covar = self.category_covar()
        if diag:
            if x1.shape != x2.shape:
                raise ValueError("diag=True requires x1 and x2 to have the same shape.")
            return ((x1 @ category_covar) * x2).sum(dim=-1)
        return (x1 @ category_covar) @ x2.transpose(-1, -2)


class AugmentedUncertainCategoricalKernel(Kernel):
    """Product covariance over continuous features and category probabilities."""

    has_lengthscale = False

    def __init__(self, continuous_dim: int, num_categories: int) -> None:
        super().__init__()
        if continuous_dim < 0:
            raise ValueError("continuous_dim must be nonnegative.")
        self.continuous_dim = int(continuous_dim)
        self.num_categories = int(num_categories)
        self.categorical_kernel = ExpectedCategoricalKernel(num_categories)
        self.continuous_kernel = (
            None
            if continuous_dim == 0
            else ScaleKernel(get_covar_module_with_dim_scaled_prior(ard_num_dims=continuous_dim))
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        """Evaluate covariance on concatenated continuous/probability inputs."""
        expected_width = self.continuous_dim + self.num_categories
        if x1.shape[-1] != expected_width or x2.shape[-1] != expected_width:
            raise ValueError("Inputs have an incompatible augmented feature dimension.")
        p1 = x1[..., self.continuous_dim :]
        p2 = x2[..., self.continuous_dim :]
        categorical = self.categorical_kernel(p1, p2, diag=diag, **params)
        if self.continuous_kernel is None:
            return categorical
        continuous = self.continuous_kernel(
            x1[..., : self.continuous_dim],
            x2[..., : self.continuous_dim],
            diag=diag,
            **params,
        )
        return continuous * categorical


class UncertainCategoricalSingleTaskGP(SingleTaskGP):
    """Exact GP marginalizing explicit uncertainty in one categorical feature."""

    def __init__(
        self,
        train_X_cont: Tensor,
        train_category_probabilities: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
    ) -> None:
        if train_X_cont.shape[:-1] != train_category_probabilities.shape[:-1]:
            raise ValueError("Continuous inputs and category probabilities must align.")
        if train_X_cont.device != train_category_probabilities.device:
            raise ValueError("Continuous inputs and category probabilities must share a device.")
        if train_X_cont.dtype != train_category_probabilities.dtype:
            raise ValueError("Continuous inputs and category probabilities must share a dtype.")
        num_categories = train_category_probabilities.shape[-1]
        if num_categories < 2:
            raise ValueError("At least two categories are required.")
        _validate_category_probabilities(
            train_category_probabilities,
            num_categories=num_categories,
        )

        self._continuous_dim = train_X_cont.shape[-1]
        self._num_categories = num_categories
        augmented_X = torch.cat([train_X_cont, train_category_probabilities], dim=-1)
        kernel = AugmentedUncertainCategoricalKernel(
            continuous_dim=self._continuous_dim,
            num_categories=num_categories,
        )
        super().__init__(
            augmented_X,
            train_Y,
            train_Yvar=train_Yvar,
            covar_module=kernel,
        )
        self._store_raw_tensor("train_X_cont", train_X_cont.detach().clone())
        self._store_raw_tensor(
            "train_category_probabilities",
            train_category_probabilities.detach().clone(),
        )

    @property
    def raw_train_X_cont(self) -> Tensor:
        """Caller-supplied continuous training features."""
        value = self._get_raw_tensor("train_X_cont")
        if value is None:
            raise RuntimeError("raw_train_X_cont was unexpectedly stored as None.")
        return value

    @property
    def raw_train_category_probabilities(self) -> Tensor:
        """Caller-supplied categorical probability vectors."""
        value = self._get_raw_tensor("train_category_probabilities")
        if value is None:
            raise RuntimeError("raw_train_category_probabilities was unexpectedly stored as None.")
        return value

    @property
    def category_kernel(self) -> ExpectedCategoricalKernel:
        """Expected categorical covariance component."""
        kernel = self.covar_module
        if not isinstance(kernel, AugmentedUncertainCategoricalKernel):
            raise RuntimeError("Expected AugmentedUncertainCategoricalKernel.")
        return kernel.categorical_kernel

    def augment_inputs(self, X_cont: Tensor, category_probabilities: Tensor) -> Tensor:
        """Validate and concatenate continuous features with category probabilities."""
        if X_cont.shape[:-1] != category_probabilities.shape[:-1]:
            raise ValueError("Continuous inputs and category probabilities must align.")
        if X_cont.shape[-1] != self._continuous_dim:
            raise ValueError("Continuous inputs have an incompatible feature dimension.")
        same_device = X_cont.device == category_probabilities.device
        same_dtype = X_cont.dtype == category_probabilities.dtype
        if not same_device or not same_dtype:
            raise ValueError(
                "Continuous inputs and category probabilities must share dtype and device."
            )
        _validate_category_probabilities(
            category_probabilities,
            num_categories=self._num_categories,
        )
        return torch.cat([X_cont, category_probabilities], dim=-1)

    def posterior_with_category_probabilities(
        self,
        X_cont: Tensor,
        category_probabilities: Tensor,
        **kwargs,
    ):
        """Return posterior for explicit categorical probability vectors."""
        return super().posterior(self.augment_inputs(X_cont, category_probabilities), **kwargs)

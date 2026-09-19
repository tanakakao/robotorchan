"""Additive GP wrappers with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.additive_gp import OrthogonalAdditiveGP as BoTorchOrthogonalAdditiveGP
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from torch import Tensor

from robotorchan.models.base import (
    ExactGPModelMixin,
    CategoricalOneHotInputTransform,
)


class OrthogonalAdditiveGP(ExactGPModelMixin, BoTorchOrthogonalAdditiveGP):
    """BoTorch ``OrthogonalAdditiveGP`` with robotorchan raw-data retention."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        covar_module: OrthogonalAdditiveKernel | None = None,
        second_order: bool = False,
        likelihood: Likelihood | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            second_order=second_order,
            likelihood=likelihood,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, None)


class MixedOrthogonalAdditiveGP(OrthogonalAdditiveGP):
    """Orthogonal additive GP for mixed inputs via internal one-hot encoding.

    ``OrthogonalAdditiveKernel`` orthogonalizes one-dimensional base kernels via
    Gauss-Legendre quadrature on the continuous interval ``[0, 1]``. Replacing
    those scalar components with a standard categorical kernel would violate the
    kernel's defining orthogonalization measure. This wrapper therefore uses the
    shared internal one-hot fallback and applies the original OAK to the encoded
    numeric representation.

    Continuous raw inputs must still lie in ``[0, 1]`` as required by the
    upstream OAK. Each one-hot column becomes its own additive component, so
    component-level interpretation is in encoded rather than raw-feature space.
    A future native categorical OAK would require a discrete orthogonalization
    measure rather than a simple categorical-kernel substitution.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        covar_module: OrthogonalAdditiveKernel | None = None,
        second_order: bool = False,
        likelihood: Likelihood | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if input_transform is not None:
            raise ValueError(
                "MixedOrthogonalAdditiveGP manages input_transform internally for "
                "categorical encoding; a custom input_transform is not currently "
                "supported."
            )

        mixed_transform = CategoricalOneHotInputTransform(
            train_X=train_X,
            cat_dims=cat_dims,
        )
        if covar_module is None:
            covar_module = OrthogonalAdditiveKernel(
                dim=mixed_transform.encoded_input_dim,
                second_order=second_order,
                dtype=train_X.dtype,
                device=train_X.device,
            )
        elif covar_module.dim != mixed_transform.encoded_input_dim:
            raise ValueError(
                "covar_module.dim must match the internally encoded input dimension "
                f"({mixed_transform.encoded_input_dim}), got {covar_module.dim}."
            )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            second_order=second_order,
            likelihood=likelihood,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=mixed_transform,
        )
        self.cat_dims = mixed_transform.cat_dims
        self.raw_input_dim = mixed_transform.raw_input_dim
        self.encoded_input_dim = mixed_transform.encoded_input_dim

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Observed category values in normalized ``cat_dims`` order."""
        return self.input_transform.category_values

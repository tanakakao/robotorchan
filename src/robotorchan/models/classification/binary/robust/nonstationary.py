"""Binary GP classification with input-dependent latent smoothness."""

from __future__ import annotations

import torch
from gpytorch.kernels import IndexKernel, ProductKernel, ScaleKernel
from torch import Tensor

from robotorchan.models.classification.binary.robust.base import (
    ClassificationRobustnessType,
    RobustBinaryClassificationMixin,
)
from robotorchan.models.base import (
    continuous_feature_dims,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.binary.standard.multitask import (
    MultiTaskBinaryGPClassifier,
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
        covar_module = ScaleKernel(gibbs_kernel).to(train_X)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            **kwargs,
        )

    @property
    def gibbs_kernel(self) -> GibbsKernel:
        """Return the Gibbs kernel controlling latent decision-boundary smoothness."""
        kernel = self.model.covar_module.base_kernel
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


class MixedNonstationaryBinarySingleTaskGPClassifier(
    RobustBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Mixed binary classifier with nonstationary continuous covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        lengthscale_floor: float = 1e-3,
        **kwargs: object,
    ) -> None:
        input_dim = train_X.shape[-1]
        normalized_cat_dims = normalize_feature_dims(cat_dims, input_dim, name="cat_dims")
        continuous_dims = continuous_feature_dims(input_dim, cat_dims=normalized_cat_dims)
        if not continuous_dims:
            raise ValueError(
                "MixedNonstationaryBinarySingleTaskGPClassifier requires a continuous dimension."
            )

        def gibbs_factory(batch_shape, ard_num_dims, active_dims):
            del batch_shape, ard_num_dims
            return ScaleKernel(
                GibbsKernel(len(active_dims), lengthscale_floor=lengthscale_floor),
                active_dims=active_dims,
            )

        covar_module = make_mixed_covar_module(
            input_dim=input_dim,
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=gibbs_factory,
        ).to(train_X)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            **kwargs,
        )
        self.cat_dims = normalized_cat_dims
        self.continuous_dims = continuous_dims

    def local_lengthscale(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return local lengthscales from both continuous Gibbs branches."""
        continuous_X = X[..., list(self.continuous_dims)]
        covar_module = self.model.covar_module
        additive = covar_module.kernels[0].base_kernel
        interaction = covar_module.kernels[2].kernels[0].base_kernel
        if not isinstance(additive, GibbsKernel) or not isinstance(interaction, GibbsKernel):
            raise RuntimeError("Expected GibbsKernel in both continuous covariance branches.")
        return (
            additive.local_lengthscale(continuous_X),
            interaction.local_lengthscale(continuous_X),
        )

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return nonstationary latent-process robustness metadata."""
        return frozenset({ClassificationRobustnessType.NONSTATIONARY})


class NonstationaryMultiTaskBinaryGPClassifier(
    RobustBinaryClassificationMixin,
    MultiTaskBinaryGPClassifier,
):
    """Long-format multi-task classifier with nonstationary data covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        rank: int | None = None,
        lengthscale_floor: float = 1e-3,
        **kwargs: object,
    ) -> None:
        input_dim = train_X.shape[-1]
        task_dim = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        data_dims = continuous_feature_dims(input_dim, cat_dims=[task_dim])
        if not data_dims:
            raise ValueError(
                "NonstationaryMultiTaskBinaryGPClassifier requires a non-task data dimension."
            )
        task_values = torch.unique(train_X[..., task_dim]).round().to(dtype=torch.long)
        num_tasks = int(task_values.max()) + 1
        resolved_rank = min(num_tasks, 1) if rank is None else rank
        data_kernel = ScaleKernel(
            GibbsKernel(len(data_dims), lengthscale_floor=lengthscale_floor),
            active_dims=data_dims,
        )
        task_kernel = IndexKernel(
            num_tasks=num_tasks,
            rank=resolved_rank,
            active_dims=[task_dim],
        )
        covar_module = ProductKernel(data_kernel, task_kernel).to(train_X)
        BinarySingleTaskGPClassifier.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            **kwargs,
        )
        self.task_feature = task_dim
        self.num_tasks = num_tasks
        self.rank = resolved_rank
        self.continuous_dims = tuple(data_dims)

    @property
    def gibbs_kernel(self) -> GibbsKernel:
        """Return the Gibbs kernel used by the data covariance branch."""
        data_kernel = self.model.covar_module.kernels[0]
        kernel = data_kernel.base_kernel
        if not isinstance(kernel, GibbsKernel):
            raise RuntimeError("Expected GibbsKernel in the data covariance branch.")
        return kernel

    def local_lengthscale(self, X: Tensor) -> Tensor:
        """Return local lengthscales for non-task data features."""
        data_X = X[..., list(self.continuous_dims)]
        return self.gibbs_kernel.local_lengthscale(data_X)

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return nonstationary latent-process robustness metadata."""
        return frozenset({ClassificationRobustnessType.NONSTATIONARY})

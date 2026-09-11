"""Fully Bayesian SAAS wrappers with robotorchan model conventions.

Mixed continuous/categorical SAAS variants are intentionally not exposed yet.
BoTorch's fully Bayesian models evaluate their covariance inside the lightweight
JAX/NumPyro sampling path used by NUTS, then load sampled hyperparameters into
GPyTorch modules afterwards. That architecture means robotorchan's normal mixed
kernel injection helper cannot make NUTS use a categorical covariance.

A correct mixed SAAS implementation therefore requires a dedicated mixed Pyro
model whose sampling-time covariance matches the GPyTorch model loaded after
MCMC. Until that path exists and is tested, these wrappers retain upstream
BoTorch's continuous-input SAAS semantics rather than accepting ``cat_dims``
that would only affect the post-fit model.
"""

from __future__ import annotations

from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP as BoTorchSaasFullyBayesianSingleTaskGP,
)
from botorch.models.fully_bayesian_multitask import MultitaskSaasPyroModel
from botorch.models.fully_bayesian_multitask import (
    SaasFullyBayesianMultiTaskGP as BoTorchSaasFullyBayesianMultiTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor

from robotorchan.models.base import ModelTrainingMixin, SupervisedTrainingDataMixin


class SaasFullyBayesianSingleTaskGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    BoTorchSaasFullyBayesianSingleTaskGP,
):
    """BoTorch SAAS single-task GP with robotorchan raw-data conventions.

    Fully Bayesian fitting remains delegated to BoTorch's
    ``fit_fully_bayesian_model_nuts``. This wrapper only retains the caller's
    untransformed training tensors and makes the lack of MLL-style fitting
    explicit through ``supports_mll = False``.

    Mixed continuous/categorical SAAS is deliberately deferred because the
    NUTS-side JAX/NumPyro covariance must be extended together with the loaded
    GPyTorch covariance. This wrapper therefore preserves the upstream
    continuous-input constructor surface and does not accept ``cat_dims``.
    """

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> None:
        """Initialize the SAAS model while retaining caller raw tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            use_input_warping=use_input_warping,
            indices_to_warp=indices_to_warp,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )


class SaasFullyBayesianMultiTaskGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    BoTorchSaasFullyBayesianMultiTaskGP,
):
    """BoTorch SAAS multi-task GP with robotorchan raw-data conventions.

    The long-format task-feature representation, Pyro model, MCMC loading,
    transforms, and posterior computation are inherited from BoTorch. Fitting
    is performed with ``fit_fully_bayesian_model_nuts`` rather than an MLL.

    Mixed continuous/categorical SAAS is deliberately deferred for the same
    sampling-path reason as the single-task wrapper. A future implementation
    must also exclude the task feature from the mixed design covariance while
    preserving the categorical feature indices used during NUTS.
    """

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        pyro_model: MultitaskSaasPyroModel | None = None,
        validate_task_values: bool = True,
    ) -> None:
        """Initialize the multi-task SAAS model and retain raw training data."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            pyro_model=pyro_model,
            validate_task_values=validate_task_values,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )

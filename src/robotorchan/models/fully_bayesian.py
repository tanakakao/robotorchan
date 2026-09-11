"""Fully Bayesian SAAS wrappers with robotorchan model conventions.

For categorical variables, the supported Phase 8 path is one-hot preprocessing
before constructing the SAAS model. The one-hot columns are then part of the
same numeric feature tensor used both by the JAX/NumPyro NUTS covariance and by
the GPyTorch model loaded after MCMC, so fitting and prediction stay consistent.

Native categorical-kernel SAAS remains a separate future extension. BoTorch's
fully Bayesian models evaluate covariance inside the lightweight JAX/NumPyro
sampling path, so robotorchan's normal GPyTorch mixed-kernel injection helper
cannot provide a true categorical kernel during NUTS.
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

    Categorical variables are supported through one-hot preprocessing before
    construction. Pass the same encoded feature layout for training and
    prediction. A dedicated native categorical-kernel SAAS model is not exposed
    because that would require matching changes in BoTorch's JAX/NumPyro NUTS
    covariance implementation.
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

    Categorical design variables may be one-hot encoded before the task feature
    is appended. The task feature remains a single long-format task column and
    must not itself be one-hot encoded. Native categorical-kernel SAAS remains a
    future extension of the NUTS-side covariance rather than this thin wrapper.
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

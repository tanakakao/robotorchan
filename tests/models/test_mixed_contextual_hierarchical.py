import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedHierarchicalConditionalKernelGP,
    MixedHierarchicalConditionalKernelMultiTaskGP,
    MixedLCEMGP,
)


def _contains_categorical_kernel(module: torch.nn.Module) -> bool:
    return any(isinstance(child, CategoricalKernel) for child in module.modules())


def test_mixed_lcemgp_uses_native_mixed_design_covariance() -> None:
    train_X = torch.tensor(
        [
            [0.10, 0.0, 0.0],
            [0.20, 1.0, 0.0],
            [0.40, 0.0, 1.0],
            [0.55, 1.0, 1.0],
            [0.75, 0.0, 0.0],
            [0.90, 1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.25 * train_X[:, 1:2] + 0.4 * train_X[:, 2:3])

    model = MixedLCEMGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        cat_dims=[1],
        outcome_transform=None,
    )

    assert model.cat_dims == (1,)
    assert model.data_cat_dims == (1,)
    assert model.raw_task_feature == 2
    assert torch.equal(model.raw_train_X, train_X)
    assert _contains_categorical_kernel(model.covar_module)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def _hierarchical_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.10, 0.0, 0.20],
            [1.0, 0.25, 0.0, 0.35],
            [0.0, 0.40, 1.0, 0.55],
            [1.0, 0.60, 1.0, 0.70],
            [0.0, 0.75, 0.0, 0.80],
            [1.0, 0.90, 1.0, 0.95],
        ],
        dtype=torch.double,
    )
    train_Y = (
        0.3 * train_X[:, :1]
        + train_X[:, 1:2]
        + 0.2 * train_X[:, 3:4]
    )
    return train_X, train_Y


def test_mixed_hierarchical_single_task_keeps_conditional_structure() -> None:
    train_X, train_Y = _hierarchical_training_data()
    dependencies = {2: {0: [1], 1: [3]}}

    model = MixedHierarchicalConditionalKernelGP(
        train_X=train_X,
        train_Y=train_Y,
        hierarchical_dependencies=dependencies,
        cat_dims=[0],
        use_saas_prior=False,
        outcome_transform=None,
    )

    assert model.cat_dims == (0,)
    assert torch.equal(model.raw_train_X, train_X)
    assert _contains_categorical_kernel(model.covar_module)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_hierarchical_rejects_structural_parent_as_cat_dim() -> None:
    train_X, train_Y = _hierarchical_training_data()
    dependencies = {2: {0: [1], 1: [3]}}

    try:
        MixedHierarchicalConditionalKernelGP(
            train_X=train_X,
            train_Y=train_Y,
            hierarchical_dependencies=dependencies,
            cat_dims=[2],
            outcome_transform=None,
        )
    except ValueError as error:
        assert "structural" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected structural parent overlap to be rejected.")


def test_mixed_hierarchical_multitask_remaps_cat_dims_around_task_feature() -> None:
    base_X, base_Y = _hierarchical_training_data()
    task = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=torch.double)
    train_X = torch.cat([base_X[:, :2], task, base_X[:, 2:]], dim=-1)
    dependencies = {2: {0: [1], 1: [3]}}

    model = MixedHierarchicalConditionalKernelMultiTaskGP(
        train_X=train_X,
        train_Y=base_Y,
        task_feature=2,
        hierarchical_dependencies=dependencies,
        cat_dims=[0],
        use_saas_prior=False,
        outcome_transform=None,
    )

    assert model.cat_dims == (0,)
    assert model.data_cat_dims == (0,)
    assert model.raw_task_feature == 2
    assert _contains_categorical_kernel(model.covar_module.kernels[0])
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)

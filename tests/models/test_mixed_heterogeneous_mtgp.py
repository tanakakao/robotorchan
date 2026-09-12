import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import AdditiveKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedHeterogeneousMTGP


def _training_data() -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
]:
    train_Xs = [
        torch.tensor(
            [[0.10, 0.0], [0.30, 1.0], [0.60, 0.0], [0.85, 1.0]],
            dtype=torch.double,
        ),
        torch.tensor(
            [
                [0.15, 0.0, 0.20],
                [0.35, 1.0, 0.40],
                [0.65, 0.0, 0.70],
                [0.90, 1.0, 0.95],
            ],
            dtype=torch.double,
        ),
    ]
    train_Ys = [
        train_Xs[0][:, :1] + 0.2 * train_Xs[0][:, 1:2],
        train_Xs[1][:, :1] + 0.2 * train_Xs[1][:, 1:2] + 0.1 * train_Xs[1][:, 2:3],
    ]
    feature_indices = [[0, 1], [0, 1, 2]]
    return train_Xs, train_Ys, feature_indices


def _contains_categorical_kernel(module: torch.nn.Module) -> bool:
    return any(isinstance(child, CategoricalKernel) for child in module.modules())


def test_mixed_heterogeneous_uses_native_subset_mixed_kernel() -> None:
    train_Xs, train_Ys, feature_indices = _training_data()
    model = MixedHeterogeneousMTGP(
        train_Xs=train_Xs,
        train_Ys=train_Ys,
        train_Yvars=None,
        feature_indices=feature_indices,
        full_feature_dim=3,
        cat_dims=[1],
        use_saas_prior=False,
    )

    assert model.cat_dims == (1,)
    assert _contains_categorical_kernel(model.covar_module)
    assert any(isinstance(kernel, AdditiveKernel) for kernel in model.covar_module.kernels)
    assert torch.equal(model.raw_train_Xs[0], train_Xs[0])
    assert torch.equal(model.raw_train_Xs[1], train_Xs[1])
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_heterogeneous_posterior_uses_raw_target_feature_space() -> None:
    train_Xs, train_Ys, feature_indices = _training_data()
    model = MixedHeterogeneousMTGP(
        train_Xs=train_Xs,
        train_Ys=train_Ys,
        train_Yvars=None,
        feature_indices=feature_indices,
        full_feature_dim=3,
        cat_dims=[1],
        use_saas_prior=False,
    )
    model.eval()

    test_X = torch.tensor(
        [[0.25, 1.0, 0.0], [0.75, 0.0, 0.0]],
        dtype=torch.double,
    )
    posterior = model.posterior(test_X)

    assert posterior.mean.shape[-2:] == torch.Size([2, 1])
    assert posterior.variance.shape[-2:] == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

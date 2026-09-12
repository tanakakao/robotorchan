import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedRobustRelevancePursuitSingleTaskGP


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.10, 0.0],
            [0.20, 1.0],
            [0.35, 2.0],
            [0.50, 0.0],
            [0.65, 1.0],
            [0.80, 2.0],
            [0.90, 0.0],
            [0.95, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = (
        torch.sin(train_X[:, :1] * 3.0)
        + 0.15 * train_X[:, 1:2]
    )
    train_Y[-1] = train_Y[-1] + 2.0
    return train_X, train_Y


def _contains_categorical_kernel(module: torch.nn.Module) -> bool:
    return any(isinstance(child, CategoricalKernel) for child in module.modules())


def test_mixed_robust_gp_uses_native_categorical_kernel() -> None:
    train_X, train_Y = _training_data()
    model = MixedRobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[-1],
    )

    assert model.cat_dims == (1,)
    assert torch.equal(model.raw_train_X, train_X)
    assert _contains_categorical_kernel(model.covar_module)


def test_mixed_robust_gp_posterior_accepts_raw_mixed_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedRobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )
    model.eval()

    test_X = torch.tensor([[0.25, 1.0], [0.75, 2.0]], dtype=torch.double)
    posterior = model.posterior(test_X)

    assert posterior.mean.shape[-2:] == torch.Size([2, 1])
    assert posterior.variance.shape[-2:] == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_robust_gp_to_standard_model_preserves_mixed_covariance() -> None:
    train_X, train_Y = _training_data()
    model = MixedRobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )

    standard_model = model.to_standard_model()

    assert standard_model.covar_module is model.covar_module
    assert _contains_categorical_kernel(standard_model.covar_module)


def test_mixed_robust_gp_exposes_exact_mll() -> None:
    train_X, train_Y = _training_data()
    model = MixedRobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )

    assert model.supports_mll is True
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_robust_gp_allows_unseen_category_values() -> None:
    train_X, train_Y = _training_data()
    model = MixedRobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )
    model.eval()

    posterior = model.posterior(
        torch.tensor([[0.40, 99.0]], dtype=torch.double)
    )

    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

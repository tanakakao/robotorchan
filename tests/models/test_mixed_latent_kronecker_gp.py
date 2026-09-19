import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedLatentKroneckerGP


def _training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.10, 0.0],
            [0.30, 1.0],
            [0.55, 2.0],
            [0.80, 0.0],
        ],
        dtype=torch.double,
    )
    train_T = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_Y = train_X[:, :1] + 0.2 * train_X[:, 1:2] + train_T.squeeze(-1)
    return train_X, train_T, train_Y


def _contains_categorical_kernel(module: torch.nn.Module) -> bool:
    return any(isinstance(child, CategoricalKernel) for child in module.modules())


def test_mixed_latent_kronecker_uses_native_categorical_kernel() -> None:
    train_X, train_T, train_Y = _training_data()
    model = MixedLatentKroneckerGP(
        train_X=train_X,
        train_T=train_T,
        train_Y=train_Y,
        cat_dims=[-1],
    )

    assert model.cat_dims == (1,)
    assert _contains_categorical_kernel(model.covar_module_X)
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_T, train_T)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_latent_kronecker_posterior_accepts_raw_mixed_inputs() -> None:
    train_X, train_T, train_Y = _training_data()
    model = MixedLatentKroneckerGP(
        train_X=train_X,
        train_T=train_T,
        train_Y=train_Y,
        cat_dims=[1],
    )
    model.eval()

    test_X = torch.tensor([[0.20, 1.0], [0.70, 2.0]], dtype=torch.double)
    test_T = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    with model.use_iterative_methods():
        posterior = model.posterior(test_X, test_T)

    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

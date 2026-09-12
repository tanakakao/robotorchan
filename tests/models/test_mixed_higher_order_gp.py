import pytest
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedHigherOrderGP


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.10, 0.0],
            [0.25, 1.0],
            [0.40, 2.0],
            [0.60, 0.0],
            [0.75, 1.0],
            [0.90, 2.0],
        ],
        dtype=torch.double,
    )
    base = train_X[:, :1] + 0.2 * train_X[:, 1:2]
    offsets = torch.tensor([[0.0, 0.2], [0.1, 0.3]], dtype=torch.double)
    train_Y = base[..., None] + offsets
    return train_X, train_Y


def _contains_categorical_kernel(module: torch.nn.Module) -> bool:
    return any(isinstance(child, CategoricalKernel) for child in module.modules())


def test_mixed_higher_order_gp_uses_native_mixed_input_factor() -> None:
    train_X, train_Y = _training_data()
    model = MixedHigherOrderGP(train_X=train_X, train_Y=train_Y, cat_dims=[-1])

    assert model.cat_dims == (1,)
    assert torch.equal(model.raw_train_X, train_X)
    assert _contains_categorical_kernel(model.covar_modules[0])
    assert not _contains_categorical_kernel(model.covar_modules[1])
    assert len(model.covar_modules) == 3


def test_mixed_higher_order_gp_posterior_accepts_raw_mixed_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedHigherOrderGP(train_X=train_X, train_Y=train_Y, cat_dims=[1])
    model.eval()

    test_X = torch.tensor([[0.20, 1.0], [0.80, 2.0]], dtype=torch.double)
    posterior = model.posterior(test_X)

    assert posterior.mean.shape == torch.Size([2, 2, 2])
    assert posterior.variance.shape == torch.Size([2, 2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_higher_order_gp_exposes_exact_mll() -> None:
    train_X, train_Y = _training_data()
    model = MixedHigherOrderGP(train_X=train_X, train_Y=train_Y, cat_dims=[1])

    assert model.supports_mll is True
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_higher_order_gp_rejects_custom_covar_modules() -> None:
    train_X, train_Y = _training_data()

    with pytest.raises(ValueError, match="manages the design-input covariance"):
        MixedHigherOrderGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[1],
            covar_modules=[RBFKernel(), RBFKernel(), RBFKernel()],
        )

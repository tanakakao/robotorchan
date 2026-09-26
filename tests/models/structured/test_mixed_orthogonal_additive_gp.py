import pytest
import torch
from botorch.models.kernels.orthogonal_additive_kernel import OrthogonalAdditiveKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedOrthogonalAdditiveGP


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.10, 10.0],
            [0.20, 20.0],
            [0.35, 30.0],
            [0.50, 10.0],
            [0.65, 20.0],
            [0.80, 30.0],
            [0.90, 10.0],
            [0.95, 20.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.sin(train_X[:, :1] * 3.0) + 0.01 * train_X[:, 1:2]
    return train_X, train_Y


def test_mixed_orthogonal_additive_gp_encodes_categories_internally() -> None:
    train_X, train_Y = _training_data()
    model = MixedOrthogonalAdditiveGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[-1],
    )

    assert model.cat_dims == (1,)
    assert model.raw_input_dim == 2
    assert model.encoded_input_dim == 4
    assert model.covar_module.dim == 4
    assert torch.equal(model.raw_train_X, train_X)
    expected_categories = torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
    assert torch.equal(model.category_values[0], expected_categories)
    assert model.num_components == 5


def test_mixed_orthogonal_additive_gp_posterior_accepts_raw_mixed_inputs() -> None:
    train_X, train_Y = _training_data()
    model = MixedOrthogonalAdditiveGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )
    model.eval()

    test_X = torch.tensor([[0.25, 20.0], [0.75, 30.0]], dtype=torch.double)
    posterior = model.posterior(test_X)

    assert posterior.mean.shape[-2:] == torch.Size([2, 1])
    assert posterior.variance.shape[-2:] == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_orthogonal_additive_gp_rejects_unseen_categories() -> None:
    train_X, train_Y = _training_data()
    model = MixedOrthogonalAdditiveGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )
    model.eval()

    with pytest.raises(ValueError, match="unseen category"):
        model.posterior(torch.tensor([[0.5, 40.0]], dtype=torch.double))


def test_mixed_orthogonal_additive_gp_exposes_exact_mll() -> None:
    train_X, train_Y = _training_data()
    model = MixedOrthogonalAdditiveGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )

    assert model.supports_mll is True
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_orthogonal_additive_gp_validates_custom_encoded_kernel_dim() -> None:
    train_X, train_Y = _training_data()
    wrong_dim_kernel = OrthogonalAdditiveKernel(dim=2, dtype=torch.double)

    with pytest.raises(ValueError, match="encoded input dimension"):
        MixedOrthogonalAdditiveGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[1],
            covar_module=wrong_dim_kernel,
        )


def test_mixed_orthogonal_additive_gp_rejects_custom_input_transform() -> None:
    train_X, train_Y = _training_data()

    with pytest.raises(ValueError, match="manages input_transform internally"):
        MixedOrthogonalAdditiveGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[1],
            input_transform=torch.nn.Identity(),
        )

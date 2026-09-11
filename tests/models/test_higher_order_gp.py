import inspect

import torch
from botorch.models.higher_order_gp import HigherOrderGP as BoTorchHigherOrderGP
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.higher_order import HigherOrderGP


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.05, 0.10],
            [0.20, 0.30],
            [0.40, 0.50],
            [0.60, 0.70],
            [0.80, 0.85],
            [0.95, 0.90],
        ],
        dtype=torch.double,
    )
    base = train_X[:, :1] + 0.5 * train_X[:, 1:2]
    offsets = torch.tensor(
        [[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]],
        dtype=torch.double,
    )
    train_Y = base[..., None] + offsets
    return train_X, train_Y


def test_higher_order_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(HigherOrderGP.__init__)
    upstream = inspect.signature(BoTorchHigherOrderGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_higher_order_gp_uses_common_exact_gp_contract() -> None:
    train_X, train_Y = _training_data()
    original_Y = train_Y.clone()

    model = HigherOrderGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=2),
    )

    assert isinstance(model, BoTorchHigherOrderGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, original_Y)
    assert model.raw_train_Yvar is None
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert set(model.raw_data) == {"train_X", "train_Y", "train_Yvar"}
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_higher_order_gp_raw_buffers_follow_dtype_and_serialize() -> None:
    train_X, train_Y = _training_data()
    model = HigherOrderGP(train_X=train_X, train_Y=train_Y)

    state_dict = model.state_dict()
    model = model.to(dtype=torch.float32)

    assert "_raw_train_X" in state_dict
    assert "_raw_train_Y" in state_dict
    assert "_raw_train_Yvar" not in state_dict
    assert model.raw_train_X.dtype == torch.float32
    assert model.raw_train_Y.dtype == torch.float32


def test_higher_order_gp_matches_upstream_posterior() -> None:
    train_X, train_Y = _training_data()
    wrapper = HigherOrderGP(train_X=train_X, train_Y=train_Y)
    upstream = BoTorchHigherOrderGP(train_X=train_X, train_Y=train_Y)

    upstream_state = {
        name: value for name, value in wrapper.state_dict().items() if not name.startswith("_raw_")
    }
    upstream.load_state_dict(upstream_state)

    test_X = torch.tensor([[0.15, 0.25], [0.75, 0.65]], dtype=torch.double)
    wrapper.eval()
    upstream.eval()

    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)

import inspect

import torch
from botorch.models import ModelListGP as BoTorchModelListGP
from botorch.models import SingleTaskGP as BoTorchSingleTaskGP
from gpytorch.mlls import SumMarginalLogLikelihood

from robotorchan.models import MixedSingleTaskGP, ModelListGP, SingleTaskGP


def _make_child_data(n: int, offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.rand(n, 2, dtype=torch.double)
    train_Y = torch.sin(train_X[:, :1] * 2.0) + offset
    return train_X, train_Y


def _make_mixed_child_data(
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous = torch.rand(n, 2, dtype=torch.double)
    categorical = torch.randint(0, 3, (n, 1)).to(dtype=torch.double)
    train_X = torch.cat([continuous[:, :1], categorical, continuous[:, 1:]], dim=-1)
    mixed_Y = torch.sin(continuous[:, :1] * 2.0) + 0.2 * categorical
    standard_Y = continuous[:, 1:] + 0.1 * categorical
    return train_X, mixed_Y, standard_Y


def test_model_list_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(ModelListGP.__init__)
    upstream = inspect.signature(BoTorchModelListGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_model_list_gp_uses_grouped_raw_data_contract() -> None:
    train_X1, train_Y1 = _make_child_data(8)
    train_X2, train_Y2 = _make_child_data(11, offset=0.5)
    train_Yvar2 = torch.full_like(train_Y2, 1e-4)

    child1 = SingleTaskGP(train_X=train_X1, train_Y=train_Y1)
    child2 = SingleTaskGP(train_X=train_X2, train_Y=train_Y2, train_Yvar=train_Yvar2)
    model = ModelListGP(child1, child2)

    assert isinstance(model, BoTorchModelListGP)
    assert model.supports_mll is True
    assert len(model.models) == 2
    assert torch.equal(model.raw_train_Xs[0], train_X1)
    assert torch.equal(model.raw_train_Xs[1], train_X2)
    assert torch.equal(model.raw_train_Ys[0], train_Y1)
    assert torch.equal(model.raw_train_Ys[1], train_Y2)
    assert model.raw_train_Yvars[0] is None
    assert torch.equal(model.raw_train_Yvars[1], train_Yvar2)
    assert model.raw_train_Xs[0].data_ptr() != train_X1.data_ptr()
    assert model.raw_train_Xs[1].data_ptr() != train_X2.data_ptr()


def test_model_list_gp_make_mll() -> None:
    train_X1, train_Y1 = _make_child_data(7)
    train_X2, train_Y2 = _make_child_data(9, offset=0.2)
    model = ModelListGP(
        SingleTaskGP(train_X=train_X1, train_Y=train_Y1),
        SingleTaskGP(train_X=train_X2, train_Y=train_Y2),
    )

    mll = model.make_mll()

    assert isinstance(mll, SumMarginalLogLikelihood)
    assert mll.model is model
    assert mll.likelihood is model.likelihood
    assert len(mll.mlls) == 2


def test_model_list_gp_does_not_invent_raw_data_for_upstream_children() -> None:
    train_X1, train_Y1 = _make_child_data(8)
    train_X2, train_Y2 = _make_child_data(8, offset=0.3)

    upstream_child = BoTorchSingleTaskGP(train_X=train_X1, train_Y=train_Y1)
    wrapped_child = SingleTaskGP(train_X=train_X2, train_Y=train_Y2)
    model = ModelListGP(upstream_child, wrapped_child)

    assert model.raw_train_Xs[0] is None
    assert model.raw_train_Ys[0] is None
    assert model.raw_train_Yvars[0] is None
    assert torch.equal(model.raw_train_Xs[1], train_X2)
    assert torch.equal(model.raw_train_Ys[1], train_Y2)


def test_model_list_gp_matches_upstream_posterior() -> None:
    train_X1, train_Y1 = _make_child_data(8)
    train_X2, train_Y2 = _make_child_data(10, offset=0.4)

    wrapped_child1 = SingleTaskGP(train_X=train_X1, train_Y=train_Y1)
    wrapped_child2 = SingleTaskGP(train_X=train_X2, train_Y=train_Y2)
    upstream_child1 = BoTorchSingleTaskGP(train_X=train_X1, train_Y=train_Y1)
    upstream_child2 = BoTorchSingleTaskGP(train_X=train_X2, train_Y=train_Y2)

    for wrapped, upstream in (
        (wrapped_child1, upstream_child1),
        (wrapped_child2, upstream_child2),
    ):
        upstream_state = {
            name: value
            for name, value in wrapped.state_dict().items()
            if not name.startswith("_raw_")
        }
        upstream.load_state_dict(upstream_state)

    wrapper = ModelListGP(wrapped_child1, wrapped_child2)
    upstream = BoTorchModelListGP(upstream_child1, upstream_child2)
    test_X = torch.rand(5, 2, dtype=torch.double)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)


def test_model_list_gp_composes_mixed_and_standard_children() -> None:
    train_X, mixed_Y, standard_Y = _make_mixed_child_data(14)
    mixed_child = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=mixed_Y,
        cat_dims=[1],
    )
    standard_child = SingleTaskGP(
        train_X=train_X,
        train_Y=standard_Y,
    )
    model = ModelListGP(mixed_child, standard_child)

    assert len(model.models) == 2
    assert isinstance(model.models[0], MixedSingleTaskGP)
    assert isinstance(model.models[1], SingleTaskGP)
    assert torch.equal(model.raw_train_Xs[0], train_X)
    assert torch.equal(model.raw_train_Xs[1], train_X)
    assert torch.equal(model.raw_train_Ys[0], mixed_Y)
    assert torch.equal(model.raw_train_Ys[1], standard_Y)
    assert isinstance(model.make_mll(), SumMarginalLogLikelihood)


def test_model_list_gp_mixed_composition_posterior() -> None:
    train_X, mixed_Y, standard_Y = _make_mixed_child_data(16)
    model = ModelListGP(
        MixedSingleTaskGP(
            train_X=train_X,
            train_Y=mixed_Y,
            cat_dims=[1],
        ),
        SingleTaskGP(
            train_X=train_X,
            train_Y=standard_Y,
        ),
    )

    continuous = torch.rand(5, 2, dtype=torch.double)
    categorical = torch.randint(0, 3, (5, 1)).to(dtype=torch.double)
    test_X = torch.cat(
        [continuous[:, :1], categorical, continuous[:, 1:]],
        dim=-1,
    )

    model.eval()
    posterior = model.posterior(test_X)

    assert posterior.mean.shape == torch.Size([5, 2])
    assert posterior.variance.shape == torch.Size([5, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

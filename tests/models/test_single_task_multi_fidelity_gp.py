import inspect

import torch
from botorch.models import SingleTaskMultiFidelityGP as BoTorchSingleTaskMultiFidelityGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import SingleTaskMultiFidelityGP


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.rand(14, 2, dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 14, dtype=torch.double).unsqueeze(-1)
    train_X = torch.cat([design, fidelity], dim=-1)
    train_Y = torch.sin(design[:, :1] * 2.0) + design[:, 1:2] + 0.3 * (1.0 - fidelity)
    return train_X, train_Y


def test_multi_fidelity_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(SingleTaskMultiFidelityGP.__init__)
    upstream = inspect.signature(BoTorchSingleTaskMultiFidelityGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_multi_fidelity_gp_uses_common_wrapper_contract() -> None:
    train_X, train_Y = _make_data()
    train_Yvar = torch.full_like(train_Y, 1e-4)

    model = SingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        data_fidelities=[-1],
    )

    assert isinstance(model, BoTorchSingleTaskMultiFidelityGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_multi_fidelity_gp_matches_upstream_posterior() -> None:
    train_X, train_Y = _make_data()
    wrapper = SingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        data_fidelities=[-1],
    )
    upstream = BoTorchSingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        data_fidelities=[-1],
    )

    upstream_state = {
        name: value
        for name, value in wrapper.state_dict().items()
        if not name.startswith("_raw_")
    }
    upstream.load_state_dict(upstream_state)

    test_design = torch.rand(5, 2, dtype=torch.double)
    test_fidelity = torch.tensor([[1.0], [0.8], [0.6], [0.4], [0.2]], dtype=torch.double)
    test_X = torch.cat([test_design, test_fidelity], dim=-1)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)

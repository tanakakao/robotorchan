import inspect

import pytest
import torch
from botorch.models import SingleTaskMultiFidelityGP as BoTorchSingleTaskMultiFidelityGP
from botorch.models.kernels.downsampling import DownsamplingKernel
from gpytorch.kernels import AdditiveKernel, Kernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedSingleTaskMultiFidelityGP, SingleTaskMultiFidelityGP


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.rand(14, 2, dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 14, dtype=torch.double).unsqueeze(-1)
    train_X = torch.cat([design, fidelity], dim=-1)
    train_Y = torch.sin(design[:, :1] * 2.0) + design[:, 1:2] + 0.3 * (1.0 - fidelity)
    return train_X, train_Y


def _make_mixed_data() -> tuple[torch.Tensor, torch.Tensor]:
    continuous = torch.rand(16, 1, dtype=torch.double)
    categorical = torch.randint(0, 3, (16, 1)).to(dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 16, dtype=torch.double).unsqueeze(-1)
    train_X = torch.cat([continuous, categorical, fidelity], dim=-1)
    train_Y = torch.sin(continuous * 2.0) + 0.15 * categorical + 0.3 * (1.0 - fidelity)
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
        name: value for name, value in wrapper.state_dict().items() if not name.startswith("_raw_")
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


def test_mixed_multi_fidelity_gp_uses_mixed_data_and_native_fidelity_kernels() -> None:
    train_X, train_Y = _make_mixed_data()
    train_Yvar = torch.full_like(train_Y, 1e-4)

    model = MixedSingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        cat_dims=[1],
        data_fidelities=[-1],
    )

    assert isinstance(model, BoTorchSingleTaskMultiFidelityGP)
    assert model.supports_mll is True
    assert model.cat_dims == (1,)
    assert model.fidelity_dims == (2,)
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    assert isinstance(model.covar_module.kernels[0], AdditiveKernel)
    assert isinstance(model.covar_module.kernels[1], DownsamplingKernel)


def test_mixed_multi_fidelity_gp_posterior_supports_mixed_features() -> None:
    train_X, train_Y = _make_mixed_data()
    model = MixedSingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        data_fidelities=[2],
    )

    test_continuous = torch.rand(5, 1, dtype=torch.double)
    test_categorical = torch.randint(0, 3, (5, 1)).to(dtype=torch.double)
    test_fidelity = torch.tensor([[1.0], [0.8], [0.6], [0.4], [0.2]], dtype=torch.double)
    test_X = torch.cat([test_continuous, test_categorical, test_fidelity], dim=-1)

    model.eval()
    posterior = model.posterior(test_X)

    assert posterior.mean.shape == torch.Size([5, 1])
    assert posterior.variance.shape == torch.Size([5, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_multi_fidelity_gp_supports_categorical_only_design_features() -> None:
    categorical = torch.randint(0, 3, (12, 2)).to(dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_X = torch.cat([categorical, fidelity], dim=-1)
    train_Y = categorical[:, :1] + 0.2 * categorical[:, 1:2] + 0.3 * (1.0 - fidelity)

    model = MixedSingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[0, 1],
        data_fidelities=[-1],
    )

    assert model.cat_dims == (0, 1)
    assert model.fidelity_dims == (2,)
    assert isinstance(model.covar_module.kernels[0], ScaleKernel)
    assert isinstance(model.covar_module.kernels[1], DownsamplingKernel)


def test_mixed_multi_fidelity_gp_custom_continuous_factory_excludes_fidelity() -> None:
    train_X, train_Y = _make_mixed_data()
    calls: list[tuple[torch.Size, int, list[int]]] = []

    def factory(batch_shape: torch.Size, ard_num_dims: int, active_dims: list[int]) -> Kernel:
        calls.append((batch_shape, ard_num_dims, active_dims))
        return ScaleKernel(
            RBFKernel(
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
            ),
            batch_shape=batch_shape,
        )

    MixedSingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        data_fidelities=[2],
        cont_kernel_factory=factory,
    )

    assert len(calls) == 2
    assert all(call[1] == 1 for call in calls)
    assert all(call[2] == [0] for call in calls)


def test_mixed_multi_fidelity_gp_rejects_categorical_fidelity_overlap() -> None:
    train_X, train_Y = _make_mixed_data()

    with pytest.raises(ValueError, match="must be disjoint"):
        MixedSingleTaskMultiFidelityGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[-1],
            data_fidelities=[-1],
        )


def test_mixed_multi_fidelity_gp_rejects_linear_truncated_kernel() -> None:
    train_X, train_Y = _make_mixed_data()

    with pytest.raises(ValueError, match="requires linear_truncated=False"):
        MixedSingleTaskMultiFidelityGP(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[1],
            data_fidelities=[2],
            linear_truncated=True,
        )

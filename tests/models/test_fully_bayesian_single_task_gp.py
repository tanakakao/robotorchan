import inspect

import pytest
import torch
from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP as BoTorchSaasFullyBayesianSingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize

from robotorchan.models import (
    MixedSaasFullyBayesianSingleTaskGP,
    SaasFullyBayesianSingleTaskGP,
    UnsupportedModelOperationError,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.05, 0.95, 24, dtype=torch.double).reshape(8, 3)
    train_Y = (2.0 * train_X[:, :1] - train_X[:, 1:2]).clone()
    train_Yvar = torch.full_like(train_Y, 1e-4)
    return train_X, train_Y, train_Yvar


def test_saas_single_task_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(SaasFullyBayesianSingleTaskGP.__init__)
    upstream = inspect.signature(BoTorchSaasFullyBayesianSingleTaskGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_saas_single_task_retains_raw_data_before_outcome_transform() -> None:
    train_X, train_Y, train_Yvar = _training_data()
    original_Y = train_Y.clone()
    original_Yvar = train_Yvar.clone()

    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        outcome_transform=Standardize(m=1),
    )

    assert isinstance(model, BoTorchSaasFullyBayesianSingleTaskGP)
    assert model.supports_mll is False
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, original_Y)
    assert torch.equal(model.raw_train_Yvar, original_Yvar)
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert model.raw_train_Yvar.data_ptr() != train_Yvar.data_ptr()
    assert set(model.raw_data) == {"train_X", "train_Y", "train_Yvar"}
    assert not torch.equal(model.pyro_model.train_Y, original_Y)


def test_saas_single_task_rejects_mll_training() -> None:
    train_X, train_Y, _ = _training_data()
    model = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)

    with pytest.raises(UnsupportedModelOperationError, match="does not support make_mll"):
        model.make_mll()


def test_saas_single_task_raw_buffers_follow_dtype_and_serialize() -> None:
    train_X, train_Y, train_Yvar = _training_data()
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
    )

    state_dict = model.state_dict()
    model = model.to(dtype=torch.float32)

    assert "_raw_train_X" in state_dict
    assert "_raw_train_Y" in state_dict
    assert "_raw_train_Yvar" in state_dict
    assert model.raw_train_X.dtype == torch.float32
    assert model.raw_train_Y.dtype == torch.float32
    assert model.raw_train_Yvar.dtype == torch.float32


def test_saas_single_task_matches_upstream_posterior_with_same_mcmc_samples() -> None:
    train_X, train_Y, _ = _training_data()
    wrapper = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
    upstream = BoTorchSaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)

    samples = wrapper.pyro_model.get_dummy_mcmc_samples(
        num_mcmc_samples=3,
        dtype=train_X.dtype,
        device=train_X.device,
    )
    wrapper.load_mcmc_samples(samples)
    upstream.load_mcmc_samples({name: value.clone() for name, value in samples.items()})

    test_X = torch.tensor(
        [[0.15, 0.35, 0.55], [0.45, 0.65, 0.85]],
        dtype=torch.double,
    )
    wrapper.eval()
    upstream.eval()

    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)


def test_mixed_saas_single_task_encodes_categories_internally() -> None:
    continuous = torch.linspace(0.1, 0.9, 9, dtype=torch.double).unsqueeze(-1)
    category = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.double).unsqueeze(-1)
    train_X = torch.cat([continuous, category], dim=-1)
    train_Y = continuous + 0.2 * category

    model = MixedSaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[-1],
    )
    samples = model.pyro_model.get_dummy_mcmc_samples(
        num_mcmc_samples=3,
        dtype=train_X.dtype,
        device=train_X.device,
    )
    model.load_mcmc_samples(samples)

    test_X = torch.tensor([[0.25, 1.0], [0.75, 2.0]], dtype=torch.double)
    model.eval()
    posterior = model.posterior(test_X)

    assert model.cat_dims == (1,)
    assert model.raw_input_dim == 2
    assert model.encoded_input_dim == 4
    assert torch.equal(model.raw_train_X, train_X)
    assert model.pyro_model.train_X.shape[-1] == 4
    assert posterior.mean.shape == torch.Size([3, 2, 1])
    assert posterior.variance.shape == torch.Size([3, 2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_saas_single_task_rejects_unseen_categories() -> None:
    train_X = torch.tensor(
        [[0.1, 0.0], [0.3, 1.0], [0.6, 0.0], [0.9, 1.0]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1]
    model = MixedSaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
    )
    samples = model.pyro_model.get_dummy_mcmc_samples(
        num_mcmc_samples=2,
        dtype=train_X.dtype,
        device=train_X.device,
    )
    model.load_mcmc_samples(samples)

    with pytest.raises(ValueError, match="unseen category"):
        model.posterior(torch.tensor([[0.5, 2.0]], dtype=torch.double))

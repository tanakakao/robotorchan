import inspect

import pytest
import torch
from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP as BoTorchSaasFullyBayesianSingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize

from robotorchan.models import (
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


def test_saas_single_task_raw_buffers_follow_dtype_and_are_non_persistent() -> None:
    train_X, train_Y, train_Yvar = _training_data()
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
    )

    state_dict = model.state_dict()
    model = model.to(dtype=torch.float32)

    assert "_raw_train_X" not in state_dict
    assert "_raw_train_Y" not in state_dict
    assert "_raw_train_Yvar" not in state_dict
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

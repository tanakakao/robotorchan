import inspect

import pytest
import torch
from botorch.models.fully_bayesian_multitask import (
    SaasFullyBayesianMultiTaskGP as BoTorchSaasFullyBayesianMultiTaskGP,
)

from robotorchan.models import (
    SaasFullyBayesianMultiTaskGP,
    UnsupportedModelOperationError,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [0.10, 0.20],
            [0.30, 0.40],
            [0.60, 0.50],
            [0.85, 0.75],
            [0.15, 0.25],
            [0.35, 0.45],
            [0.65, 0.55],
            [0.90, 0.80],
        ],
        dtype=torch.double,
    )
    tasks = torch.tensor([[0.0]] * 4 + [[1.0]] * 4, dtype=torch.double)
    train_X = torch.cat([features, tasks], dim=-1)
    train_Y = torch.tensor(
        [[-1.0], [-0.3], [0.3], [1.0], [-0.9], [-0.2], [0.2], [0.9]],
        dtype=torch.double,
    )
    train_Yvar = torch.full_like(train_Y, 1e-4)
    return train_X, train_Y, train_Yvar


def test_saas_multi_task_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(SaasFullyBayesianMultiTaskGP.__init__)
    upstream = inspect.signature(BoTorchSaasFullyBayesianMultiTaskGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_saas_multi_task_retains_raw_long_format_training_data() -> None:
    train_X, train_Y, train_Yvar = _training_data()

    model = SaasFullyBayesianMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        train_Yvar=train_Yvar,
    )

    assert isinstance(model, BoTorchSaasFullyBayesianMultiTaskGP)
    assert model.supports_mll is False
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert model.raw_train_Yvar.data_ptr() != train_Yvar.data_ptr()
    assert set(model.raw_data) == {"train_X", "train_Y", "train_Yvar"}


def test_saas_multi_task_rejects_mll_training() -> None:
    train_X, train_Y, _ = _training_data()
    model = SaasFullyBayesianMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
    )

    with pytest.raises(UnsupportedModelOperationError, match="does not support make_mll"):
        model.make_mll()


def test_saas_multi_task_raw_buffers_follow_dtype_and_serialize() -> None:
    train_X, train_Y, train_Yvar = _training_data()
    model = SaasFullyBayesianMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
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


def test_saas_multi_task_matches_upstream_posterior_with_same_mcmc_samples() -> None:
    train_X, train_Y, _ = _training_data()
    wrapper = SaasFullyBayesianMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
    )
    upstream = BoTorchSaasFullyBayesianMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
    )

    samples = wrapper.pyro_model.get_dummy_mcmc_samples(
        num_mcmc_samples=3,
        dtype=train_X.dtype,
        device=train_X.device,
    )
    wrapper.load_mcmc_samples(samples)
    upstream.load_mcmc_samples({name: value.clone() for name, value in samples.items()})

    test_X = torch.tensor([[0.20, 0.30], [0.70, 0.60]], dtype=torch.double)
    wrapper.eval()
    upstream.eval()

    wrapper_posterior = wrapper.posterior(test_X, output_indices=[0, 1])
    upstream_posterior = upstream.posterior(test_X, output_indices=[0, 1])

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)


def test_saas_multi_task_supports_one_hot_encoded_categories() -> None:
    continuous = torch.tensor(
        [[0.1], [0.3], [0.6], [0.8], [0.15], [0.35], [0.65], [0.85]],
        dtype=torch.double,
    )
    category = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0])
    one_hot = torch.nn.functional.one_hot(category, num_classes=3).to(torch.double)
    tasks = torch.tensor([[0.0]] * 4 + [[1.0]] * 4, dtype=torch.double)
    train_X = torch.cat([continuous, one_hot, tasks], dim=-1)
    train_Y = continuous + 0.1 * category.to(torch.double).unsqueeze(-1)

    model = SaasFullyBayesianMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
    )
    samples = model.pyro_model.get_dummy_mcmc_samples(
        num_mcmc_samples=3,
        dtype=train_X.dtype,
        device=train_X.device,
    )
    model.load_mcmc_samples(samples)

    test_continuous = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    test_category = torch.tensor([1, 2])
    test_one_hot = torch.nn.functional.one_hot(
        test_category, num_classes=3
    ).to(torch.double)
    test_X = torch.cat([test_continuous, test_one_hot], dim=-1)

    model.eval()
    posterior = model.posterior(test_X, output_indices=[0, 1])

    assert torch.equal(model.raw_train_X, train_X)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

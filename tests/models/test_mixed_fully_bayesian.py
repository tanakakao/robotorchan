import pytest
import torch

from robotorchan.models import (
    MixedSaasFullyBayesianMultiTaskGP,
    MixedSaasFullyBayesianSingleTaskGP,
    UnsupportedModelOperationError,
)


def test_mixed_saas_single_task_encodes_before_pyro_and_accepts_raw_posterior():
    X = torch.tensor([[0.1, 10.0], [0.3, 20.0], [0.6, 10.0], [0.9, 20.0]], dtype=torch.double)
    Y = torch.sin(X[:, :1])
    model = MixedSaasFullyBayesianSingleTaskGP(X, Y, cat_dims=[-1])
    assert model.cat_dims == (1,)
    assert model.encoded_input_dim == 3
    torch.testing.assert_close(model.raw_train_X, X)
    assert model.pyro_model.train_X.shape[-1] == 3
    with pytest.raises(UnsupportedModelOperationError):
        model.make_mll()
    samples = model.pyro_model.get_dummy_mcmc_samples(num_mcmc_samples=2, dtype=X.dtype, device=X.device)
    model.load_mcmc_samples(samples)
    model.eval()
    assert model.posterior(X[:2]).mean.shape[-2:] == torch.Size([2, 1])
    with pytest.raises(ValueError, match="unseen category"):
        model.posterior(torch.tensor([[0.2, 30.0]], dtype=torch.double))


def test_mixed_saas_multitask_keeps_task_feature_structural():
    design = torch.tensor([[0.1, 10.0], [0.3, 20.0], [0.6, 10.0], [0.9, 20.0]], dtype=torch.double)
    X = torch.cat([design.repeat(2, 1), torch.tensor([[0.0]] * 4 + [[1.0]] * 4, dtype=torch.double)], dim=-1)
    Y = torch.linspace(-1, 1, 8, dtype=torch.double).unsqueeze(-1)
    model = MixedSaasFullyBayesianMultiTaskGP(X, Y, task_feature=-1, cat_dims=[1])
    assert model.raw_task_feature == 2
    assert model.encoded_task_feature == 3
    assert model.cat_dims == (1,)
    torch.testing.assert_close(model.raw_train_X, X)
    with pytest.raises(ValueError, match="structural dimensions"):
        MixedSaasFullyBayesianMultiTaskGP(X, Y, task_feature=-1, cat_dims=[-1])

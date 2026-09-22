"""Runtime smoke tests for capability-advertised BoTorch workflows."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import KroneckerMultiTaskGP, MixedSingleTaskGP, SingleTaskGP


def test_single_task_gp_runtime_supports_mc_acquisition() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)

    model = SingleTaskGP(train_x, train_y)
    model.eval()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    samples = posterior.rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_mixed_single_task_gp_runtime_supports_mc_acquisition() -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 0.0],
            [0.6, 1.0],
            [0.8, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.sin(train_x[:, :1] * 3.0) + 0.2 * train_x[:, 1:].eq(1.0).to(dtype=torch.double)

    model = MixedSingleTaskGP(train_x, train_y, cat_dims=[1])
    model.eval()

    test_x = torch.tensor([[0.25, 0.0], [0.75, 1.0]], dtype=torch.double)
    samples = model.posterior(test_x).rsample(torch.Size([4]))

    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(torch.tensor([[[0.5, 1.0]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()



def test_kronecker_multitask_gp_runtime_supports_multi_output_sampling() -> None:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat(
        [
            torch.sin(train_x * 3.0),
            torch.cos(train_x * 3.0),
        ],
        dim=-1,
    )

    model = KroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    test_x = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([4, 2, 2])
    assert torch.isfinite(samples).all()

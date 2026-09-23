"""Runtime contracts for nonstationary block-design Kronecker models."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import IIDNormalSampler

from robotorchan.models import NonstationaryKroneckerMultiTaskGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    y1 = torch.sin(train_x.squeeze(-1) * 3.0)
    y2 = torch.cos(train_x.squeeze(-1) * 2.0)
    return train_x, torch.stack((y1, y2), dim=-1)


def test_nonstationary_kronecker_runtime_contract() -> None:
    train_x, train_y = _data()
    model = NonstationaryKroneckerMultiTaskGP(train_x, train_y)

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.local_lengthscale(train_x).shape == train_x.shape

    model.eval()
    posterior = model.posterior(train_x[:2])
    assert posterior.mean.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    samples = posterior.rsample(torch.Size([3]))
    assert samples.shape == torch.Size([3, 2, 2])
    assert torch.isfinite(samples).all()


def test_nonstationary_kronecker_scalarized_mc_acquisition() -> None:
    train_x, train_y = _data()
    model = NonstationaryKroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    objective = GenericMCObjective(lambda samples, X=None: samples.sum(dim=-1))
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=IIDNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    value = acquisition(torch.tensor([[[0.25]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()

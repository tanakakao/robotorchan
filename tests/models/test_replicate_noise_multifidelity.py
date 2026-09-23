"""Runtime contracts for replicate-noise multi-fidelity regression."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import IIDNormalSampler

from robotorchan.models import ReplicateNoiseMultiFidelityGP


def _replicates() -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        [0.1, 0.2],
        [0.1, 0.2],
        [0.1, 1.0],
        [0.1, 1.0],
        [0.7, 0.2],
        [0.7, 0.2],
        [0.7, 1.0],
        [0.7, 1.0],
    ]
    values = [[0.2], [0.4], [0.8], [1.0], [0.7], [0.9], [1.3], [1.5]]
    return torch.tensor(rows, dtype=torch.double), torch.tensor(values, dtype=torch.double)


def test_replicate_noise_multifidelity_keeps_fidelity_groups_separate() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        data_fidelities=[-1],
    )

    assert model.raw_train_X.shape == torch.Size([4, 2])
    assert model.raw_replicate_X.shape == train_x.shape
    assert torch.equal(model.raw_replicate_X, train_x)
    assert torch.equal(model.raw_replicate_Y, train_y)
    assert torch.equal(model.replicate_counts, torch.full((4,), 2, dtype=torch.long))
    assert torch.unique(model.raw_train_X[:, -1]).numel() == 2
    assert torch.all(model.raw_train_Yvar > 0)


def test_replicate_noise_multifidelity_posterior_and_mc_acquisition() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    model.eval()

    candidates = torch.tensor([[0.3, 1.0], [0.8, 1.0]], dtype=torch.double)
    posterior = model.posterior(candidates)
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()

    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=IIDNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(candidates[:1].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_replicate_noise_multifidelity_make_mll() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    assert model.make_mll().model is model

"""Runtime contracts for replicate-noise multi-fidelity regression."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf
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


def test_replicate_noise_multifidelity_uses_variance_of_mean() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        data_fidelities=[-1],
    )

    expected_observation_variance = torch.full((4, 1), 0.02, dtype=torch.double)
    expected_mean_variance = expected_observation_variance / 2.0
    assert torch.allclose(model.replicate_variance, expected_observation_variance)
    assert torch.allclose(model.raw_train_Yvar, expected_mean_variance)


def test_replicate_noise_multifidelity_accepts_iteration_fidelity() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        iteration_fidelity=-1,
    )
    assert torch.equal(model.raw_replicate_X, train_x)
    assert torch.isfinite(model.posterior(model.raw_train_X[:2]).mean).all()


def test_replicate_noise_multifidelity_supports_native_mf_kg() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    model.eval()

    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

    def project(X: torch.Tensor) -> torch.Tensor:
        projected = X.clone()
        projected[..., 1] = 1.0
        return projected

    _, current_value = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=8,
        fixed_features={1: 1.0},
    )
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=4,
        current_value=current_value,
        cost_aware_utility=cost_utility,
        project=project,
    )
    X = torch.rand(
        1,
        acquisition.get_augmented_q_batch_size(q=1),
        2,
        dtype=torch.double,
    )
    X[..., 1] = 0.2
    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_replicate_noise_multifidelity_optimizes_candidate() -> None:
    train_x, train_y = _replicates()
    model = ReplicateNoiseMultiFidelityGP.from_replicates(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    model.eval()
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=IIDNormalSampler(sample_shape=torch.Size([16])),
    )
    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0, 0.2], [1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=16,
    )
    assert candidate.shape == torch.Size([1, 2])
    assert 0.0 <= candidate[0, 0] <= 1.0
    assert 0.2 <= candidate[0, 1] <= 1.0
    assert torch.isfinite(value).all()

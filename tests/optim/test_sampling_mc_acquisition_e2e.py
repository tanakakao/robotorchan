"""End-to-end posterior sampling through MC acquisition optimization."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import RandomForestSurrogate, SingleTaskGP
from robotorchan.optim import TreeEnsembleSearchStrategy
from robotorchan.optim.backends import optimize_acqf_botorch


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_y = -(train_x - 0.7).square() + 1.0
    return train_x, train_y


def test_gp_sampling_mc_acquisition_runs_through_botorch_optimizer() -> None:
    train_x, train_y = _training_data()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=123)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf_botorch(
        acquisition,
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=32,
    )

    assert candidate.shape == torch.Size([2, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])


def test_tree_sampling_mc_acquisition_runs_through_derivative_free_search() -> None:
    train_x, train_y = _training_data()
    model = RandomForestSurrogate(
        train_x,
        train_y,
        n_estimators=12,
        random_state=0,
    )
    model.fit()
    sampler = IndexSampler(torch.Size([32]), seed=456)
    acquisition = qExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )
    strategy = TreeEnsembleSearchStrategy(
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        num_samples=128,
        seed=17,
    )

    result = strategy.optimize(acquisition, q=2)

    assert result.candidates.shape == torch.Size([2, 1])
    assert result.acquisition_value is not None
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(result.acquisition_value)

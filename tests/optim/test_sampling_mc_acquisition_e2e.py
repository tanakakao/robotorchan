"""End-to-end posterior sampling through MC acquisition optimization."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.risk_measures import Expectation
from botorch.models.transforms.input import InputPerturbation
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


def test_gp_qlogei_preserves_q_contract_and_candidate_gradients() -> None:
    train_x, train_y = _training_data()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=789)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    for q in (1, 2, 3):
        X = torch.linspace(0.2, 0.8, q, dtype=torch.double).reshape(1, q, 1)
        X.requires_grad_(True)
        acquisition_value = acquisition(X)
        gradient = torch.autograd.grad(acquisition_value.sum(), X)[0]

        assert acquisition_value.shape == torch.Size([1])
        assert gradient.shape == X.shape
        assert torch.isfinite(gradient).all()

        candidate, optimized_value = optimize_acqf_botorch(
            acquisition,
            bounds,
            q=q,
            num_restarts=2,
            raw_samples=16,
        )

        assert candidate.shape == torch.Size([q, 1])
        assert optimized_value.numel() == 1
        assert torch.isfinite(candidate).all()
        assert torch.isfinite(optimized_value).all()


def test_gp_qlogei_runs_native_sequential_q_batch() -> None:
    train_x, train_y = _training_data()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=321),
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidate, value = optimize_acqf_botorch(
        acquisition,
        bounds,
        q=3,
        num_restarts=2,
        raw_samples=16,
        sequential=True,
    )

    assert candidate.shape == torch.Size([3, 1])
    assert value.shape == torch.Size([3])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])


def test_gp_async_pending_lifecycle_runs_end_to_end() -> None:
    train_x, train_y = _training_data()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    model = SingleTaskGP(train_x, train_y)
    model.eval()
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=401),
    )
    candidate_a, _ = optimize_acqf_botorch(
        acquisition,
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
    )

    acquisition.set_X_pending(candidate_a)
    candidate_b, _ = optimize_acqf_botorch(
        acquisition,
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
    )

    observed_a = -(candidate_a - 0.7).square() + 1.0
    updated_x = torch.cat([train_x, candidate_a], dim=0)
    updated_y = torch.cat([train_y, observed_a], dim=0)
    updated_model = SingleTaskGP(updated_x, updated_y)
    updated_model.eval()
    updated_acquisition = qLogNoisyExpectedImprovement(
        model=updated_model,
        X_baseline=updated_x,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=402),
        X_pending=candidate_b,
    )
    candidate_c, value_c = optimize_acqf_botorch(
        updated_acquisition,
        bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
    )

    assert candidate_a.shape == torch.Size([1, 1])
    assert candidate_b.shape == torch.Size([1, 1])
    assert candidate_c.shape == torch.Size([1, 1])
    assert updated_x.shape[0] == train_x.shape[0] + 1
    assert torch.equal(updated_x[-1:], candidate_a)
    assert not torch.any(torch.all(updated_x == candidate_b, dim=-1))
    assert torch.isfinite(candidate_c).all()
    assert torch.isfinite(value_c).all()


def test_input_perturbation_qbatch_runs_with_pending_candidate() -> None:
    train_x, train_y = _training_data()
    perturbations = torch.tensor([[0.0], [0.01], [-0.01]], dtype=torch.double)
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=InputPerturbation(perturbation_set=perturbations),
    )
    model.eval()
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=159),
        objective=Expectation(n_w=perturbations.shape[0]),
        prune_baseline=False,
    )
    acquisition.set_X_pending(torch.tensor([[0.45]], dtype=torch.double))
    bounds = torch.tensor([[0.05], [0.95]], dtype=torch.double)

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
    assert torch.equal(model.train_inputs[0], train_x)

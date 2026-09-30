"""Tests for mixed-variable acquisition optimization."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.standard.single_task import MixedSingleTaskGP
from robotorchan.optim import CandidateConstraints, MixedSpaceStrategy


def test_mixed_space_strategy_respects_category_and_linear_constraint() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.0], [0.7, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2 * train_X[:, 1:2]).sin()
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = PosteriorMean(model)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0]),
                torch.tensor([-1.0], dtype=torch.double),
                -0.6,
            ),
        ),
    )
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=16,
        constraints=constraints,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert result.candidates[0, 0] <= 0.6 + 1e-6
    assert result.candidates[0, 1].item() in {0.0, 1.0}


def test_mixed_space_strategy_respects_intrapoint_nonlinear_constraint() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.0], [0.7, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2 * train_X[:, 1:2]).sin()
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = PosteriorMean(model)

    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.36) - x[0].square()

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((constraint, True),),
    )
    initial_conditions = torch.tensor(
        [[[0.1, 0.0]], [[0.2, 0.0]], [[0.3, 0.0]], [[0.4, 0.0]]],
        dtype=torch.double,
    )
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=4,
        raw_samples=32,
        constraints=constraints,
        batch_initial_conditions=initial_conditions,
    )

    result = strategy.optimize(acquisition)

    assert result.candidates.shape == torch.Size([1, 2])
    assert result.candidates[0, 1].item() in {0.0, 1.0}
    assert constraint(result.candidates[0]) >= -1e-6


def test_mixed_space_strategy_rejects_interpoint_nonlinear_constraint() -> None:
    def constraint(X: torch.Tensor) -> torch.Tensor:
        return (X[0] - X[-1]).square().sum() - X.new_tensor(0.25)

    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        constraints=CandidateConstraints(
            nonlinear_inequality_constraints=((constraint, False),),
        ),
        batch_initial_conditions=torch.tensor([[[0.1, 0.0]]], dtype=torch.double),
    )

    try:
        strategy.optimize(None, q=2)  # type: ignore[arg-type]
    except ValueError as error:
        assert str(error) == (
            "BoTorch mixed acquisition optimization does not support "
            "inter-point nonlinear constraints."
        )
    else:
        raise AssertionError("Expected inter-point nonlinear constraints to be rejected.")


def test_mixed_space_strategy_requires_initial_conditions_for_nonlinear_constraint() -> None:
    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.36) - x[0].square()

    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        constraints=CandidateConstraints(
            nonlinear_inequality_constraints=((constraint, True),),
        ),
    )

    try:
        strategy.optimize(None)  # type: ignore[arg-type]
    except ValueError as error:
        assert str(error) == (
            "Nonlinear candidate constraints require feasible batch_initial_conditions "
            "or an ic_generator."
        )
    else:
        raise AssertionError("Expected nonlinear constraints without initial conditions to fail.")


def test_mixed_nonlinear_constraint_forwards_batch_limit(monkeypatch) -> None:
    def constraint(x: torch.Tensor) -> torch.Tensor:
        return x[0]

    initial_conditions = torch.tensor([[[0.5, 0.0]]], dtype=torch.double)
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        constraints=CandidateConstraints(
            nonlinear_inequality_constraints=((constraint, True),),
        ),
        batch_initial_conditions=initial_conditions,
    )
    captured = {}

    def fake_optimize_acqf_mixed(**kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 2, dtype=torch.double), torch.tensor(0.0, dtype=torch.double)

    monkeypatch.setattr(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf_mixed",
        fake_optimize_acqf_mixed,
    )
    strategy.optimize(None)  # type: ignore[arg-type]

    assert captured["options"]["batch_limit"] == 1
    assert captured["batch_initial_conditions"] is initial_conditions


def test_mixed_space_strategy_runs_q_batch_with_pending_candidate() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.25, 0.0], [0.5, 1.0], [0.75, 1.0], [1.0, 0.0]],
        dtype=torch.double,
    )
    train_Y = torch.sin(train_X[:, :1] * 3.0) + 0.15 * train_X[:, 1:2]
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=23)
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=sampler,
    )
    pending = torch.tensor([[0.4, 1.0]], dtype=torch.double)
    acquisition.set_X_pending(pending)
    strategy = MixedSpaceStrategy(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=16,
    )

    result = strategy.optimize(acquisition, q=2)

    assert result.candidates.shape == torch.Size([2, 2])
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(result.acquisition_value).all()
    assert torch.all(result.candidates >= strategy.bounds[0])
    assert torch.all(result.candidates <= strategy.bounds[1])
    assert set(result.candidates[:, 1].tolist()) <= {0.0, 1.0}


def test_mixed_space_strategy_forwards_none_raw_samples_with_explicit_initial_conditions(
    monkeypatch,
) -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    initial_conditions = torch.tensor([[[0.25, 0.0]], [[0.75, 1.0]]], dtype=torch.double)
    strategy = MixedSpaceStrategy(
        bounds,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=None,
        batch_initial_conditions=initial_conditions,
    )
    captured = {}

    def fake_optimize_acqf_mixed(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.75, 1.0]], dtype=torch.double), torch.tensor(1.75)

    monkeypatch.setattr(
        "robotorchan.optim.backends.botorch.botorch_optimize_acqf_mixed",
        fake_optimize_acqf_mixed,
    )
    strategy.optimize(None)  # type: ignore[arg-type]

    assert captured["raw_samples"] is None
    assert captured["num_restarts"] == 2
    assert captured["batch_initial_conditions"] is initial_conditions


def test_mixed_space_strategy_runs_qbatch_nonlinear_with_ic_generator() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.0], [0.7, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.2 * train_X[:, 1:2]).sin()
    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=41),
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    def nonlinear_constraint(x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(0.64) - x[0].square()

    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((nonlinear_constraint, True),),
    )
    generated_q: list[int] = []

    def ic_generator(*, q, num_restarts, **kwargs):
        generated_q.append(q)
        initial = torch.full(
            (num_restarts, q, 2),
            0.5,
            dtype=bounds.dtype,
            device=bounds.device,
        )
        initial[..., 1] = 0.0
        return initial

    strategy = MixedSpaceStrategy(
        bounds,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=16,
        constraints=constraints,
        ic_generator=ic_generator,
    )

    result = strategy.optimize(acquisition, q=2)

    assert result.candidates.shape == torch.Size([2, 2])
    assert torch.isfinite(result.candidates).all()
    assert torch.isfinite(result.acquisition_value).all()
    assert torch.all(result.candidates[:, 0] <= 0.8 + 1e-6)
    assert set(result.candidates[:, 1].tolist()) <= {0.0, 1.0}
    assert generated_q == [1, 1, 1, 1]

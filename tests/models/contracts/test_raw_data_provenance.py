import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends import optimize_acqf_botorch


def _make_model() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 3.0)
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    return model, train_X, train_Y


def test_conditioning_preserves_constructor_raw_snapshot() -> None:
    model, train_X, train_Y = _make_model()
    new_X = torch.tensor([[0.33]], dtype=torch.double)
    new_Y = torch.sin(new_X * 3.0)

    model.eval()
    _ = model.posterior(new_X)
    conditioned = model.condition_on_observations(X=new_X, Y=new_Y)

    assert isinstance(conditioned, SingleTaskGP)
    assert conditioned.raw_data_names == ("train_X", "train_Y", "train_Yvar")
    assert torch.equal(conditioned.raw_train_X, train_X)
    assert torch.equal(conditioned.raw_train_Y, train_Y)
    assert conditioned.raw_train_Yvar is None
    assert conditioned.train_inputs[0].shape[-2] == train_X.shape[-2] + new_X.shape[-2]
    assert model.train_inputs[0].shape[-2] == train_X.shape[-2]


def test_fantasy_preserves_constructor_raw_snapshot() -> None:
    model, train_X, train_Y = _make_model()
    fantasy_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]), seed=1234)

    model.eval()
    fantasy_model = model.fantasize(X=fantasy_X, sampler=sampler)

    assert isinstance(fantasy_model, SingleTaskGP)
    assert fantasy_model.raw_data_names == ("train_X", "train_Y", "train_Yvar")
    assert torch.equal(fantasy_model.raw_train_X, train_X)
    assert torch.equal(fantasy_model.raw_train_Y, train_Y)
    assert fantasy_model.raw_train_Yvar is None
    assert fantasy_model.train_inputs[0].shape[-2] == train_X.shape[-2] + fantasy_X.shape[-2]


def test_fantasy_model_posterior_preserves_fantasy_batch_contract() -> None:
    model, train_X, _ = _make_model()
    fantasy_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    test_X = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]), seed=4321)

    model.eval()
    fantasy_model = model.fantasize(X=fantasy_X, sampler=sampler)
    posterior = fantasy_model.posterior(test_X)
    samples = posterior.rsample(torch.Size([4]))

    assert model.train_inputs[0].shape == train_X.shape
    assert fantasy_model.train_inputs[0].shape == torch.Size([2, 10, 1])
    assert posterior.mean.shape == torch.Size([2, 3, 1])
    assert posterior.variance.shape == torch.Size([2, 3, 1])
    assert samples.shape == torch.Size([4, 2, 3, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(samples).all()


def test_fantasy_model_runs_acquisition_and_qbatch_optimization() -> None:
    model, _, train_Y = _make_model()
    pending = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    fantasy_model = model.fantasize(
        X=pending,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=5678),
    )
    acquisition = qLogExpectedImprovement(
        model=fantasy_model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=6789),
    )
    X = torch.tensor([[[0.3], [0.7]]], dtype=torch.double, requires_grad=True)
    value = acquisition(X)
    gradient = torch.autograd.grad(value.sum(), X)[0]
    candidate, optimized_value = optimize_acqf_botorch(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_restarts=2,
        raw_samples=16,
    )

    assert value.shape == torch.Size([1])
    assert gradient.shape == X.shape
    assert torch.isfinite(gradient).all()
    assert candidate.shape == torch.Size([2, 1])
    assert optimized_value.numel() == 1
    assert candidate.dtype == torch.double
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(optimized_value).all()


def test_fantasy_batch_does_not_change_explicit_initial_condition_axes() -> None:
    model, _, train_Y = _make_model()
    pending = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    fantasy_model = model.fantasize(
        X=pending,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([2]), seed=7890),
    )
    acquisition = qLogExpectedImprovement(
        model=fantasy_model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=8901),
    )
    initial_conditions = torch.tensor(
        [
            [[0.2], [0.8]],
            [[0.3], [0.7]],
        ],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf_botorch(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_restarts=2,
        raw_samples=None,
        batch_initial_conditions=initial_conditions,
    )

    assert fantasy_model.train_inputs[0].shape[0] == 2
    assert initial_conditions.shape == torch.Size([2, 2, 1])
    assert candidate.shape == torch.Size([2, 1])
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()

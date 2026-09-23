"""Runtime contracts for heteroskedastic multi-fidelity regression."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler

from robotorchan.models import HeteroskedasticMultiFidelityGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    fidelity = torch.tensor([0.25, 0.5, 1.0, 0.25, 0.5, 1.0, 0.5, 1.0], dtype=torch.double)
    fidelity = fidelity.unsqueeze(-1)
    train_x = torch.cat((design, fidelity), dim=-1)
    train_y = torch.sin(design * 3.0) + 0.15 * fidelity
    return train_x, train_y


def test_heteroskedastic_multifidelity_raw_contract_and_posterior() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert torch.all(model.raw_train_Yvar > 0)

    model.eval()
    posterior = model.posterior(train_x[:2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()


def test_heteroskedastic_multifidelity_mc_acquisition_and_mll() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    assert model.make_mll().model is model
    model.eval()
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=IIDNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(train_x[:1].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_heteroskedastic_multifidelity_noise_model_retains_fidelity_contract() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    model.fit_heteroskedastic(iterations=1)
    assert model.noise_model is not None
    assert torch.equal(model.noise_model.raw_train_X, train_x)
    predicted_noise = model.predicted_noise(train_x[:2])
    assert predicted_noise.shape == torch.Size([2, 1])
    assert torch.isfinite(predicted_noise).all()
    assert torch.all(predicted_noise > 0)


def test_heteroskedastic_multifidelity_supports_native_mf_kg() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(train_x, train_y, data_fidelities=[-1])
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
    X[..., 1] = 0.5
    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_heteroskedastic_multifidelity_noise_model_uses_fidelity_coordinate() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[-1],
    ).fit_heteroskedastic(iterations=1)

    assert model.noise_model is not None
    assert model._data_fidelities == [-1]
    assert torch.equal(model.noise_model.raw_train_X[..., 1], train_x[..., 1])
    assert model.noise_model.covar_module is not None

    probe = torch.tensor([[0.5, 0.25], [0.5, 1.0]], dtype=torch.double)
    predicted_noise = model.predicted_noise(probe)
    assert predicted_noise.shape == torch.Size([2, 1])
    assert torch.isfinite(predicted_noise).all()
    assert torch.all(predicted_noise >= model.noise_floor)


def test_heteroskedastic_multifidelity_updates_response_noise_after_fit() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(
        train_x,
        train_y,
        data_fidelities=[-1],
    )
    initial_noise = model.likelihood.noise.detach().clone()
    model.fit_heteroskedastic(iterations=1)

    fitted_noise = model.likelihood.noise.detach()
    predicted_train_noise = model.predicted_noise(train_x).detach().squeeze(-1)

    assert torch.isfinite(fitted_noise).all()
    assert torch.all(fitted_noise >= model.noise_floor)
    torch.testing.assert_close(fitted_noise, predicted_train_noise)
    assert torch.equal(model.raw_train_Yvar, initial_noise.unsqueeze(-1))


def test_heteroskedastic_multifidelity_optimizes_candidate() -> None:
    train_x, train_y = _data()
    model = HeteroskedasticMultiFidelityGP(train_x, train_y, data_fidelities=[-1])
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

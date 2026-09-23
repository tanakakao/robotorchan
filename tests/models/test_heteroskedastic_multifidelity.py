"""Runtime contracts for heteroskedastic multi-fidelity regression."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
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

"""Runtime contracts for PCA multi-fidelity regression."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import IIDNormalSampler

from robotorchan.models import PCAMultiFidelityGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.rand(10, 5, dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, fidelity), dim=-1)
    train_y = (
        design[:, :2].sum(dim=-1, keepdim=True)
        + 0.2 * fidelity
    )
    return train_x, train_y


def test_pca_multifidelity_preserves_structural_fidelity() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.fidelity_dims == (5,)
    assert model.design_dims == (0, 1, 2, 3, 4)
    assert model.encoded_fidelity_dims == (2,)
    encoded = model._encode_inputs(train_x)
    assert encoded.shape[-1] == 3
    assert torch.equal(encoded[..., -1], train_x[..., -1])


def test_pca_multifidelity_posterior_sampling_and_mc_acquisition() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )
    model.eval()

    candidates = train_x[:2]
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


def test_pca_multifidelity_make_mll() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )

    mll = model.make_mll()
    assert mll.model is model

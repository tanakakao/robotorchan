"""Runtime contracts for PCA multi-fidelity regression."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling.normal import IIDNormalSampler

from robotorchan.models import PCAMultiFidelityGP, PLSMultiFidelityGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.rand(10, 5, dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, fidelity), dim=-1)
    train_y = design[:, :2].sum(dim=-1, keepdim=True) + 0.2 * fidelity
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


def test_pca_multifidelity_preserves_iteration_and_data_roles() -> None:
    torch.manual_seed(0)
    design = torch.rand(10, 4, dtype=torch.double)
    iteration = torch.linspace(0.1, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    data = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, iteration, data), dim=-1)
    train_y = design[:, :2].sum(dim=-1, keepdim=True) + 0.1 * iteration + 0.2 * data

    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        iteration_fidelity=-2,
        data_fidelities=[-1],
    )

    assert model.iteration_fidelity == 4
    assert model.data_fidelities == (5,)
    assert model.fidelity_dims == (4, 5)
    assert model.encoded_fidelity_dims == (2, 3)
    encoded = model._encode_inputs(train_x)
    assert torch.equal(encoded[..., 2], train_x[..., 4])
    assert torch.equal(encoded[..., 3], train_x[..., 5])


def test_pca_multifidelity_rejects_overlapping_fidelity_roles() -> None:
    train_x, train_y = _data()
    try:
        PCAMultiFidelityGP(
            train_x,
            train_y,
            n_components=2,
            iteration_fidelity=-1,
            data_fidelities=[-1],
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("Expected overlapping fidelity dimensions to be rejected.")


def test_pls_multifidelity_preserves_structural_fidelity() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PLSMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.fidelity_dims == (5,)
    assert model.design_dims == (0, 1, 2, 3, 4)
    encoded = model._encode_inputs(train_x)
    assert encoded.shape[-1] == 3
    assert torch.equal(encoded[..., -1], train_x[..., -1])

    model.eval()
    posterior = model.posterior(train_x[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()
    assert model.make_mll().model is model

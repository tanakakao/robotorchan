"""Mathematical invariants of the data-fidelity downsampling kernel."""

import torch

from robotorchan.models import MixedSingleTaskMultiFidelityGP


def test_data_fidelity_downsampling_has_rank_one_boundary_structure() -> None:
    """Verify the fidelity-only covariance and its high-fidelity boundary."""
    design = torch.tensor([[0.1], [0.3], [0.6], [0.8]], dtype=torch.float64)
    categories = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float64)
    fidelities = torch.tensor([[0.2], [0.5], [0.8], [1.0]], dtype=torch.float64)
    train_x = torch.cat((design, categories, fidelities), dim=-1)
    train_y = torch.tensor([[0.2], [0.4], [0.1], [0.5]], dtype=torch.float64)

    model = MixedSingleTaskMultiFidelityGP(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[1],
        data_fidelities=[2],
        outcome_transform=None,
    )
    downsampling = model.covar_module.kernels[1]
    with torch.no_grad():
        actual = downsampling(train_x).to_dense()

    # The native kernel includes a positive constant offset:
    # K_f(s, t) = offset + c * (1-s)^p * (1-t)^p.
    assert actual.shape == (4, 4)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, actual.T, atol=1e-10, rtol=1e-10)
    offset = actual[-1, -1]
    assert offset.item() > 0
    torch.testing.assert_close(actual[-1], offset.expand(4), atol=1e-10, rtol=0)
    torch.testing.assert_close(actual[:, -1], offset.expand(4), atol=1e-10, rtol=0)
    residual = actual - offset
    assert residual[0, 0].item() > 0

    # Independent rank-one identity after subtracting the constant offset.
    reference = residual[:, :1] @ residual[:1, :] / residual[0, 0]
    torch.testing.assert_close(residual, reference, atol=1e-10, rtol=1e-10)
    assert torch.linalg.eigvalsh(actual).min().item() >= -1e-10

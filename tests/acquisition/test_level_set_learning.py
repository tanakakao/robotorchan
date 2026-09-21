import pytest
import torch

from robotorchan.acquisition import BoundaryVariance, Straddle
from robotorchan.models import SingleTaskGP


def _model() -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y)


@pytest.mark.parametrize("acquisition_class", [Straddle, BoundaryVariance])
def test_level_set_acquisition_returns_finite_batch_values(acquisition_class: type) -> None:
    model = _model()
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    acquisition = acquisition_class(model, target=0.0)

    value = acquisition(X)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()


def test_straddle_validates_beta_and_q() -> None:
    model = _model()

    with pytest.raises(ValueError, match="non-negative"):
        Straddle(model, target=0.0, beta=-1.0)

    X = torch.tensor([[[0.25], [0.75]]], dtype=torch.double)
    with pytest.raises(ValueError, match="q=1"):
        Straddle(model, target=0.0)(X)


def test_straddle_score_matches_definition() -> None:
    model = _model()
    X = torch.tensor([[[0.4]]], dtype=torch.double)
    target = 0.1
    beta = 2.0

    posterior = model.posterior(X)
    expected = beta * posterior.variance.sqrt() - (posterior.mean - target).abs()
    actual = Straddle(model, target=target, beta=beta)(X)

    assert torch.allclose(actual, expected.squeeze(-1).squeeze(-1))

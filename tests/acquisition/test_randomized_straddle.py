import torch

from robotorchan.acquisition import RandomizedStraddle
from robotorchan.models import SingleTaskGP


def test_randomized_straddle_is_reproducible_with_generator() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = SingleTaskGP(train_X, train_Y)
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    value_a = RandomizedStraddle(model, target=0.0, generator=generator_a)(X)
    value_b = RandomizedStraddle(model, target=0.0, generator=generator_b)(X)

    assert torch.allclose(value_a, value_b)
    assert torch.isfinite(value_a).all()


def test_randomized_straddle_reuses_beta_until_resampled() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = SingleTaskGP(train_X, train_Y)
    X = torch.tensor([[[0.4]]], dtype=torch.double)
    acquisition = RandomizedStraddle(
        model,
        target=0.0,
        generator=torch.Generator().manual_seed(11),
    )

    first = acquisition(X)
    second = acquisition(X)
    assert torch.allclose(first, second)

    acquisition.resample()
    third = acquisition(X)
    assert not torch.allclose(first, third)


def test_randomized_straddle_preserves_double_beta_precision() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    acquisition = RandomizedStraddle(
        SingleTaskGP(train_X, train_Y),
        target=0.0,
        generator=torch.Generator().manual_seed(19),
    )

    acquisition(torch.tensor([[[0.35]]], dtype=torch.double))

    assert acquisition._random_beta is not None
    assert acquisition._random_beta.dtype == torch.double


class _FlaggedEnsembleModel:
    num_outputs = 1
    _is_ensemble = True

    def posterior(self, X: torch.Tensor):
        raise AssertionError("posterior should not be evaluated for flagged ensemble models")


def test_randomized_straddle_rejects_flagged_ensemble_before_posterior() -> None:
    acquisition = RandomizedStraddle(_FlaggedEnsembleModel(), target=0.0)

    try:
        acquisition(torch.tensor([[[0.2]]], dtype=torch.double))
    except ValueError as error:
        assert "ensemble posteriors" in str(error)
    else:
        raise AssertionError("Expected flagged-ensemble validation.")

import torch
from botorch.posteriors.ensemble import EnsemblePosterior

from robotorchan.acquisition import BoundaryVariance, PosteriorVariance, Straddle


class _EnsembleModel:
    num_outputs = 1

    def posterior(self, X: torch.Tensor) -> EnsemblePosterior:
        values = torch.stack((X[..., 0], X[..., 0] + 0.1), dim=-1).unsqueeze(-1)
        return EnsemblePosterior(values=values)


def test_posterior_variance_rejects_ensemble_posterior() -> None:
    acquisition = PosteriorVariance(_EnsembleModel())
    try:
        acquisition(torch.tensor([[[0.2]]], dtype=torch.double))
    except ValueError as error:
        assert "ensemble posteriors" in str(error)
    else:
        raise AssertionError("Expected ensemble-posterior validation.")


def test_straddle_rejects_ensemble_posterior() -> None:
    acquisition = Straddle(_EnsembleModel(), target=0.0)
    try:
        acquisition(torch.tensor([[[0.2]]], dtype=torch.double))
    except ValueError as error:
        assert "ensemble posteriors" in str(error)
    else:
        raise AssertionError("Expected ensemble-posterior validation.")


def test_boundary_variance_rejects_ensemble_posterior() -> None:
    acquisition = BoundaryVariance(_EnsembleModel(), target=0.0)
    try:
        acquisition(torch.tensor([[[0.2]]], dtype=torch.double))
    except ValueError as error:
        assert "ensemble posteriors" in str(error)
    else:
        raise AssertionError("Expected ensemble-posterior validation.")


class _FlaggedEnsembleModel:
    num_outputs = 1
    _is_ensemble = True

    def posterior(self, X: torch.Tensor):
        raise AssertionError("posterior should not be evaluated for flagged ensemble models")


def test_active_learning_rejects_flagged_ensemble_before_posterior() -> None:
    acquisitions = [
        PosteriorVariance(_FlaggedEnsembleModel()),
        Straddle(_FlaggedEnsembleModel(), target=0.0),
        BoundaryVariance(_FlaggedEnsembleModel(), target=0.0),
    ]
    X = torch.tensor([[[0.2]]], dtype=torch.double)

    for acquisition in acquisitions:
        try:
            acquisition(X)
        except ValueError as error:
            assert "ensemble posteriors" in str(error)
        else:
            raise AssertionError("Expected flagged-ensemble validation.")

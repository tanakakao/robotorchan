import torch
from botorch.posteriors.ensemble import EnsemblePosterior

from robotorchan.acquisition import qBoundaryVariance, qPosteriorVariance, qStraddle
from robotorchan.acquisition.registry import get_acquisition_registry_entry


class _EnsembleModel:
    num_outputs = 1

    def posterior(self, X: torch.Tensor) -> EnsemblePosterior:
        values = torch.stack((X[..., 0], X[..., 0] + 0.1), dim=-1).unsqueeze(-1)
        return EnsemblePosterior(values=values)


def test_posterior_variance_rejects_ensemble_posterior() -> None:
    acquisition = qPosteriorVariance(_EnsembleModel())
    try:
        acquisition(torch.tensor([[[0.2]]], dtype=torch.double))
    except ValueError as error:
        assert "ensemble posteriors" in str(error)
    else:
        raise AssertionError("Expected ensemble-posterior validation.")


def test_straddle_rejects_ensemble_posterior() -> None:
    acquisition = qStraddle(_EnsembleModel(), target=0.0)
    try:
        acquisition(torch.tensor([[[0.2]]], dtype=torch.double))
    except ValueError as error:
        assert "ensemble posteriors" in str(error)
    else:
        raise AssertionError("Expected ensemble-posterior validation.")


def test_boundary_variance_rejects_ensemble_posterior() -> None:
    acquisition = qBoundaryVariance(_EnsembleModel(), target=0.0)
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
        qPosteriorVariance(_FlaggedEnsembleModel()),
        qStraddle(_FlaggedEnsembleModel(), target=0.0),
        qBoundaryVariance(_FlaggedEnsembleModel(), target=0.0),
    ]
    X = torch.tensor([[[0.2]]], dtype=torch.double)

    for acquisition in acquisitions:
        try:
            acquisition(X)
        except ValueError as error:
            assert "ensemble posteriors" in str(error)
        else:
            raise AssertionError("Expected flagged-ensemble validation.")


def test_pointwise_active_learning_registry_rejects_ensemble_support() -> None:
    for name in (
        "qBoundaryVariance",
        "qPosteriorStd",
        "qPosteriorVariance",
        "qRandomizedStraddle",
        "qStraddle",
    ):
        entry = get_acquisition_registry_entry(name)
        assert entry.capabilities.supports_ensemble is False
        assert "ensemble posteriors are not supported" in entry.limitations

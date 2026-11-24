"""Constructor identity and runtime contracts for the acquisition factory."""

import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorMean
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

from robotorchan.acquisition import (
    create_acquisition,
    get_acquisition_class,
    list_acquisitions,
    qPosteriorVariance,
    qPredictiveEntropy,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models import SingleTaskGP
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


def test_registry_and_native_constructor_identity() -> None:
    names = list_acquisitions()
    assert names == tuple(sorted(set(names)))
    assert set(ACQUISITION_REGISTRY) <= set(names)
    for cls in (
        ExpectedImprovement,
        PosteriorMean,
        qLogExpectedImprovement,
        qMultiStepLookahead,
        AnalyticExpectedUtilityOfBestOption,
        qPosteriorVariance,
        qPredictiveEntropy,
    ):
        assert get_acquisition_class(cls.__name__) is cls
    for name in names:
        assert get_acquisition_class(name).__name__ == name


@pytest.mark.parametrize("name", ["unknown", "AcquisitionFunction", "make_qei_acquisition"])
def test_unknown_names(name: str) -> None:
    with pytest.raises(KeyError, match="Unknown acquisition"):
        get_acquisition_class(name)


def test_callable_arguments_are_preserved() -> None:
    marker = object()
    assert create_acquisition(lambda *, model: model, model=marker) is marker


def test_native_and_local_candidate_generation() -> None:
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(X * 5)
    model = SingleTaskGP(X, Y)
    bounds = torch.tensor([[0.0], [1.0]], dtype=X.dtype)
    for name, kwargs in (
        ("qLogExpectedImprovement", {"best_f": Y.max()}),
        ("qPosteriorVariance", {}),
    ):
        acq = create_acquisition(name, model=model, **kwargs)
        assert type(acq) is get_acquisition_class(name)
        assert acq.model is model
        candidate, value = optimize_acqf(
            acq, bounds, q=2, num_restarts=2, raw_samples=8, options={"maxiter": 5}
        )
        assert candidate.shape == (2, 1)
        assert torch.isfinite(value).all()


def test_classification_acquisition() -> None:
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    labels = (X.squeeze(-1) > 0.5).to(dtype=X.dtype)
    model = BinarySingleTaskGPClassifier(X, labels)
    acq = create_acquisition("qPredictiveEntropy", model=model)
    direct = create_acquisition(qPredictiveEntropy, model)
    batch = X[:4].reshape(2, 2, 1)
    torch.testing.assert_close(acq(batch), direct(batch))

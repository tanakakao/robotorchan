"""Phase 14 integration of calibration, conformal sets, decisions and AL."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    PredictiveEntropy,
    qPredictiveEntropy,
)
from robotorchan.models.classification import (
    CalibratedMulticlassClassifier,
    CalibratedOrdinalClassifier,
    MulticlassSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
    RandomForestMulticlassClassifier,
    SplitConformalClassifier,
    TemperatureScalingCalibrator,
    multiclass_cost_sensitive_prediction,
    multiclass_expected_decision_cost,
    ordinal_cost_matrix,
    ordinal_cost_sensitive_prediction,
)


@pytest.mark.parametrize(
    "base_class, calibrated_class",
    [
        (MulticlassSingleTaskGPClassifier, CalibratedMulticlassClassifier),
        (OrdinalSingleTaskGPClassifier, CalibratedOrdinalClassifier),
    ],
)
def test_gp_calibration_conformal_posterior_and_al(base_class, calibrated_class):
    torch.manual_seed(314)
    x = torch.linspace(0, 1, 18, dtype=torch.double).unsqueeze(-1)
    y = torch.bucketize(x[:, 0], torch.tensor([0.32, 0.68], dtype=torch.double))
    model = base_class(x, y, num_classes=3)
    model.fit(epochs=2, lr=0.005)
    pool = x[:6]
    calibrator = TemperatureScalingCalibrator(temperature=1.5).to(x)
    calibrated = calibrated_class(model, calibrator)
    with torch.no_grad():
        probabilities = calibrated.predict_proba(pool)
        samples = calibrated.sample_class_probabilities(
            pool, sample_shape=torch.Size([8])
        )
        posterior = calibrated.probability_posterior(pool, num_samples=8)
        score = PredictiveEntropy(calibrated)(pool.unsqueeze(-2))
        q_score = qPredictiveEntropy(calibrated)(pool.reshape(3, 2, 1))
        bald_score = BALD(calibrated, num_samples=8)(pool.unsqueeze(-2))
        mi = calibrated.mutual_information(pool, num_samples=8)
    assert probabilities.shape == (6, 3)
    assert samples.shape == (8, 6, 3)
    assert posterior.mean.shape == posterior.variance.shape == (6, 3)
    assert score.shape == (6,)
    assert q_score.shape == (3,)
    assert bald_score.shape == (6,)
    assert mi.shape == (6,)
    assert torch.isfinite(mi).all()
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(6, dtype=torch.double), atol=1e-5, rtol=1e-5
    )
    assert pool[score.argmax()].shape == (1,)

    # Conformal calibration requires an independent holdout set in practice.
    # Here a small separate slice exercises the existing API contract.
    conformal = SplitConformalClassifier(calibrated, alpha=0.2)
    with pytest.raises(RuntimeError, match="calibrate"):
        conformal.prediction_set(pool)
    conformal.calibrate(x[9:], y[9:])
    with torch.no_grad():
        sets = conformal.prediction_set(pool)
    assert sets.shape == (6, 3)
    assert sets.dtype == torch.bool


def test_temperature_scaling_fit_multiclass_validation():
    calibration_probs = torch.tensor(
        [[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7],
         [0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.15, 0.75, 0.1]],
        dtype=torch.double,
    )
    y = torch.tensor([0, 1, 2, 0, 2, 1])
    calibrator = TemperatureScalingCalibrator(temperature=1.5).to(calibration_probs)
    calibrator.fit(calibration_probs, y, max_iter=6)
    with torch.no_grad():
        result = calibrator(calibration_probs)
    assert result.shape == calibration_probs.shape
    torch.testing.assert_close(
        result.sum(-1), torch.ones(6, dtype=torch.double), atol=1e-6, rtol=1e-6
    )


def test_multiclass_decisions_cost_conventions_and_ordinal_loss():
    p = torch.tensor([[0.4, 0.35, 0.25], [0.1, 0.2, 0.7]], dtype=torch.double)
    cost = torch.tensor(
        [[0., 2., 5.], [1., 0., 2.], [4., 1., 0.]], dtype=torch.double
    )
    expected = p @ cost.T
    torch.testing.assert_close(multiclass_expected_decision_cost(p, cost_matrix=cost), expected)
    torch.testing.assert_close(
        multiclass_cost_sensitive_prediction(p, cost_matrix=cost),
        expected.argmin(-1),
    )
    for power in (1.0, 2.0):
        ordinal_cost = ordinal_cost_matrix(3, power=power, reference=p)
        torch.testing.assert_close(ordinal_cost, ordinal_cost.T)
        decision = ordinal_cost_sensitive_prediction(p, power=power)
        torch.testing.assert_close(
            decision,
            multiclass_cost_sensitive_prediction(p, cost_matrix=ordinal_cost),
        )
    with pytest.raises(ValueError, match="shape"):
        multiclass_cost_sensitive_prediction(p, cost_matrix=torch.eye(2, dtype=p.dtype))
    with pytest.raises(ValueError, match="nonnegative"):
        multiclass_cost_sensitive_prediction(p, cost_matrix=-torch.eye(3, dtype=p.dtype))


def test_deterministic_non_gp_calibration_does_not_invent_epistemic_posterior():
    pytest.importorskip("sklearn")
    x = torch.rand(18, 2, dtype=torch.double)
    y = torch.arange(18) % 3
    model = RandomForestMulticlassClassifier(
        x, y, num_classes=3, n_estimators=8, random_state=13
    )
    model.fit()
    calibrated = CalibratedMulticlassClassifier(
        model, TemperatureScalingCalibrator(temperature=1.2).to(x)
    )
    with torch.no_grad():
        assert calibrated.predict_proba(x[:4]).shape == (4, 3)
    with pytest.raises(NotImplementedError, match="epistemic"):
        calibrated.probability_posterior(x[:4], num_samples=8)


def test_calibration_preserves_gp_latent_distribution():
    x = torch.rand(9, 2, dtype=torch.double)
    y = torch.arange(9) % 3
    model = OrdinalSingleTaskGPClassifier(x, y, num_classes=3)
    calibrated = CalibratedOrdinalClassifier(
        model, TemperatureScalingCalibrator(temperature=1.3).to(x)
    )
    with torch.no_grad():
        mean_before = model.latent_posterior(x[:3]).mean
        mean_after = calibrated.latent_posterior(x[:3]).mean
    torch.testing.assert_close(mean_before, mean_after)

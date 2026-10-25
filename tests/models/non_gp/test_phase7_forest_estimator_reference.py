"""Estimator-level mathematical reference for the non-GP forest adapter."""

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from robotorchan.models import RandomForestSurrogate


def test_random_forest_posterior_matches_individual_estimator_predictions() -> None:
    """Check tree samples, sample moments, and unsupported noise explicitly."""
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.float64).unsqueeze(-1)
    train_y = torch.sin(5.0 * train_x)
    model = RandomForestSurrogate(train_x, train_y, n_estimators=7, random_state=17)
    model.fit()
    candidates = torch.tensor([[[0.1], [0.6]], [[0.35], [0.9]]], dtype=torch.float64)

    posterior = model.posterior(candidates)
    flat = candidates.reshape(-1, 1).numpy()
    individual = np.stack(
        [estimator.predict(flat) for estimator in model._forest.estimators_],
        axis=0,
    )
    reference_values = torch.as_tensor(individual.reshape(7, 2, 2, 1), dtype=torch.float64).movedim(
        0, 1
    )
    reference_mean = reference_values.mean(dim=1)
    reference_variance = reference_values.var(dim=1, unbiased=True)

    torch.testing.assert_close(posterior.values, reference_values, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(posterior.mean, reference_mean, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(posterior.variance, reference_variance, atol=1e-12, rtol=1e-12)
    with pytest.raises(NotImplementedError, match="observation noise"):
        model.posterior(candidates, observation_noise=True)

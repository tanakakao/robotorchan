"""NUTS fit contract for fully Bayesian SAAS model families."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.models import (
    MixedSaasFullyBayesianSingleTaskGP,
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
)


@pytest.mark.parametrize("kind", ["single", "multi", "mixed"])
def test_saas_fit_delegates_to_botorch_nuts(kind: str) -> None:
    pytest.importorskip("numpyro")
    pytest.importorskip("jax")
    X = torch.rand(8, 2, dtype=torch.double)
    Y = torch.sin(X[:, :1])
    if kind == "multi":
        task = (torch.arange(8) % 2).to(X).unsqueeze(-1)
        model = SaasFullyBayesianMultiTaskGP(torch.cat([X, task], dim=-1), Y, task_feature=2)
    elif kind == "mixed":
        X[:, 1] = (torch.arange(8) % 2).to(X)
        model = MixedSaasFullyBayesianSingleTaskGP(X, Y, cat_dims=[1])
    else:
        model = SaasFullyBayesianSingleTaskGP(X, Y)

    assert model.supports_fit is True
    assert model.supports_mll is False
    with patch(
        "robotorchan.models.high_dimensional.fully_bayesian.fit_fully_bayesian_model_nuts"
    ) as fit:
        assert (
            model.fit(
                warmup_steps=4,
                num_samples=4,
                thinning=2,
                disable_progbar=True,
            )
            is None
        )

    fit.assert_called_once_with(
        model,
        warmup_steps=4,
        num_samples=4,
        thinning=2,
        disable_progbar=True,
    )

from __future__ import annotations

import inspect

from robotorchan import models


def test_mixed_saas_model_is_not_exposed_without_sound_semantics() -> None:
    assert not hasattr(models, "MixedSaasSingleTaskGP")
    assert "MixedSaasSingleTaskGP" not in models.__all__


def test_existing_saas_and_mixed_models_remain_distinct_public_models() -> None:
    assert hasattr(models, "MixedSingleTaskGP")
    assert hasattr(models, "AdditiveMapSaasSingleTaskGP")
    assert hasattr(models, "EnsembleMapSaasSingleTaskGP")

    mixed_signature = inspect.signature(models.MixedSingleTaskGP)
    additive_signature = inspect.signature(models.AdditiveMapSaasSingleTaskGP)
    ensemble_signature = inspect.signature(models.EnsembleMapSaasSingleTaskGP)

    assert "cat_dims" in mixed_signature.parameters
    assert "cat_dims" not in additive_signature.parameters
    assert "cat_dims" not in ensemble_signature.parameters

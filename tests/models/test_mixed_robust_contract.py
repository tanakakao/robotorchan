from __future__ import annotations

from robotorchan import models


def test_mixed_robust_model_is_not_exposed_without_verified_semantics() -> None:
    assert not hasattr(models, "MixedRobustRelevancePursuitSingleTaskGP")
    assert "MixedRobustRelevancePursuitSingleTaskGP" not in models.__all__


def test_existing_mixed_and_robust_models_remain_distinct() -> None:
    assert hasattr(models, "MixedSingleTaskGP")
    assert hasattr(models, "RobustRelevancePursuitSingleTaskGP")
    assert not issubclass(
        models.RobustRelevancePursuitSingleTaskGP,
        models.MixedSingleTaskGP,
    )

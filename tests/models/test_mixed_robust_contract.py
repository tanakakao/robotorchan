from __future__ import annotations

from robotorchan import models


def test_mixed_robust_model_is_exposed_after_semantic_verification() -> None:
    assert hasattr(models, "MixedRobustRelevancePursuitSingleTaskGP")
    assert "MixedRobustRelevancePursuitSingleTaskGP" in models.__all__


def test_standard_robust_model_remains_distinct_from_mixed_single_task() -> None:
    assert hasattr(models, "MixedSingleTaskGP")
    assert hasattr(models, "RobustRelevancePursuitSingleTaskGP")
    assert not issubclass(
        models.RobustRelevancePursuitSingleTaskGP,
        models.MixedSingleTaskGP,
    )

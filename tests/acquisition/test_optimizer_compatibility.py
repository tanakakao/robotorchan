from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)


def test_standard_acquisition_accepts_mixed_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qLogExpectedImprovement",
        CandidateOptimizer.MIXED,
    )

    assert result.compatible
    assert result.reasons == ()


def test_qkg_rejects_standard_mixed_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qKnowledgeGradient",
        CandidateOptimizer.MIXED,
    )

    assert not result.compatible
    assert "one-shot augmented batch" in result.reasons[0]


def test_qmfkg_rejects_standard_mixed_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qMultiFidelityKnowledgeGradient",
        CandidateOptimizer.MIXED,
    )

    assert not result.compatible
    assert "one-shot augmented batch" in result.reasons[0]


def test_qmfkg_accepts_continuous_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qMultiFidelityKnowledgeGradient",
        CandidateOptimizer.CONTINUOUS,
    )

    assert result.compatible
    assert result.reasons == ()


def test_qkg_accepts_mixed_one_shot_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qKnowledgeGradient",
        CandidateOptimizer.MIXED_ONE_SHOT,
    )

    assert result.compatible
    assert result.reasons == ()


def test_qmfkg_accepts_mixed_one_shot_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qMultiFidelityKnowledgeGradient",
        CandidateOptimizer.MIXED_ONE_SHOT,
    )

    assert result.compatible
    assert result.reasons == ()


def test_standard_acquisition_rejects_mixed_one_shot_optimizer() -> None:
    result = check_acquisition_optimizer_compatibility(
        "qLogExpectedImprovement",
        CandidateOptimizer.MIXED_ONE_SHOT,
    )

    assert not result.compatible
    assert result.reasons == ("mixed one-shot optimization requires a one-shot acquisition",)

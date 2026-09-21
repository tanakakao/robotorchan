from robotorchan import acquisition


def test_acquisition_public_api_is_intentionally_small() -> None:
    assert acquisition.__all__ == [
        "BoundaryVariance",
        "ExpectedPredictiveInformationGain",
        "PosteriorStd",
        "PosteriorVariance",
        "RandomizedStraddle",
        "Straddle",
        "make_non_gp_acquisition",
        "select_thompson_candidates",
        "validate_non_gp_acquisition",
    ]

from robotorchan import acquisition


def test_acquisition_public_api_is_intentionally_small() -> None:
    assert acquisition.__all__ == [
        "AcquisitionCapabilities",
        "AcquisitionPurpose",
        "AcquisitionRegistryEntry",
        "BoundaryVariance",
        "ExpectedPredictiveInformationGain",
        "PosteriorRequirement",
        "PosteriorStd",
        "PosteriorVariance",
        "RandomizedStraddle",
        "Straddle",
        "get_acquisition_registry_entry",
        "make_non_gp_acquisition",
        "select_thompson_candidates",
        "validate_non_gp_acquisition",
    ]

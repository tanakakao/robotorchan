from robotorchan import acquisition


def test_acquisition_public_api_is_intentionally_small() -> None:
    assert acquisition.__all__ == [
        "BALD",
        "AcquisitionCapabilities",
        "AcquisitionCompositionPlan",
        "AcquisitionPurpose",
        "AcquisitionRegistryEntry",
        "BoundaryVariance",
        "ExpectedPredictiveInformationGain",
        "FeasibilityBinding",
        "LatentStraddle",
        "MarginUncertainty",
        "ObjectiveBinding",
        "PosteriorRequirement",
        "PosteriorStd",
        "PosteriorVariance",
        "PredictiveEntropy",
        "ProbabilityVariance",
        "RandomizedStraddle",
        "Straddle",
        "get_acquisition_registry_entry",
        "make_non_gp_acquisition",
        "resolve_acquisition_composition",
        "select_thompson_candidates",
        "validate_non_gp_acquisition",
    ]

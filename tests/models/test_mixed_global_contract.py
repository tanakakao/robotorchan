import importlib

MODELS = importlib.import_module("robotorchan.models")

EXPECTED_MIXED_MODELS = {
    "MixedAdditiveMapSaasSingleTaskGP",
    "MixedAutoEncoderGP",
    "MixedContaminatedSingleTaskGP",
    "MixedEnsembleMapSaasSingleTaskGP",
    "MixedHeterogeneousMTGP",
    "MixedHierarchicalConditionalKernelGP",
    "MixedHierarchicalConditionalKernelMultiTaskGP",
    "MixedHigherOrderGP",
    "MixedHybridAutoEncoderGP",
    "MixedJointEncoderGP",
    "MixedJointVAEGP",
    "MixedKroneckerMultiTaskGP",
    "MixedLCEMGP",
    "MixedLatentKroneckerGP",
    "MixedMultiTaskGP",
    "MixedOrthogonalAdditiveGP",
    "MixedPCAGP",
    "MixedPLSGP",
    "MixedRandomProjectionGP",
    "MixedReducedGP",
    "MixedReducedKroneckerMultiTaskGP",
    "MixedReducedMultiTaskGP",
    "MixedRobustRelevancePursuitSingleTaskGP",
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "MixedSingleTaskGP",
    "MixedSingleTaskMultiFidelityGP",
    "MixedSingleTaskVariationalGP",
    "MixedStudentTSingleTaskGP",
    "MixedSupervisedAutoEncoderGP",
    "MixedSupervisedVAEGP",
    "MixedVAEGP",
}

INTENTIONALLY_ABSENT = {
    "MixedALEBOGP",
    "MixedLCEAGP",
    "MixedModelListGP",
    "MixedPairwiseGP",
    "MixedSACGP",
}


def test_global_mixed_public_inventory_is_explicit() -> None:
    exported_mixed = {name for name in MODELS.__all__ if name.startswith("Mixed")}
    assert exported_mixed == EXPECTED_MIXED_MODELS


def test_global_mixed_boundaries_remain_unexposed() -> None:
    assert INTENTIONALLY_ABSENT.isdisjoint(MODELS.__all__)
    assert all(not hasattr(MODELS, name) for name in INTENTIONALLY_ABSENT)


def test_every_public_mixed_model_has_training_contract() -> None:
    for name in EXPECTED_MIXED_MODELS:
        model = getattr(MODELS, name)
        assert isinstance(model, type)
        assert hasattr(model, "supports_mll")
        assert hasattr(model, "make_mll")

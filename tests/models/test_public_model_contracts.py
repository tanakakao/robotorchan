import importlib  # noqa: I001


MODELS = importlib.import_module("robotorchan.models")

PUBLIC_MODEL_NAMES = {
    "ALEBOGP",
    "AdditiveMapSaasSingleTaskGP",
    "AutoEncoderGP",
    "AutoEncoderKroneckerMultiTaskGP",
    "AutoEncoderMultiTaskGP",
    "EnsembleMapSaasSingleTaskGP",
    "HeterogeneousMTGP",
    "HeteroskedasticSingleTaskGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "HybridAutoEncoderGP",
    "HybridAutoEncoderKroneckerMultiTaskGP",
    "HybridAutoEncoderMultiTaskGP",
    "JointEncoderGP",
    "JointEncoderKroneckerMultiTaskGP",
    "JointEncoderMultiTaskGP",
    "JointVAEGP",
    "JointVAEKroneckerMultiTaskGP",
    "JointVAEMultiTaskGP",
    "KroneckerMultiTaskGP",
    "LCEAGP",
    "LCEMGP",
    "LatentKroneckerGP",
    "MixedAdditiveMapSaasSingleTaskGP",
    "MixedAutoEncoderGP",
    "MixedEnsembleMapSaasSingleTaskGP",
    "MixedHybridAutoEncoderGP",
    "MixedOrthogonalAdditiveGP",
    "MixedLatentKroneckerGP",
    "MixedLCEMGP",
    "MixedHigherOrderGP",
    "MixedHierarchicalConditionalKernelMultiTaskGP",
    "MixedHierarchicalConditionalKernelGP",
    "MixedHeterogeneousMTGP",
    "MixedJointEncoderGP",
    "MixedJointVAEGP",
    "MixedKroneckerMultiTaskGP",
    "MixedMultiTaskGP",
    "MixedPCAGP",
    "MixedPLSGP",
    "MixedRandomProjectionGP",
    "MixedRobustRelevancePursuitSingleTaskGP",
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "MixedReducedGP",
    "MixedReducedKroneckerMultiTaskGP",
    "MixedReducedMultiTaskGP",
    "MixedSupervisedAutoEncoderGP",
    "MixedSupervisedVAEGP",
    "MixedSingleTaskGP",
    "MixedSingleTaskMultiFidelityGP",
    "MixedSingleTaskVariationalGP",
    "MixedVAEGP",
    "ModelListGP",
    "MultiTaskGP",
    "OrthogonalAdditiveGP",
    "OutputPCAGP",
    "OutputPLSGP",
    "PCAGP",
    "PCAKroneckerMultiTaskGP",
    "PCAMultiTaskGP",
    "PLSGP",
    "PLSKroneckerMultiTaskGP",
    "PLSMultiTaskGP",
    "PairwiseGP",
    "RandomProjectionGP",
    "RandomProjectionKroneckerMultiTaskGP",
    "RandomProjectionMultiTaskGP",
    "ReducedGP",
    "ReducedKroneckerMultiTaskGP",
    "ReducedMultiTaskGP",
    "RobustRelevancePursuitSingleTaskGP",
    "SACGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "SupervisedAutoEncoderGP",
    "SupervisedAutoEncoderKroneckerMultiTaskGP",
    "SupervisedAutoEncoderMultiTaskGP",
    "SupervisedVAEGP",
    "SupervisedVAEKroneckerMultiTaskGP",
    "SupervisedVAEMultiTaskGP",
    "VAEGP",
    "VAEKroneckerMultiTaskGP",
    "VAEMultiTaskGP",
}

NON_MLL_MODELS = {
    "MixedSaasFullyBayesianMultiTaskGP",
    "MixedSaasFullyBayesianSingleTaskGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
}


def test_public_model_exports_are_complete_and_explicit() -> None:
    exported = set(MODELS.__all__)

    assert exported == PUBLIC_MODEL_NAMES | {"UnsupportedModelOperationError"}


def test_public_model_names_resolve_to_matching_classes() -> None:
    for name in PUBLIC_MODEL_NAMES:
        model_class = getattr(MODELS, name)

        assert isinstance(model_class, type)
        assert model_class.__name__ == name


def test_all_public_models_expose_training_capability_contract() -> None:
    for name in PUBLIC_MODEL_NAMES:
        model_class = getattr(MODELS, name)

        assert hasattr(model_class, "supports_mll")
        assert hasattr(model_class, "make_mll")
        assert model_class.supports_mll is (name not in NON_MLL_MODELS)

import importlib  # noqa: I001


MODELS = importlib.import_module("robotorchan.models")

PUBLIC_MODEL_NAMES = {
    "AdditiveMapSaasSingleTaskGP",
    "AutoEncoderGP",
    "EnsembleMapSaasSingleTaskGP",
    "HeterogeneousMTGP",
    "HierarchicalConditionalKernelGP",
    "HierarchicalConditionalKernelMultiTaskGP",
    "HigherOrderGP",
    "HybridAutoEncoderGP",
    "JointEncoderGP",
    "KroneckerMultiTaskGP",
    "LCEAGP",
    "LCEMGP",
    "LatentKroneckerGP",
    "MixedSingleTaskGP",
    "ModelListGP",
    "MultiTaskGP",
    "OrthogonalAdditiveGP",
    "OutputPCAGP",
    "OutputPLSGP",
    "PCAGP",
    "PLSGP",
    "PairwiseGP",
    "RandomProjectionGP",
    "ReducedGP",
    "RobustRelevancePursuitSingleTaskGP",
    "SACGP",
    "SaasFullyBayesianMultiTaskGP",
    "SaasFullyBayesianSingleTaskGP",
    "SingleTaskGP",
    "SingleTaskMultiFidelityGP",
    "SingleTaskVariationalGP",
    "SupervisedAutoEncoderGP",
    "SupervisedVAEGP",
    "VAEGP",
}

NON_MLL_MODELS = {
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

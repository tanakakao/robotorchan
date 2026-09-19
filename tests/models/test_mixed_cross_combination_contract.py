import importlib

from robotorchan.models import (
    MixedAutoEncoderGP,
    MixedHierarchicalConditionalKernelGP,
    MixedJointEncoderGP,
    MixedPCAGP,
    MixedSingleTaskMultiFidelityGP,
    MixedSingleTaskVariationalGP,
    ModelListGP,
)

MODELS = importlib.import_module("robotorchan.models")


def test_cross_combination_public_surface_is_compositional() -> None:
    supported = {
        MixedAutoEncoderGP,
        MixedHierarchicalConditionalKernelGP,
        MixedJointEncoderGP,
        MixedPCAGP,
        MixedSingleTaskMultiFidelityGP,
        MixedSingleTaskVariationalGP,
    }
    assert all(model.__name__ in MODELS.__all__ for model in supported)


def test_model_list_composes_mixed_models_without_redundant_wrapper() -> None:
    assert ModelListGP.__name__ in MODELS.__all__
    assert "MixedModelListGP" not in MODELS.__all__
    assert not hasattr(MODELS, "MixedModelListGP")


def test_specialized_unsupported_combinations_are_not_advertised() -> None:
    for name in ("MixedALEBOGP", "MixedPairwiseGP", "MixedSACGP", "MixedLCEAGP"):
        assert name not in MODELS.__all__
        assert not hasattr(MODELS, name)

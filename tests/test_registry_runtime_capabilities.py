"""Regression tests for explicit runtime capability metadata."""

from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from torch import Size

from robotorchan.acquisition.samplers import make_model_sampler
from robotorchan.models.capabilities import InputType, PosteriorSamplingType
from robotorchan.models.registry import MODEL_REGISTRY


def test_runtime_capabilities_are_not_blanket_enabled() -> None:
    random_forest = MODEL_REGISTRY["RandomForestSurrogate"].capabilities
    variational = MODEL_REGISTRY["MixedSingleTaskVariationalGP"].capabilities

    assert random_forest.supports_posterior_samples
    assert not random_forest.supports_fantasize
    assert variational.supports_posterior_samples
    assert not variational.supports_fantasize


def test_runtime_validated_exact_models_keep_required_capabilities() -> None:
    for name in (
        "SingleTaskGP",
        "MixedSingleTaskGP",
        "SingleTaskMultiFidelityGP",
        "MixedSingleTaskMultiFidelityGP",
    ):
        capabilities = MODEL_REGISTRY[name].capabilities
        assert capabilities.supports_posterior_samples
        assert capabilities.supports_fantasize


def test_kronecker_models_do_not_advertise_unsupported_fantasize() -> None:
    for name in (
        "KroneckerMultiTaskGP",
        "MixedKroneckerMultiTaskGP",
        "SpectralMixtureKroneckerMultiTaskGP",
        "MixedSpectralMixtureKroneckerMultiTaskGP",
        "InfiniteWidthBNNKroneckerMultiTaskGP",
        "MixedInfiniteWidthBNNKroneckerMultiTaskGP",
    ):
        assert not MODEL_REGISTRY[name].capabilities.supports_fantasize


def test_mixed_reduced_models_keep_mixed_input_metadata() -> None:
    for name in (
        "MixedReducedGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
    ):
        assert MODEL_REGISTRY[name].capabilities.input_type is InputType.MIXED


def test_runtime_validated_reduced_models_support_fantasize() -> None:
    for name in (
        "ReducedGP",
        "PCAGP",
        "PLSGP",
        "RandomProjectionGP",
        "MixedReducedGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
    ):
        assert MODEL_REGISTRY[name].capabilities.supports_fantasize


def test_posterior_sampling_type_distinguishes_gaussian_and_ensemble_models() -> None:
    assert (
        MODEL_REGISTRY["SingleTaskGP"].capabilities.posterior_sampling_type
        is PosteriorSamplingType.GAUSSIAN
    )
    assert (
        MODEL_REGISTRY["RandomForestSurrogate"].capabilities.posterior_sampling_type
        is PosteriorSamplingType.ENSEMBLE
    )


def test_sampling_type_is_none_when_posterior_sampling_is_disabled() -> None:
    for entry in MODEL_REGISTRY.values():
        capabilities = entry.capabilities
        if not capabilities.supports_posterior_samples:
            assert capabilities.posterior_sampling_type is PosteriorSamplingType.NONE


def test_map_saas_ensembles_use_gaussian_sampling_metadata() -> None:
    for name in ("EnsembleMapSaasSingleTaskGP", "MixedEnsembleMapSaasSingleTaskGP"):
        capabilities = MODEL_REGISTRY[name].capabilities
        assert capabilities.ensemble_posterior
        assert capabilities.posterior_sampling_type is PosteriorSamplingType.GAUSSIAN


def test_empirical_tree_ensembles_use_index_sampling_metadata() -> None:
    for name in (
        "RandomForestSurrogate",
        "ExtraTreesSurrogate",
        "GradientBoostingSurrogate",
        "HistGradientBoostingSurrogate",
    ):
        capabilities = MODEL_REGISTRY[name].capabilities
        assert capabilities.ensemble_posterior
        assert capabilities.posterior_sampling_type is PosteriorSamplingType.ENSEMBLE


def test_posterior_sampling_support_and_type_are_bidirectionally_consistent() -> None:
    for entry in MODEL_REGISTRY.values():
        capabilities = entry.capabilities
        if capabilities.supports_posterior_samples:
            assert capabilities.posterior_sampling_type is not PosteriorSamplingType.NONE
        else:
            assert capabilities.posterior_sampling_type is PosteriorSamplingType.NONE


def test_ensemble_sampling_type_is_reserved_for_empirical_ensemble_posteriors() -> None:
    ensemble_sampling_models = {
        name
        for name, entry in MODEL_REGISTRY.items()
        if entry.capabilities.posterior_sampling_type is PosteriorSamplingType.ENSEMBLE
    }
    assert ensemble_sampling_models == {
        "RandomForestSurrogate",
        "ExtraTreesSurrogate",
        "GradientBoostingSurrogate",
        "HistGradientBoostingSurrogate",
    }


def test_sampling_type_always_matches_sampling_support() -> None:
    for entry in MODEL_REGISTRY.values():
        capabilities = entry.capabilities
        has_sampling_type = capabilities.posterior_sampling_type is not PosteriorSamplingType.NONE
        assert capabilities.supports_posterior_samples is has_sampling_type


def test_sampler_factory_matches_registered_sampling_type() -> None:
    expected_sampler_type = {
        PosteriorSamplingType.GAUSSIAN: SobolQMCNormalSampler,
        PosteriorSamplingType.ENSEMBLE: IndexSampler,
        PosteriorSamplingType.STOCHASTIC: StochasticSampler,
    }

    for name, entry in MODEL_REGISTRY.items():
        sampling_type = entry.capabilities.posterior_sampling_type
        if sampling_type is PosteriorSamplingType.NONE:
            continue

        sampler = make_model_sampler(name, Size([4]))
        assert isinstance(sampler, expected_sampler_type[sampling_type])

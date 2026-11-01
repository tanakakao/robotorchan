"""Fit API contract for standard exact-GP wrappers."""

from robotorchan.models.standard.multi_fidelity import (
    MixedSingleTaskMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from robotorchan.models.standard.multitask import (
    KroneckerMultiTaskGP,
    MixedKroneckerMultiTaskGP,
    MixedMultiTaskGP,
    MultiTaskGP,
)


def test_standard_exact_gp_fit_capabilities() -> None:
    for model_class in (
        KroneckerMultiTaskGP,
        MixedKroneckerMultiTaskGP,
        MixedMultiTaskGP,
        MixedSingleTaskMultiFidelityGP,
        MultiTaskGP,
        SingleTaskMultiFidelityGP,
    ):
        assert model_class.supports_fit is True
        assert callable(model_class.fit)
        assert callable(model_class.make_mll)

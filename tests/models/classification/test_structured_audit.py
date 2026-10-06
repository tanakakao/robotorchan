"""Tests for the structured regression-to-classification family audit."""

from robotorchan.models.classification.structured_audit import (
    ClassificationStructuredDecision,
    classification_structured_audit,
)


def test_structured_audit_covers_regression_structured_families() -> None:
    entries = classification_structured_audit()
    names = {entry.regression_family for entry in entries}

    assert names == {
        "OrthogonalAdditiveGP",
        "SACGP/LCEAGP",
        "LCEMGP/HeterogeneousMTGP",
        "HierarchicalConditionalKernelGP",
        "HigherOrderGP",
        "LatentKroneckerGP",
    }


def test_gaussian_structured_outputs_are_not_claimed_as_classifiers() -> None:
    entries = {
        entry.regression_family: entry.decision
        for entry in classification_structured_audit()
    }

    assert entries["HigherOrderGP"] is ClassificationStructuredDecision.UNSUPPORTED
    assert entries["LatentKroneckerGP"] is ClassificationStructuredDecision.UNSUPPORTED


def test_input_structure_families_are_deferred_for_native_variational_models() -> None:
    entries = classification_structured_audit()

    for entry in entries:
        if entry.regression_family in {
            "OrthogonalAdditiveGP",
            "SACGP/LCEAGP",
            "LCEMGP/HeterogeneousMTGP",
            "HierarchicalConditionalKernelGP",
        }:
            assert entry.decision is ClassificationStructuredDecision.DEFER

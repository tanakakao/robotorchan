"""Contracts for reproducible public acquisition API auditing."""

import json
import runpy
from pathlib import Path

import pytest
import torch

from robotorchan.acquisition import PosteriorVariance, PredictiveEntropy
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models import SingleTaskGP
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)

_AUDIT = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "scripts" / "audit_acquisition_api.py")
)
build_inventory = _AUDIT["build_inventory"]


def test_all_registered_local_acquisitions_are_public_and_indexed() -> None:
    report = build_inventory()
    assert report["verification"] == "static_only"
    assert report["gaps"]["unregistered_public_acquisition_classes"] == []
    assert report["gaps"]["registered_local_acquisitions_not_exported"] == []
    assert json.loads(json.dumps(report)) == report


def test_native_botorch_acquisitions_are_metadata_only() -> None:
    report = build_inventory()
    native = report["native_botorch_metadata"]
    assert "qLogExpectedImprovement" in native
    assert "qKnowledgeGradient" in native
    assert "qLogExpectedImprovement" not in report["exports"]
    assert native["qKnowledgeGradient"]["one_shot"]
    assert native["qKnowledgeGradient"]["requires_fantasize"]
    assert native["qLogExpectedHypervolumeImprovement"]["supports_multi_objective"]


def test_public_custom_acquisition_signatures_preserve_specific_parameters() -> None:
    report = build_inventory()["exports"]
    assert report["PosteriorVariance"]["kind"] == "local_acquisition_class"
    assert "output_index" in report["PosteriorVariance"]["parameters"]
    assert "target" in report["Straddle"]["parameters"]
    assert "num_samples" in report["BALD"]["parameters"]
    assert report["make_qei_acquisition"]["kind"] == "composition_or_support_helper"


def test_one_shot_compatibility_is_kept_separate_from_acquisition_contract() -> None:
    name = "qKnowledgeGradient"
    assert name in ACQUISITION_REGISTRY
    standard = check_acquisition_optimizer_compatibility(name, CandidateOptimizer.MIXED)
    dedicated = check_acquisition_optimizer_compatibility(name, CandidateOptimizer.MIXED_ONE_SHOT)
    assert not standard.compatible
    assert dedicated.compatible


def test_regression_acquisition_q_one_batch_shape() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X)
    model = SingleTaskGP(train_X, train_Y)
    acquisition = PosteriorVariance(model)
    X = train_X[:3].unsqueeze(1)
    score = acquisition(X)
    assert score.shape == (3,)
    assert torch.isfinite(score).all()
    with pytest.raises(ValueError, match="q=1"):
        acquisition(train_X[:4].reshape(2, 2, 1))


def test_classification_acquisition_predictive_semantics_and_q() -> None:
    train_X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    labels = (train_X.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_X, labels)
    acquisition = PredictiveEntropy(model)
    score = acquisition(train_X[:2].unsqueeze(1))
    assert score.shape == (2,)
    assert torch.isfinite(score).all()
    with pytest.raises(ValueError, match="q=1"):
        acquisition(train_X[:4].reshape(2, 2, 1))

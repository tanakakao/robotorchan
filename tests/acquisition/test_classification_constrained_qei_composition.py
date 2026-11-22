"""Regression qEI with classifier class-set feasibility contracts."""

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition import make_classification_constrained_qei_acquisition
from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import ClassificationConstraint, ProblemSemantics, RegressionObjective


def _problem(*, threshold=None):
    train_x = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    model = HeterogeneousModel(
        SingleTaskGP(train_x, torch.sin(train_x * 3)),
        BinarySingleTaskGPClassifier(train_x, torch.tensor([0, 0, 1, 1, 1, 0])),
        output_names=["score", "status"],
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("score"),),
        constraints=(
            ClassificationConstraint(
                "status", feasible_classes=(0, 1), probability_threshold=threshold
            ),
        ),
    )
    return model, semantics


def test_class_set_qei_weights_native_qei_without_joint_posterior():
    model, semantics = _problem()
    acquisition = make_classification_constrained_qei_acquisition(
        model,
        semantics,
        best_f=0.0,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=19),
    )
    assert isinstance(acquisition, FeasibilityWeightedAcquisition)
    X = torch.tensor([[[0.3]], [[0.7]]], dtype=torch.double)
    torch.testing.assert_close(
        acquisition(X), acquisition.objective_acquisition(X), rtol=1e-5, atol=1e-6
    )


def test_classification_qei_rejects_probability_threshold_as_hard_constraint():
    model, semantics = _problem(threshold=0.8)
    with pytest.raises(TypeError, match="probability thresholds"):
        make_classification_constrained_qei_acquisition(model, semantics, best_f=0.0)

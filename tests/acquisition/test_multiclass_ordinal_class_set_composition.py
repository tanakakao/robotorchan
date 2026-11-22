"""Targeted audit: multiclass / ordinal feasibility with native qEI and qEHVI.

The notebook exercises fitted end-to-end candidate generation. These tests
verify the acquisition algebra, heterogeneous output ownership and t/q shapes
for both classifier families without adding slow model-fitting tests.
"""

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from robotorchan.acquisition import (
    make_classification_constrained_qei_acquisition,
    make_constrained_qehvi_acquisition,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
    ProblemSemantics,
    RegressionObjective,
)


@pytest.mark.parametrize("classifier_kind", ["multiclass", "ordinal"])
@pytest.mark.parametrize("objective_kind", ["single", "multi"])
@pytest.mark.parametrize("q", [1, 2])
def test_class_set_weighting_preserves_native_q_acquisition(
    classifier_kind: str, objective_kind: str, q: int
) -> None:
    """All four model combinations produce the expected q-batch weighted value."""
    train_x = torch.linspace(0.05, 0.95, 9, dtype=torch.double).unsqueeze(-1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    classifier_type = (
        MulticlassSingleTaskGPClassifier
        if classifier_kind == "multiclass"
        else OrdinalSingleTaskGPClassifier
    )
    classifier = classifier_type(train_x, labels, num_classes=3, inducing_points=4)
    classifier.eval()

    y1 = 1.0 - (train_x - 0.30).square()
    y2 = 1.0 - (train_x - 0.75).square()
    train_y = y1 if objective_kind == "single" else torch.cat((y1, y2), dim=-1)
    regression = SingleTaskGP(train_x, train_y)
    regression.eval()
    output_names = (
        ["score", "quality"]
        if objective_kind == "single"
        else ["objective_1", "objective_2", "quality"]
    )
    model = HeterogeneousModel(regression, classifier, output_names=output_names)
    assert model.entry_num_outputs == (train_y.shape[-1], 1)
    assert model.output_classification_metadata("quality").num_classes == 3

    constraint = ClassificationConstraint("quality", feasible_classes=(1, 2))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=25)
    if objective_kind == "single":
        semantics = ProblemSemantics(
            objectives=(RegressionObjective("score"),), constraints=(constraint,)
        )
        acquisition = make_classification_constrained_qei_acquisition(
            model,
            semantics,
            best_f=y1[labels >= 1].max().item(),
            sampler=sampler,
        )
    else:
        semantics = ProblemSemantics(
            objectives=(
                RegressionObjective("objective_1"),
                RegressionObjective("objective_2"),
            ),
            constraints=(constraint,),
        )
        ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y[labels >= 1])
        acquisition = make_constrained_qehvi_acquisition(
            model,
            semantics,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )

    # t-batch=2, simultaneous candidate count q and input dimension d=1.
    candidates = torch.linspace(0.20, 0.80, 2 * q, dtype=torch.double).view(2, q, 1)
    # Capture the exact PoF used *inside this forward call*: multiclass
    # predict_proba re-samples latent GP draws, so a separate call is not
    # guaranteed to return the same Monte Carlo estimate.
    observed_pof = []

    def capture_pof(_module, _inputs, output):
        observed_pof.append(output.detach().clone())

    handle = acquisition.probability_of_feasibility.register_forward_hook(capture_pof)
    try:
        with torch.no_grad():
            raw_objective = acquisition.objective_acquisition(candidates)
            weighted_value = acquisition(candidates)
    finally:
        handle.remove()
    assert len(observed_pof) == 1
    per_candidate_pof = observed_pof[0]

    assert per_candidate_pof.shape == (2, q)
    assert raw_objective.shape == weighted_value.shape == (2,)
    assert torch.isfinite(weighted_value).all()
    assert ((per_candidate_pof >= 0) & (per_candidate_pof <= 1)).all()
    torch.testing.assert_close(
        weighted_value,
        raw_objective * per_candidate_pof.prod(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )

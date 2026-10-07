"""Tests for ordinal and explicit class-utility semantic contracts."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics import (
    ExpectedClassUtilityObjective,
    ObjectiveCollection,
    OrdinalProbabilityObjective,
    ProblemSemantics,
)


class _OrdinalClassifier(nn.Module, ClassificationModelMixin):
    num_outputs = 1

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def class_labels(self) -> tuple[object, ...]:
        return ("acceptable", "good", "bad")

    @property
    def likelihood_family(self) -> ClassificationLikelihoodFamily:
        return ClassificationLikelihoodFamily.CATEGORICAL

    @property
    def latent_output_structure(self) -> LatentOutputStructure:
        return LatentOutputStructure.PER_CLASS

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        probabilities = X.new_tensor([0.3, 0.6, 0.1])
        return probabilities.expand(*X.shape[:-1], 3)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        shape = torch.Size() if sample_shape is None else sample_shape
        return self.predict_proba(X).expand(*shape, *X.shape[:-1], 3)

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -(probabilities * probabilities.log()).sum(dim=-1)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)


def _model() -> HeterogeneousModel:
    return HeterogeneousModel(_OrdinalClassifier(), output_names=["grade"])


def test_ordinal_probability_uses_explicit_order_not_probability_axis_order() -> None:
    objective = OrdinalProbabilityObjective(
        output="grade",
        threshold_class="acceptable",
        class_order=("bad", "acceptable", "good"),
    )

    value = objective.evaluate(_model(), torch.rand(4, 2, dtype=torch.double))

    assert torch.allclose(value, torch.full((4,), 0.9, dtype=torch.double))


def test_ordinal_probability_can_select_lower_tail() -> None:
    objective = OrdinalProbabilityObjective(
        output="grade",
        threshold_class="acceptable",
        class_order=("bad", "acceptable", "good"),
        at_or_above=False,
    )

    value = objective.evaluate(_model(), torch.rand(3, 2, dtype=torch.double))

    assert torch.allclose(value, torch.full((3,), 0.4, dtype=torch.double))


def test_expected_class_utility_uses_explicit_mapping() -> None:
    objective = ExpectedClassUtilityObjective(
        output="grade",
        utilities=(("bad", 0.0), ("acceptable", 1.0), ("good", 3.0)),
    )

    value = objective.evaluate(_model(), torch.rand(2, 2, dtype=torch.double))

    assert torch.allclose(value, torch.full((2,), 2.1, dtype=torch.double))


def test_ordinal_order_must_cover_every_class_once() -> None:
    objective = OrdinalProbabilityObjective(
        output="grade",
        threshold_class="acceptable",
        class_order=("bad", "acceptable", "acceptable"),
    )

    try:
        objective.evaluate(_model(), torch.rand(1, 2))
    except ValueError as error:
        assert "every classifier class exactly once" in str(error)
    else:
        raise AssertionError("invalid class_order must be rejected")


def test_utility_mapping_must_cover_every_class_once() -> None:
    objective = ExpectedClassUtilityObjective(
        output="grade",
        utilities=(("bad", 0.0), ("acceptable", 1.0)),
    )

    try:
        objective.evaluate(_model(), torch.rand(1, 2))
    except ValueError as error:
        assert "exactly one value for every classifier class" in str(error)
    else:
        raise AssertionError("incomplete utilities must be rejected")


def test_ordinal_and_utility_objectives_join_problem_semantics() -> None:
    ordinal = OrdinalProbabilityObjective(
        output="grade",
        threshold_class="acceptable",
        class_order=("bad", "acceptable", "good"),
    )
    utility = ExpectedClassUtilityObjective(
        output="grade",
        utilities=(("bad", 0.0), ("acceptable", 1.0), ("good", 3.0)),
    )
    objectives = ObjectiveCollection(ordinal, utility)
    semantics = ProblemSemantics(objectives=objectives)

    semantics.validate(_model())

    assert objectives.resolve_outputs(_model()) == (0, 0)

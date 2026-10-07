"""Tests for multiclass-ready semantic class selection."""

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import ClassificationConstraint, ProbabilityObjective


@dataclass(frozen=True)
class _MulticlassMetadata:
    num_classes: int = 3
    class_labels: tuple[object, ...] = ("fail", "review", "pass")
    likelihood_family: ClassificationLikelihoodFamily = (
        ClassificationLikelihoodFamily.CATEGORICAL
    )
    latent_output_structure: LatentOutputStructure = LatentOutputStructure.PER_CLASS


class _MulticlassClassifier(nn.Module, ClassificationModelMixin):
    task_type = "classification"
    is_classification = True
    num_outputs = 1
    @property
    def num_classes(self) -> int:
        return 3

    @property
    def class_labels(self) -> tuple[object, ...]:
        return ("fail", "review", "pass")

    @property
    def likelihood_family(self) -> ClassificationLikelihoodFamily:
        return ClassificationLikelihoodFamily.CATEGORICAL

    @property
    def latent_output_structure(self) -> LatentOutputStructure:
        return LatentOutputStructure.PER_CLASS

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        probabilities = torch.tensor(
            [0.1, 0.2, 0.7],
            dtype=X.dtype,
            device=X.device,
        )
        return probabilities.expand(*X.shape[:-1], 3)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        probabilities = self.predict_proba(X)
        shape = torch.Size() if sample_shape is None else sample_shape
        return probabilities.expand(*shape, *probabilities.shape)

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -(probabilities * probabilities.log()).sum(dim=-1)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> object:
        return torch.distributions.Categorical(probs=self.predict_proba(X, **kwargs))

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)


def _model() -> HeterogeneousModel:
    train_X = torch.rand(5, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(5, 1, dtype=torch.double)),
        _MulticlassClassifier(),
        output_names=["score", "status"],
    )


def test_classification_constraint_resolves_multiclass_label() -> None:
    model = _model()
    constraint = ClassificationConstraint("status", feasible_class="pass")

    assert constraint.resolve_output(model) == 1
    assert constraint.resolve_feasible_class(model) == 2
    assert constraint.to_probability_of_feasibility(model).feasible_class == 2


def test_probability_objective_resolves_multiclass_label() -> None:
    model = _model()
    objective = ProbabilityObjective("status", class_index="review")
    X = torch.rand(4, 2, dtype=torch.double)

    assert objective.resolve_output(model) == 1
    assert objective.resolve_class_index(model) == 1
    assert torch.allclose(objective.evaluate(model, X), torch.full((4,), 0.2))


def test_multiclass_semantics_preserve_integer_class_selection() -> None:
    model = _model()

    assert ClassificationConstraint("status", feasible_class=0).resolve_feasible_class(model) == 0
    assert ProbabilityObjective("status", class_index=2).resolve_class_index(model) == 2


@pytest.mark.parametrize("selector", ["unknown", 3, -1, True])
def test_multiclass_semantics_reject_unknown_class_selection(selector: object) -> None:
    model = _model()

    with pytest.raises(ValueError):
        ClassificationConstraint("status", feasible_class=selector).resolve_feasible_class(model)

    with pytest.raises(ValueError):
        ProbabilityObjective("status", class_index=selector).resolve_class_index(model)

"""Tests for classification probability objective semantics."""

import pytest
import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import ObjectiveDirection, ProbabilityObjective


def _model() -> HeterogeneousModel:
    train_X = torch.rand(8, 2, dtype=torch.double)
    return HeterogeneousModel(
        SingleTaskGP(train_X, torch.rand(8, 1, dtype=torch.double)),
        BinarySingleTaskGPClassifier(
            train_X,
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        ),
        output_names=["strength", "pass"],
    )


def test_probability_objective_resolves_classification_output() -> None:
    model = _model()

    assert ProbabilityObjective(output="pass", class_index=1).resolve_output(model) == 1


def test_probability_objective_rejects_regression_output() -> None:
    model = _model()

    with pytest.raises(TypeError, match="requires a classification output"):
        ProbabilityObjective(output="strength", class_index=0).resolve_output(model)


@pytest.mark.parametrize("class_index", [-1, 2])
def test_probability_objective_rejects_out_of_range_class(
    class_index: int,
) -> None:
    model = _model()

    with pytest.raises(ValueError, match="outside the classifier class range"):
        ProbabilityObjective(output="pass", class_index=class_index).resolve_output(model)


def test_probability_objective_rejects_non_integer_class_index() -> None:
    model = _model()

    with pytest.raises(TypeError, match="class_index must be an integer"):
        ProbabilityObjective(output="pass", class_index=True).resolve_output(model)


@pytest.mark.parametrize(
    "direction",
    [ObjectiveDirection.MAXIMIZE, ObjectiveDirection.MINIMIZE],
)
def test_probability_objective_evaluates_predictive_probability(
    direction: ObjectiveDirection,
) -> None:
    torch.manual_seed(7)
    model = _model()
    X = torch.rand(3, 2, dtype=torch.double)
    objective = ProbabilityObjective(
        output="pass",
        class_index=1,
        direction=direction,
    )
    classifier = model[1]
    expected = direction.apply(classifier.predict_proba(X)[..., 1])

    values = objective.evaluate(model, X)

    torch.testing.assert_close(values, expected)


def test_probability_objective_samples_posterior_induced_probabilities() -> None:
    torch.manual_seed(7)
    model = _model()
    X = torch.rand(3, 2, dtype=torch.double)
    objective = ProbabilityObjective(output="pass", class_index=0)
    values = objective.sample(model, X, sample_shape=torch.Size([4]))

    assert values.shape == torch.Size([4, 3])
    assert values.dtype == X.dtype
    assert torch.all((values >= 0.0) & (values <= 1.0))



def test_probability_objective_is_distinct_from_latent_posterior() -> None:
    torch.manual_seed(11)
    model = _model()
    X = torch.rand(3, 2, dtype=torch.double)
    classifier = model[1]
    objective = ProbabilityObjective(output="pass", class_index=1)

    latent_mean = classifier.latent_posterior(X).mean.squeeze(-1)
    probabilities = objective.evaluate(model, X)

    assert probabilities.shape == latent_mean.shape
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert not torch.equal(probabilities, latent_mean)


def test_probability_objective_forwards_native_prediction_kwargs() -> None:
    model = _model()
    X = torch.rand(2, 2, dtype=torch.double)
    classifier = model[1]
    objective = ProbabilityObjective(output="pass", class_index=1)
    calls: list[object] = []

    original = classifier.predict_proba

    def predict_proba(query: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append(kwargs["marker"])
        return original(query)

    classifier.predict_proba = predict_proba  # type: ignore[method-assign]

    objective.evaluate(model, X, marker="predictive")

    assert calls == ["predictive"]


def test_probability_objective_forwards_native_sampling_kwargs() -> None:
    model = _model()
    X = torch.rand(2, 2, dtype=torch.double)
    classifier = model[1]
    objective = ProbabilityObjective(output="pass", class_index=1)
    calls: list[object] = []

    original = classifier.sample_class_probabilities

    def sample_probabilities(
        query: torch.Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append(kwargs["marker"])
        return original(query, sample_shape=sample_shape)

    classifier.sample_class_probabilities = sample_probabilities  # type: ignore[method-assign]

    values = objective.sample(
        model,
        X,
        sample_shape=torch.Size([3]),
        marker="posterior",
    )

    assert calls == ["posterior"]
    assert values.shape == torch.Size([3, 2])

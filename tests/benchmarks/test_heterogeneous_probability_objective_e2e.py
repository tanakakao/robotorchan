"""Phase 11: fitted probability-objective evaluation and candidate selection."""

import pytest
import torch

from robotorchan.acquisition.composition import (
    make_probability_objective_bridge,
    resolve_acquisition_composition,
)
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics import ProbabilityObjective, ProblemSemantics


@pytest.mark.parametrize("class_index", [0, 1])
def test_probability_objective_candidate_selection(class_index: int) -> None:
    """Fit a binary GP and select the candidate maximizing predictive class probability."""
    generator = torch.Generator().manual_seed(1111)
    train_X = torch.rand(32, 3, generator=generator, dtype=torch.double)
    labels = observe(train_X, seed=2111).passed
    classifier = BinarySingleTaskGPClassifier(train_X, labels, inducing_points=8)
    classifier.train()
    classifier.likelihood.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.03)
    for _ in range(12):
        optimizer.zero_grad()
        latent = classifier.model(train_X)
        loss = -classifier.make_mll()(latent, labels.to(dtype=train_X.dtype))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    classifier.eval()
    classifier.likelihood.eval()

    model = HeterogeneousModel(classifier, output_names=["pass"])
    objective = ProbabilityObjective("pass", class_index=class_index)
    semantics = ProblemSemantics(objectives=(objective,))
    binding = resolve_acquisition_composition(model, semantics).objectives[0]
    bridge = make_probability_objective_bridge(model, binding)

    candidates = torch.rand(48, 3, generator=generator, dtype=torch.double)
    with torch.no_grad():
        probabilities = bridge.evaluate(candidates)
        expected = classifier.predict_proba(candidates)[..., class_index]
    assert probabilities.shape == (48,)
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    torch.testing.assert_close(probabilities, expected)

    selected = candidates[probabilities.argmax()].unsqueeze(0)
    assert selected.shape == (1, 3)
    assert ((selected >= 0) & (selected <= 1)).all()
    with torch.no_grad():
        strength, conductivity, true_pass = evaluate_truth(selected)
    assert torch.isfinite(strength).all()
    assert torch.isfinite(conductivity).all()
    assert torch.isfinite(true_pass).all()

    if bridge.supports_probability_samples:
        with torch.no_grad():
            samples = bridge.sample(candidates[:3], sample_shape=torch.Size([8]))
        assert samples.shape == (8, 3)
        assert torch.isfinite(samples).all()
        assert ((samples >= 0) & (samples <= 1)).all()


def test_probability_objective_rejects_regression_output() -> None:
    """ProbabilityObjective cannot silently interpret regression as classification."""
    from robotorchan.models.standard.single_task import SingleTaskGP

    X = torch.rand(8, 3, dtype=torch.double)
    Y = torch.rand(8, 1, dtype=torch.double)
    model = HeterogeneousModel(SingleTaskGP(X, Y), output_names=["strength"])
    objective = ProbabilityObjective("strength", class_index=1)
    with pytest.raises(TypeError, match="classification"):
        objective.resolve_output(model)

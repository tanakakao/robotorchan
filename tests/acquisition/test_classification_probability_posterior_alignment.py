"""Compare classification AL scores with class-probability posterior quantities."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    PredictiveEntropy,
    ProbabilityVariance,
    qProbabilityVariance,
)
from robotorchan.models.classification.binary.ensemble.gp import (
    GPBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)


def _classifier(kind: str):
    train_x = torch.linspace(0.05, 0.95, 9, dtype=torch.double).unsqueeze(-1)
    binary_y = (train_x[:, 0] > 0.5).long()
    if kind == "binary":
        classifier = BinarySingleTaskGPClassifier(train_x, binary_y, inducing_points=4)
    elif kind == "ensemble":
        classifier = GPBinaryClassificationEnsemble(
            BinarySingleTaskGPClassifier(train_x, binary_y, inducing_points=4),
            BinarySingleTaskGPClassifier(train_x, binary_y, inducing_points=4),
        )
    else:
        labels = torch.bucketize(
            train_x[:, 0].contiguous(),
            torch.tensor([0.35, 0.65], dtype=torch.double),
        )
        classifier_type = (
            OrdinalSingleTaskGPClassifier if kind == "ordinal" else MulticlassSingleTaskGPClassifier
        )
        classifier = classifier_type(train_x, labels, num_classes=3, inducing_points=4)
    classifier.eval()
    return classifier


@pytest.mark.parametrize("kind", ["binary", "ordinal", "multiclass", "ensemble"])
@torch.no_grad()
def test_posterior_mean_and_probability_variance_match_active_learning(kind: str) -> None:
    model = _classifier(kind)
    x = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.double)
    kwargs = {"num_samples": 1024, "seed": 7} if kind == "multiclass" else {}
    posterior = model.probability_posterior(x, **kwargs)
    assert posterior.mean.shape == posterior.variance.shape == (3, 1, model.num_classes)

    # Entropy depends on predictive label probabilities, not epistemic variance.
    entropy = PredictiveEntropy(model)(x)
    posterior_entropy = -torch.special.xlogy(posterior.mean, posterior.mean).sum(dim=-1).squeeze(-1)
    atol = 0.05 if kind == "multiclass" else 1e-6
    torch.testing.assert_close(entropy, posterior_entropy, atol=atol, rtol=atol)

    variance = ProbabilityVariance(model, num_samples=512)(x)
    expected = posterior.variance.mean(dim=-1).squeeze(-1)
    atol = 0.035 if kind in {"ordinal", "multiclass"} else 1e-6
    torch.testing.assert_close(variance, expected, atol=atol, rtol=atol)

    if kind in {"binary", "ensemble"}:
        bald = BALD(model)(x)
        reference = model.mutual_information(x).squeeze(-1)
        torch.testing.assert_close(bald, reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", ["ordinal", "multiclass"])
@torch.no_grad()
def test_mc_active_learning_scores_match_direct_conditional_probability_samples(
    kind: str,
) -> None:
    model = _classifier(kind)
    x = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.double)
    torch.manual_seed(43)
    samples = model.sample_class_probabilities(x, sample_shape=torch.Size([256]))
    variance_reference = samples.var(dim=0, unbiased=False).mean(dim=-1).squeeze(-1)
    mean = samples.mean(dim=0)
    predictive_entropy = -torch.special.xlogy(mean, mean).sum(dim=-1)
    conditional_entropy = -torch.special.xlogy(samples, samples).sum(dim=-1).mean(0)
    bald_reference = (predictive_entropy - conditional_entropy).clamp_min(0).squeeze(-1)

    torch.manual_seed(43)
    torch.testing.assert_close(
        ProbabilityVariance(model, num_samples=256)(x),
        variance_reference,
        atol=1e-8,
        rtol=1e-8,
    )
    torch.manual_seed(43)
    torch.testing.assert_close(
        BALD(model, num_samples=256)(x),
        bald_reference,
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.parametrize("kind", ["binary", "ensemble"])
@torch.no_grad()
def test_q_probability_variance_uses_marginal_posterior_variance(kind: str) -> None:
    model = _classifier(kind)
    x = torch.tensor(
        [[[0.2], [0.8]], [[0.35], [0.65]]],
        dtype=torch.double,
    )
    posterior = model.probability_posterior(x)
    expected = posterior.variance.mean(dim=-1).mean(dim=-1)
    torch.testing.assert_close(qProbabilityVariance(model)(x), expected, atol=1e-6, rtol=1e-6)

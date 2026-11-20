"""Independent class-wise Gaussian reference for multiclass GP posterior."""

import pytest
import torch

from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)


@pytest.mark.parametrize("mixed", [False, True])
def test_multiclass_latent_posterior_matches_independent_gp_reference(mixed: bool) -> None:
    torch.manual_seed(42)
    X = torch.linspace(0.0, 1.0, 9, dtype=torch.double).unsqueeze(-1)
    y = torch.arange(9) % 3
    if mixed:
        X = torch.cat((X, (torch.arange(9) % 2).to(X).unsqueeze(-1)), dim=-1)
        model = MixedMulticlassSingleTaskGPClassifier(
            X, y, num_classes=3, cat_dims=[1], inducing_points=5
        )
    else:
        model = MulticlassSingleTaskGPClassifier(X, y, num_classes=3, inducing_points=5)
    model.eval()
    query = X[:3].unsqueeze(0)
    posterior = model.posterior(query)
    mean = posterior.mean
    covariance = posterior.distribution.covariance_matrix
    assert mean.shape == (1, 3, 3)
    assert covariance.shape == (1, 9, 9)

    # MultitaskMultivariateNormal uses interleaved (point, class) ordering.
    for class_index, gp in enumerate(model.class_models):
        reference = gp.model(query)
        torch.testing.assert_close(mean[..., class_index], reference.mean)
        torch.testing.assert_close(
            covariance[..., class_index::3, class_index::3],
            reference.covariance_matrix,
        )
        for other_class in range(class_index + 1, 3):
            torch.testing.assert_close(
                covariance[..., class_index::3, other_class::3],
                torch.zeros_like(covariance[..., class_index::3, other_class::3]),
                atol=1e-9,
                rtol=0,
            )

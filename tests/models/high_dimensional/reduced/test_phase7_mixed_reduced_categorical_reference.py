"""Independent categorical passthrough and posterior coordinate reference."""

import torch

from robotorchan.models import MixedPCAGP


def test_mixed_pca_retains_categories_and_matches_reduced_posterior() -> None:
    """Verify untouched category IDs and public/latent posterior equivalence."""
    continuous = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 2.0, 0.0],
            [3.0, 1.0, 3.0],
            [4.0, 3.0, 1.0],
            [5.0, 2.0, 4.0],
        ],
        dtype=torch.float64,
    )
    categories = torch.tensor([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=torch.float64)
    train_x = torch.cat((continuous[:, :1], categories, continuous[:, 1:]), dim=-1)
    train_y = continuous[:, :1] - 0.2 * continuous[:, 1:2]
    model = MixedPCAGP(train_x, train_y, n_components=2, cat_dims=[1])
    model.eval()
    model.likelihood.eval()

    candidates = torch.tensor(
        [
            [[0.5, 0.0, 0.5, 1.5], [1.5, 1.0, 1.0, 0.5]],
            [[2.5, 1.0, 2.0, 2.0], [4.5, 0.0, 2.5, 3.5]],
        ],
        dtype=torch.float64,
    )
    reduced = model._transform_original_inputs(candidates)
    assert model.reduced_cat_dims == [2]
    torch.testing.assert_close(reduced[..., 2], candidates[..., 1])
    torch.testing.assert_close(model.train_inputs[0][..., 2], train_x[..., 1])

    with torch.no_grad():
        public = model.posterior(candidates)
        latent = model.posterior(reduced)

    torch.testing.assert_close(public.mean, latent.mean, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(public.variance, latent.variance, atol=1e-10, rtol=1e-10)

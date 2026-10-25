"""Mathematical coordinate-equivalence reference for PCA-reduced GP posteriors."""

import torch

from robotorchan.models import PCAGP


def test_pca_reduced_gp_posterior_is_invariant_to_input_coordinates() -> None:
    """Compare posterior statistics before and after a frozen PCA transform."""
    train_x = torch.tensor(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
            [3.0, 1.0, 3.0, 0.0],
            [4.0, 3.0, 1.0, 2.0],
            [5.0, 2.0, 4.0, 1.0],
        ],
        dtype=torch.float64,
    )
    train_y = (train_x[:, :1] - 0.3 * train_x[:, 1:2]).clone()
    model = PCAGP(train_X=train_x, train_Y=train_y, n_components=2)
    model.eval()
    model.likelihood.eval()

    candidates = torch.tensor(
        [
            [[0.5, 0.8, 1.5, 0.2], [2.5, 1.3, 1.2, 0.7]],
            [[3.5, 2.0, 2.5, 1.3], [4.5, 2.5, 3.0, 1.5]],
        ],
        dtype=torch.float64,
    )
    reducer = model.input_reducer
    assert reducer is not None
    basis_before = reducer.components.detach().clone()
    mean_before = reducer.mean.detach().clone()
    latent = reducer.transform(candidates)

    with torch.no_grad():
        public = model.posterior(candidates)
        reduced = model.posterior(latent)

    assert public.mean.shape == (2, 2, 1)
    assert public.variance.shape == (2, 2, 1)
    torch.testing.assert_close(public.mean, reduced.mean, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(public.variance, reduced.variance, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(reducer.components, basis_before)
    torch.testing.assert_close(reducer.mean, mean_before)

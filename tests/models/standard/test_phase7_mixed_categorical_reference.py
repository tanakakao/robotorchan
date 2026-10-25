"""Phase 7 categorical relabeling invariance of native mixed GP covariance."""

import torch

from robotorchan.models import MixedSingleTaskGP


def test_mixed_gp_kernel_is_invariant_to_category_relabeling() -> None:
    train_x = torch.tensor(
        [
            [0.1, 0.0],
            [0.2, 1.0],
            [0.4, 2.0],
            [0.6, 0.0],
            [0.8, 1.0],
            [0.9, 2.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = MixedSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[-1],
        outcome_transform=None,
    )

    query = torch.tensor(
        [[0.15, 0.0], [0.3, 2.0], [0.7, 1.0], [0.85, 0.0]],
        dtype=torch.double,
    )
    # Arbitrary bijection: 0 -> 17, 1 -> -4, 2 -> 9.
    # The kernel must depend on category equality, not numerical distances.
    relabeled = query.clone()
    for original, replacement in ((0.0, 17.0), (1.0, -4.0), (2.0, 9.0)):
        relabeled[query[:, -1] == original, -1] = replacement

    with torch.no_grad():
        original_covariance = model.covar_module(query).to_dense()
        relabeled_covariance = model.covar_module(relabeled).to_dense()

    torch.testing.assert_close(
        original_covariance, relabeled_covariance, atol=1e-10, rtol=1e-10
    )
    assert original_covariance.shape == (4, 4)

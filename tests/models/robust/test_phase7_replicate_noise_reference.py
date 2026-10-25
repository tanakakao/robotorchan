"""Independent statistical reference for replicate-derived observation noise."""

import torch

from robotorchan.models.robust.replicate_noise import ReplicateNoiseSingleTaskGP


def test_replicate_noise_matches_unbiased_sample_variance_of_mean() -> None:
    """Check grouped means and standard errors against scalar statistics."""
    train_x = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [1.0]],
        dtype=torch.float64,
    )
    train_y = torch.tensor(
        [[1.0], [4.0], [2.0], [4.0], [4.0], [7.0], [9.0]],
        dtype=torch.float64,
    )
    model = ReplicateNoiseSingleTaskGP.from_replicates(
        train_x,
        train_y,
        noise_floor=1e-12,
    )

    for location in (0.0, 1.0):
        samples = train_y[train_x[:, 0] == location, 0]
        count = samples.numel()
        sample_mean = samples.sum() / count
        unbiased_variance = ((samples - sample_mean) ** 2).sum() / (count - 1)
        variance_of_mean = unbiased_variance / count

        index = (model.raw_train_X[:, 0] == location).nonzero(as_tuple=True)[0]
        assert index.numel() == 1
        i = index.item()
        assert model.replicate_counts[i].item() == count
        torch.testing.assert_close(model.raw_train_Y[i, 0], sample_mean)
        torch.testing.assert_close(model.replicate_variance[i, 0], unbiased_variance)
        torch.testing.assert_close(model.raw_train_Yvar[i, 0], variance_of_mean)

    assert torch.isfinite(model.raw_train_Yvar).all()
    assert (model.raw_train_Yvar > 0).all()

"""Phase 9 independent TuRBO trust-region geometry reference."""

import math

import torch

from robotorchan.optim import turbo_trust_region_bounds


def test_turbo_geometry_matches_direct_normalized_ard_formula() -> None:
    """Check unequal global widths, geometric normalization and boundary clipping."""
    bounds = torch.tensor(
        [[-2.0, 10.0, 100.0], [2.0, 30.0, 200.0]],
        dtype=torch.float64,
    )
    center = torch.tensor([-1.99, 20.0, 199.9], dtype=torch.float64)
    weights = torch.tensor([0.5, 1.0, 4.0], dtype=torch.float64)
    length = 0.2

    geometric_mean = math.prod(weights.tolist()) ** (1.0 / len(weights))
    normalized = weights / geometric_mean
    half_width = 0.5 * length * (bounds[1] - bounds[0]) * normalized
    expected_lower = torch.maximum(bounds[0], center - half_width)
    expected_upper = torch.minimum(bounds[1], center + half_width)
    expected = torch.stack((expected_lower, expected_upper))

    actual = turbo_trust_region_bounds(
        center,
        bounds,
        length=length,
        dimension_weights=weights,
    )

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.all(actual[0] >= bounds[0])
    assert torch.all(actual[1] <= bounds[1])
    assert actual[0, 0] == bounds[0, 0]
    assert actual[1, 2] == bounds[1, 2]
    torch.testing.assert_close(
        turbo_trust_region_bounds(
            center,
            bounds,
            length=length,
            dimension_weights=7.0 * weights,
        ),
        actual,
        atol=1e-12,
        rtol=1e-12,
    )

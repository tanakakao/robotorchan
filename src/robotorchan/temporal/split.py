"""Time-based rolling-origin validation with explicit label availability."""

import math
from collections.abc import Iterator, Sequence
from itertools import pairwise

from torch import Tensor

from robotorchan.temporal.axis import validate_times


def rolling_origin_splits(
    decision_times: Tensor,
    target_times: Tensor,
    *,
    origins: Sequence[float],
    test_duration: float,
    label_available_times: Tensor | None = None,
    train_duration: float | None = None,
    gap: float = 0.0,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield train/test row indices, using physical time rather than row counts.

    Training labels must be available strictly before origin minus gap.
    Test decision times lie in [origin, origin + test_duration). Equal decision
    times stay together across series. Test targets may extend past this window.
    Empty folds are rejected instead of silently changing the evaluation.
    """
    validate_times(decision_times)
    validate_times(target_times)
    available = target_times if label_available_times is None else label_available_times
    validate_times(available)
    for t in (target_times, available):
        if t.shape != decision_times.shape or t.device != decision_times.device:
            raise ValueError("time vectors must align")
    if (target_times < decision_times).any() or (available < target_times).any():
        raise ValueError("require decision <= target <= label availability")
    if not math.isfinite(test_duration) or test_duration <= 0:
        raise ValueError("test_duration must be positive and finite")
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("gap must be nonnegative and finite")
    if train_duration is not None and (
        not math.isfinite(train_duration) or train_duration <= 0
    ):
        raise ValueError("train_duration must be positive and finite")
    if not origins or any(not math.isfinite(origin) for origin in origins):
        raise ValueError("origins must be nonempty and finite")
    if any(a >= b for a, b in pairwise(origins)):
        raise ValueError("origins must be strictly increasing")
    for origin in origins:
        cutoff = origin - gap
        train = (decision_times < cutoff) & (available < cutoff)
        if train_duration is not None:
            train = train & (decision_times >= cutoff - train_duration)
        test = (decision_times >= origin) & (decision_times < origin + test_duration)
        train_idx = train.nonzero(as_tuple=True)[0]
        test_idx = test.nonzero(as_tuple=True)[0]
        if not train_idx.numel() or not test_idx.numel():
            raise ValueError(f"empty train or test fold at origin {origin}")
        yield train_idx, test_idx

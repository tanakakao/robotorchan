"""Runtime validation for candidate-space constraints."""

from __future__ import annotations

from botorch.exceptions.errors import UnsupportedError
from torch import Tensor

from robotorchan.optim.constraints.contracts import CandidateConstraints


def validate_candidate_constraints(
    constraints: CandidateConstraints | None,
    *,
    bounds: Tensor,
    q: int,
    sequential: bool = False,
) -> None:
    """Validate candidate constraints once public input dimension and q are known."""
    if constraints is None:
        return
    d = bounds.shape[-1]
    for name, linear_constraints in (
        ("inequality_constraints", constraints.inequality_constraints),
        ("equality_constraints", constraints.equality_constraints),
    ):
        for indices, _, _ in linear_constraints:
            if indices.numel() == 0:
                raise ValueError(f"{name} indices must not be empty.")
            if indices.ndim == 1:
                if (indices < 0).any() or (indices >= d).any():
                    raise ValueError(f"{name} contains an input-dimension index outside [0, d).")
            else:
                q_indices = indices[:, 0]
                d_indices = indices[:, 1]
                if (q_indices < 0).any() or (q_indices >= q).any():
                    raise ValueError(f"{name} contains a q index outside [0, q).")
                if (d_indices < 0).any() or (d_indices >= d).any():
                    raise ValueError(f"{name} contains an input-dimension index outside [0, d).")
                if sequential and q > 1:
                    raise UnsupportedError(
                        "inter-point linear constraints require joint q-batch optimization; "
                        "sequential=True is unsupported."
                    )

    if (
        sequential
        and q > 1
        and any(
            not is_intrapoint for _, is_intrapoint in constraints.nonlinear_inequality_constraints
        )
    ):
        raise UnsupportedError(
            "inter-point nonlinear constraints require joint q-batch optimization; "
            "sequential=True is unsupported."
        )

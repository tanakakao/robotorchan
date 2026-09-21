# BoundaryVariance

`BoundaryVariance` is a robotorchan-specific level-set heuristic. It is not exposed as a
named BoTorch acquisition and the current implementation should not be cited as a standard
literature algorithm.

For posterior mean `mu(x)`, variance `v(x)`, and target level `t`, the score is

`v(x) * exp(-0.5 * ((mu(x) - t) / sqrt(v(x))) ** 2)`.

The first factor prefers uncertain locations. The second suppresses locations whose posterior
mean is far from the requested boundary relative to posterior uncertainty.

Unlike `Straddle`, `BoundaryVariance` has no `beta` exploration multiplier. Earlier versions
inherited `beta` from `Straddle` even though the score never used it; that misleading API has
been removed rather than retained as a compatibility argument.

The acquisition currently supports q=1 and rejects ensemble posteriors until their reduction
semantics are explicitly defined.

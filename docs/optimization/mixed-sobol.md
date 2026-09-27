# Mixed-variable Sobol sampling

Mixed Sobol uses the same `MixedVariableSpace` contract as Mixed Random while
retaining a meaningful low-discrepancy interpretation.

A scrambled Sobol point is first generated in the ordinary continuous box. For
structured coordinates, its normalized coordinate is then mapped through a
discrete inverse CDF:

- integer dimensions use equal-width bins over every legal integer;
- categorical dimensions use equal-width bins over the explicit category set;
- continuous dimensions retain the original Sobol coordinate.

This is not nearest-value rounding. Category numeric labels are never treated as
distances or interpolation targets. Their declared order only determines the
inverse-CDF bin assignment. With uniform bins, every category has equal marginal
probability.

The mapping preserves one-dimensional Sobol stratification for each structured
coordinate, although discretization necessarily creates repeated values and the
result is not a continuous Sobol net in the original mixed domain. This limitation
is explicit: the implementation claims structured low-discrepancy sampling, not
a mathematically continuous Sobol sequence over categorical space.

q-batches are generated from a Sobol engine of dimension `q * d`, so every
point-coordinate pair occupies its own Sobol dimension. Fixed features are
applied after the mixed mapping and validated by `MixedVariableSpace`.

This design is suitable as a sampling baseline and initialization mechanism. It
does not imply an ordinal metric between categorical values.

# Mixed-variable random sampling

The random acquisition optimizer accepts the public `MixedVariableSpace`
contract.

For each sampled q-batch:

- continuous dimensions are sampled uniformly inside their bounds;
- integer dimensions are sampled uniformly from every legal integer in
  `ceil(lower) ... floor(upper)`;
- categorical dimensions are sampled uniformly from their explicit allowed set;
- fixed features are applied after sampling and are validated against the
  variable-space contract.

This is direct sampling from the declared domain. Integer and categorical
coordinates are not produced by continuous sampling followed by rounding.

The same variable semantics apply independently to every point in a q-batch.
A fixed seed reproduces both the continuous and structured coordinates.

Mixed-variable Sobol is intentionally not implemented in this phase. Supplying
integer or categorical metadata with `optimizer="sobol"` raises
`NotImplementedError` rather than silently rounding low-discrepancy points.
Its semantics are assessed separately in Phase 5.

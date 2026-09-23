# BoTorch public seam for additive MAP-SAAS × multi-fidelity

## Phase 36 finding

The upstream source audit found the public utility required by the Phase 35 gate:
`botorch.models.map_saas.get_additive_map_saas_covar_module`.

This changes the implementation status of additive MAP-SAAS × multi-fidelity from
"custom-kernel decision required" to **thin-wrapper implementation feasible**.

## Public upstream contract

`get_additive_map_saas_covar_module` constructs an `AdditiveKernel` containing `num_taus`
scaled Matern-5/2 kernels. Each component owns its own SAAS prior and sampled global shrinkage
level. The utility accepts:

- `ard_num_dims`;
- `num_taus`;
- `active_dims`;
- `batch_shape`;
- `dtype` and `device`.

The `active_dims` argument is the critical seam for robotorchan: it allows the additive design
covariance to see only non-fidelity raw dimensions.

## Intended composition

```text
design_dims
    -> get_additive_map_saas_covar_module(
           ard_num_dims=len(design_dims),
           active_dims=design_dims,
       )
    -> SingleTaskMultiFidelityGP(covar_module=...)
    -> native data / iteration fidelity kernels
```

This preserves the Phase 34/35 ownership boundary without copying BoTorch internals.

## Implementation decision

A public `AdditiveMapSaasMultiFidelityGP` is now justified as a BoTorch-first wrapper, subject to
runtime validation. It must:

1. normalize iteration and data fidelity dimensions in raw space;
2. reject duplicate fidelity roles and all-fidelity/no-design inputs;
3. require `linear_truncated=False`;
4. build the design covariance only through the public BoTorch utility;
5. pass raw design `active_dims` explicitly;
6. preserve raw training tensors and exact MLL behavior;
7. expose design/fidelity metadata for structural tests;
8. prove posterior sampling and native MF-KG execution;
9. inspect every additive component to prove fidelity dimensions are absent from `active_dims`.

## Ensemble boundary

This finding does not justify `EnsembleMapSaasMultiFidelityGP`. The ensemble model has separate
batch / mixture posterior semantics. It remains independently gated.

## Phase 36 outcome

No model is added in this audit phase. The previous uncertainty about whether a stable public
kernel seam exists is resolved: **it does**. The next phase should implement the additive
multi-fidelity wrapper using that public utility and no copied/private BoTorch construction.

# Multi-fidelity × robust regression design audit

This document defines the Phase 10 design boundary for robust multi-fidelity regression.

The implementation rule is the same as elsewhere in robotorchan: keep BoTorch's native
multi-fidelity response model and acquisition semantics whenever possible. Add robotorchan-owned
robust observation machinery only where the statistical contract is explicit.

## Statistical separation

Multi-fidelity and robustness describe different parts of the surrogate:

- fidelity kernels model how the latent response changes with fidelity;
- robust / noise models describe how observations depart from that latent response.

A combined model must not encode fidelity as ordinary noise metadata, and a noise model must not
silently remove or rewrite the fidelity coordinate.

## Candidate audit

| Family | Decision | Priority | Reason |
| --- | --- | --- | --- |
| Heteroskedastic × MultiFidelity | implement | high | input-dependent observation variance has a clear two-GP interpretation |
| ReplicateNoise × MultiFidelity | implement | high | empirical variance of replicate means maps directly to fixed observation noise |
| Robust relevance pursuit × MultiFidelity | research-gated | medium | BoTorch RRP is a specialized likelihood/mixin; composition must be audited before reuse |
| Student-t × MultiFidelity | defer | medium | custom likelihood/inference path should follow simpler exact-GP combinations |
| Contaminated × MultiFidelity | defer | medium | mixture observation semantics add inference complexity |
| Nonstationary × MultiFidelity | separate axis | low | nonstationary covariance is not primarily an observation-robustness combination |

## First implementation targets

### Replicate-noise multi-fidelity

This is the lowest-risk reference implementation.

1. group exact duplicate raw rows, including the fidelity coordinate;
2. compute the mean response and empirical variance of each replicate group;
3. use variance-of-the-mean as `train_Yvar`;
4. pass aggregated raw inputs and fixed noise to `SingleTaskMultiFidelityGP`;
5. preserve unaggregated replicate tensors for auditability.

Replicates at different fidelities are different experimental conditions and must never be pooled.

### Heteroskedastic multi-fidelity

Use the existing iterative two-GP interpretation:

1. response model: `SingleTaskMultiFidelityGP`;
2. compute residuals at the raw training rows;
3. log-noise model: another multi-fidelity GP using the same fidelity contract;
4. predict input- and fidelity-dependent observation variance;
5. update the response model's fixed-noise likelihood.

The noise model should retain the fidelity coordinate. Observation noise may legitimately vary with
fidelity, so excluding fidelity from the noise model would impose an unjustified shared-noise
assumption.

## Fidelity contract

Combined robust multi-fidelity models must:

- accept the same `iteration_fidelity` / `data_fidelities` semantics as the standard wrapper;
- normalize negative fidelity indices;
- reject duplicate fidelity dimensions;
- preserve raw fidelity values;
- keep fidelity dimensions structural for the response covariance;
- never aggregate observations across different fidelity values.

## Robust relevance pursuit boundary

Do not create `RobustRelevancePursuitMultiFidelityGP` merely through multiple inheritance.

Before implementation, verify that BoTorch's robust relevance-pursuit likelihood can wrap the
likelihood used by `SingleTaskMultiFidelityGP` without changing its intended sparse-outlier
semantics. The previous Kronecker audit demonstrated that neighboring model support is not evidence
that the likelihood contract composes safely.

If the BoTorch RRP mixin composes directly with the single-task multi-fidelity likelihood, a thin
wrapper is preferred. Otherwise this combination remains research-gated.

## Runtime acceptance criteria

Each implemented family must prove:

1. raw training-data contract;
2. fidelity-index normalization and duplicate rejection;
3. finite posterior mean / variance;
4. finite posterior samples;
5. `make_mll()` for exact-GP-compatible variants;
6. representative MC acquisition evaluation;
7. a representative multi-fidelity acquisition or candidate-generation path;
8. robust-specific behavior:
   - replicate model: grouping and empirical variance;
   - heteroskedastic model: finite positive predicted noise.

Runtime validation is model × workflow specific; this document does not introduce a model-wide
`runtime_validated` flag.

## Recommended implementation order

1. `ReplicateNoiseMultiFidelityGP`;
2. `HeteroskedasticMultiFidelityGP`;
3. audit RRP likelihood composition;
4. consider mixed-input variants only after continuous versions pass runtime validation;
5. defer Student-t / contaminated variants until they have a concrete practical requirement.

## Non-goals

Phase 10 does not:

- implement classification or ordinal models;
- create every robust × fidelity Cartesian product;
- claim RRP support from neighboring single-task models;
- merge observations across fidelity levels;
- replace BoTorch fidelity kernels with a robotorchan reimplementation.

## Phase 10 outcome

MultiFidelity × Robust moves from blanket unsupported status to a staged regression plan with two
statistically clear implementation candidates. Phase 11 should start with replicate-noise
multi-fidelity because it provides the simplest executable reference contract.

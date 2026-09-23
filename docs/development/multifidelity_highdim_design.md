# Multi-fidelity × high-dimensional design audit

This document defines the Phase 8 design boundary for high-dimensional multi-fidelity regression.
It follows the project rule: reuse BoTorch through wrappers where BoTorch provides the statistical
model, and consider a robotorchan implementation only for capability combinations that BoTorch
does not provide directly.

## Source-of-truth findings

BoTorch provides `SingleTaskMultiFidelityGP` and the fidelity kernels used by that model. BoTorch
also provides SAAS / MAP-SAAS building blocks, but it does not currently expose a dedicated
fully-Bayesian SAAS multi-fidelity model. Therefore robotorchan must not present a simple wrapper as
if that cross capability already existed upstream.

The existing robotorchan `SingleTaskMultiFidelityGP` is the base contract for this work:

- fidelity columns are structural model inputs;
- `iteration_fidelity` and `data_fidelities` remain explicit;
- caller `raw_train_X`, `raw_train_Y`, and `raw_train_Yvar` are preserved;
- multi-fidelity acquisition semantics remain delegated to BoTorch.

## Design invariant

A high-dimensional transform may act on design variables, but it must never silently consume,
project, encode, or reconstruct a fidelity column as an ordinary design feature.

For raw input

```text
X = [design features | fidelity features]
```

the conceptual path is

```text
design features -> high-dimensional transform -> reduced / regularized design representation
fidelity features ----------------------------> native fidelity covariance
```

The two paths are joined only inside the final covariance/model contract.

## Candidate audit

| Family | Decision | Priority | Rationale |
| --- | --- | --- | --- |
| PCA × MultiFidelity | implement | high | deterministic transform can exclude fidelity columns cleanly |
| PLS × MultiFidelity | implement after PCA | high | same structural split, supervised reducer needs target-aware tests |
| RandomProjection × MultiFidelity | implement after PCA | medium | simple deterministic baseline |
| MAP-SAAS × MultiFidelity | investigate / implement if covariance composition is clean | high | BoTorch exposes `add_saas_prior`, but no dedicated MF wrapper |
| Fully Bayesian SAAS × MultiFidelity | research-gated | medium | no upstream dedicated model; requires a fidelity-aware Pyro model, not a wrapper |
| AutoEncoder / VAE × MultiFidelity | defer | medium | encoder ownership and fidelity exclusion need a dedicated transform contract |
| Joint encoder × MultiFidelity | defer | low | joint training makes fidelity leakage and MLL gradient ownership more complex |
| ALEBO × MultiFidelity | defer | low | embedding must exclude fidelity dimensions and preserve target-fidelity projection |

## First implementation target

Phase 9 should implement a deterministic reduction reference model first, preferably
`PCAMultiFidelityGP`.

The preferred architecture is:

1. resolve fidelity dimensions in raw space;
2. fit PCA only on non-fidelity design columns;
3. transform raw design columns to latent continuous coordinates;
4. append untouched fidelity columns to the encoded input;
5. call the existing robotorchan `SingleTaskMultiFidelityGP` wrapper;
6. expose `posterior` from raw-space inputs by applying the same model-owned transform.

This is an intentional robotorchan composition because BoTorch does not provide a PCA
multi-fidelity wrapper. The underlying GP and fidelity covariance remain BoTorch-native.

## Public contract

A reduction-based multi-fidelity model must:

- accept raw `train_X`;
- accept the same fidelity arguments as `SingleTaskMultiFidelityGP`;
- preserve raw training tensors;
- reject duplicate fidelity dimensions;
- never include fidelity dimensions in the reducer;
- retain fidelity values exactly through the transform;
- accept raw-space `X` in `posterior`;
- expose enough reducer metadata to audit the raw-to-latent mapping.

Negative fidelity indices should follow the existing feature-dimension normalization convention.

## Runtime acceptance criteria

The Phase 9 reference implementation must prove:

1. raw training-data equality;
2. fidelity exclusion from PCA fitting;
3. encoded input contains the untouched fidelity coordinate;
4. finite posterior mean and variance;
5. finite posterior samples;
6. representative MC acquisition evaluation;
7. representative multi-fidelity acquisition or candidate-generation path;
8. `make_mll()` remains valid for the exact-GP implementation.

A normal high-dimensional GP test is not sufficient evidence because it does not verify fidelity
semantics.

## MAP-SAAS boundary

BoTorch's `add_saas_prior` is a reusable building block rather than a dedicated multi-fidelity
model. A future MAP-SAAS × MultiFidelity implementation may use that upstream utility if the SAAS
prior can be restricted to design dimensions while leaving fidelity-kernel parameters outside the
SAAS shrinkage contract.

Do not apply SAAS blindly to the complete multi-fidelity covariance.

## Fully Bayesian SAAS boundary

Do not derive `SaasFullyBayesianMultiFidelityGP` by merely subclassing the existing fully Bayesian
single-task SAAS wrapper. A sound implementation needs a Pyro model whose covariance explicitly
combines SAAS design-feature behavior with fidelity covariance.

Until that exists and has executable posterior / acquisition evidence, fully Bayesian SAAS ×
MultiFidelity remains research-gated rather than advertised as supported.

## Non-goals

Phase 8 does not:

- implement classification or ordinal models;
- add a Cartesian product of every reducer and fidelity model;
- treat fidelity as a categorical feature;
- reduce fidelity coordinates;
- claim fully Bayesian SAAS × MultiFidelity support from neighboring SAAS and MF models alone.

## Phase 8 outcome

The former blanket `MultiFidelity × High-dimensional = intentionally unsupported` status is
replaced by a staged implementation plan:

1. deterministic reduction reference path;
2. PLS / random-projection variants if the reference path is stable;
3. MAP-SAAS covariance-composition audit (completed in Phase 22; see `map_saas_multifidelity_audit.md`);
4. learned reduction only when fidelity exclusion is explicit;
5. fully Bayesian SAAS only through a dedicated fidelity-aware probabilistic model.

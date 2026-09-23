# Regression runtime validation matrix

This document is the Phase 4 source of truth for representative runtime validation of regression
model families. It records executable workflow evidence separately from structural model
capabilities.

## Rules

- A capability declaration is not runtime evidence.
- Validation is recorded per model family and workflow, never as a model-wide boolean.
- Representative tests are preferred over a Cartesian product of every model and acquisition.
- New cross-capability models must not be justified only by neighboring implementations.
- Classification and ordinal workflows are outside this regression matrix.

## Workflow levels

| Level | Required evidence |
| --- | --- |
| P | posterior construction and finite mean / variance or samples |
| S | posterior sampling with the sampler required by the model |
| MC | one representative Monte Carlo acquisition evaluates successfully |
| F | fantasy-dependent acquisition or fantasy model workflow |
| MF | multi-fidelity acquisition / candidate-generation workflow |
| AL | regression Active Learning workflow |

## Current regression matrix

| Family | Representative models | P | S | MC | F / MF | AL | Phase 4 assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard exact | SingleTaskGP | yes | yes | yes | yes | not required here | baseline complete |
| Mixed exact | MixedSingleTaskGP | yes | yes | yes | yes | not required here | baseline complete |
| Kronecker | KroneckerMultiTaskGP | yes | yes | yes | acquisition-specific | not required here | representative multi-output path exists |
| Multi-fidelity | SingleTaskMultiFidelityGP | yes | yes | yes | yes | not required here | baseline complete |
| Mixed multi-fidelity | MixedSingleTaskMultiFidelityGP | yes | yes | yes | yes | not required here | strong runtime coverage |
| Reduced | ReducedGP / PCA / PLS / RP | yes | yes | yes | yes | not required here | representative coverage exists |
| Mixed reduced | Mixed reduced family | yes | yes | yes | yes | not required here | representative coverage exists |
| MultiTask high-dimensional | reduced / neural / SAAS / expressive | yes | family-dependent | partial | partial | not required here | validation-depth gap |
| Mixed robust | robust mixed families | yes | family-dependent | partial | not required | not required here | validation-depth gap |
| MultiTask robust | robust multitask families | yes | family-dependent | partial | not required | not required here | validation-depth gap |
| Empirical tree ensemble | RF / ExtraTrees / boosting | yes | yes | yes | no | partial / unproven | regression AL gap |
| Multi-fidelity high-dimensional | PCA / PLS / RP / MAP-SAAS MF | yes | yes | yes | yes | not required here | deterministic reduction implemented; MAP-SAAS MF has native MF-KG evidence |
| Multi-fidelity robust | ReplicateNoiseMultiFidelityGP / HeteroskedasticMultiFidelityGP | yes | yes | yes | yes | not required here | both public families have native MF-KG workflow evidence |

“Yes” means representative executable evidence exists in the repository. “Partial” means the family
is structurally broad but evidence is not uniform enough to claim family-wide workflow validation.

## Phase 4 hardening targets

The following targets define the work that later implementation phases must close.

### Robust MultiTask and Mixed robust

At least one representative model from each materially different posterior family should prove:

1. posterior construction;
2. finite posterior output;
3. posterior sampling when declared;
4. one compatible MC acquisition.

Do not duplicate this test for every naming variant when implementations share the same posterior
contract.

### High-dimensional MultiTask

Keep separate representatives for:

- deterministic reduction;
- learned neural reduction;
- fully Bayesian SAAS;
- expressive / variational models.

The purpose is to catch transform, task-axis, and sampler incompatibilities rather than to test
every reducer.

### Kronecker extensions

Future robust Kronecker models require aligned block-design observations. A long-format MultiTask
test is not sufficient evidence. Each new statistical family must prove block-design posterior
shape, finite sampling, and a representative multi-output acquisition path.

### Multi-fidelity cross-capability models

The high-dimensional branch is specified in
[`multifidelity_highdim_design.md`](multifidelity_highdim_design.md). Deterministic reduction is
the first implementation path; fully Bayesian SAAS remains research-gated until a fidelity-aware
probabilistic model is designed.

Future high-dimensional or robust multi-fidelity models must preserve the fidelity feature as a
structural dimension. Reduction or learned encoders must not silently consume the fidelity column.

Required evidence is:

1. raw training-data contract;
2. fidelity-column validation;
3. posterior and sampling;
4. a multi-fidelity acquisition such as qMFKG where semantically supported;
5. candidate generation preserving fidelity semantics.


The robust branch is specified in [`multifidelity_robust_design.md`](multifidelity_robust_design.md).
Replicate-noise and heteroskedastic multi-fidelity models are the first statistically explicit
implementation targets; relevance pursuit remains research-gated pending likelihood composition.

### Probabilistic non-GP and regression AL

External probabilistic surrogates must expose uncertainty through an explicit posterior adapter.
For NGBoost, the target path is:

```text
NGBoost predictive distribution
    -> robotorchan posterior
    -> posterior samples
    -> MC BO acquisition
    -> compatible regression AL acquisition
```

Tree ensembles and future probabilistic external models must not be treated as having identical
uncertainty semantics. Regression AL compatibility must distinguish at least empirical ensemble
sampling from parametric predictive-distribution sampling. BALD-like epistemic / aleatoric
decompositions must not be inferred from predictive variance alone.

## Exit criteria

Phase 4 is complete when:

- the validation matrix is explicit and reviewable;
- intentionally unsupported combinations remain distinguishable from validation gaps;
- future Kronecker, multi-fidelity cross-capability, and probabilistic non-GP work has executable
  acceptance criteria;
- no model-wide `runtime_validated` capability is introduced.

Later phases should update this matrix only when executable evidence is added.


### RRP × multi-fidelity

The RRP composition boundary is audited in [`rrp_multifidelity_audit.md`](rrp_multifidelity_audit.md).
It remains research-gated rather than being forced through multiple inheritance.

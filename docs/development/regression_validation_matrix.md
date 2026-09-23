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

## Phase 27: reduced multi-fidelity fantasize boundary

PCA, PLS, and random-projection multi-fidelity models own a raw-to-encoded input mapping and
currently override `posterior` only. BoTorch fantasy construction conditions the underlying exact
GP on new observations; without an explicit raw-space `condition_on_observations` / `fantasize`
contract, advertising generic fantasize support would imply that raw candidate tensors are safely
encoded on every conditioning path. That has not been established.

Accordingly, these three reduced multi-fidelity models no longer advertise `supports_fantasize`.
Their posterior, sampling, and ordinary MC evidence remains valid. A future implementation may
restore the capability only after the reducer is frozen and both conditioning and fantasy paths
are tested in raw coordinates.

This restriction does not apply automatically to `MapSaasMultiFidelityGP`, replicate-noise MF, or
heteroskedastic MF because those models keep the BoTorch multi-fidelity input coordinates rather
than introducing a private reduced coordinate system.

## Phase 28: reduced multi-fidelity conditioning

PCA, PLS, and random-projection multi-fidelity models now encode raw candidate coordinates before
`condition_on_observations`, matching the raw-space lifecycle contract already used by the general
reduced GP family. Runtime tests cover all three reducers and verify that the conditioned model
stores encoded inputs with finite posterior moments.

This closes the conditioning half of the Phase 27 gap. `supports_fantasize` remains false until the
full BoTorch fantasy path is exercised directly with a sampler and raw-space candidates; capability
metadata is not restored from conditioning evidence alone.

## Phase 29: reduced multi-fidelity fantasize lifecycle

PCA, PLS, and random-projection multi-fidelity models now encode raw candidate coordinates before
delegating to the native BoTorch fantasy path. Runtime coverage exercises `fantasize` with a
Sobol QMC normal sampler for all three reducer families and verifies finite fantasy posterior
moments in the encoded GP space.

The reducer is reused rather than refitted during fantasy construction, so design coordinates stay
in the same latent system while fidelity coordinates remain structural. With direct fantasy-path
evidence in place, these three models again advertise `supports_fantasize=True`.

## Phase 30: reduced multi-fidelity native MF-KG validation

PCA, PLS, and random-projection multi-fidelity models are now exercised directly with BoTorch
`qMultiFidelityKnowledgeGradient`, including an affine fidelity cost model, inverse-cost utility,
target-fidelity projection, and a target-fidelity `PosteriorMean` current value. This validates that
the raw-space fantasy lifecycle from Phase 29 is sufficient for a native cost-aware multi-fidelity
acquisition rather than only for direct `fantasize` calls.

## Phase 31: reduced multi-fidelity constructor hardening

PCA, PLS, and random-projection multi-fidelity constructors now distinguish `None` explicitly from
provided `data_fidelities` instead of relying on container truthiness. This keeps tensor-like
fidelity specifications valid and avoids PyTorch's ambiguous Boolean evaluation for multi-element
tensors. Runtime coverage verifies negative tensor indices normalize to the expected structural
fidelity coordinate without changing the reduced design-space mapping.

## Phase 32: reduced multi-fidelity structural edge contracts

The structural contracts are now exercised uniformly across PCA, PLS, and random-projection
multi-fidelity models. Runtime coverage verifies simultaneous iteration/data fidelity roles, exact
raw-to-encoded fidelity placement, rejection of overlapping structural roles, and rejection of the
`linear_truncated=True` path. The PCA/PLS linear-truncation diagnostics were also corrected so each
model names its own reducer rather than the sibling reducer.

## Phase 33: heteroskedastic multi-fidelity noise semantics

`HeteroskedasticMultiFidelityGP` now has direct runtime evidence that its learned noise process
retains the declared fidelity coordinate, preserves that coordinate in the noise model's raw
training inputs, and returns finite positive noise predictions at multiple fidelity levels. The
test deliberately does not require two finite-data posterior means to differ: a fitted noise GP may
legitimately regress both predictions to the configured noise floor. Existing native MF-KG workflow
coverage remains unchanged.


## Phase 40: in-scope regression validation re-audit

Phase 40 returns to the agreed regression-only program after the classification / ordinal boundary
was recorded in Phase 39. Classification and ordinal work is intentionally deferred to a separate
development stream.

The current source of truth shows that the six in-scope work areas are no longer at the same
maturity level they had when this matrix was introduced:

| In-scope area | Current state | Remaining work |
| --- | --- | --- |
| Existing regression validation | Representative core workflows are executable | close remaining robust / high-dimensional MultiTask depth gaps |
| Kronecker extensions | baseline plus practical nonstationary Kronecker path exists | validate acquisition/runtime evidence before adding any further model |
| Mixed × Kronecker | public MixedKroneckerMultiTaskGP exists | strengthen representative acquisition evidence only where missing |
| Probabilistic external non-GP | NGBoost posterior, MC BO, and regression AL paths exist | retain explicit distributional semantics; no new external model required now |
| MultiFidelity × High-dimensional | PCA, PLS, random projection, MAP-SAAS, and additive MAP-SAAS MF paths exist | keep ensemble MAP-SAAS MF research-gated; validation takes priority over more classes |
| MultiFidelity × Robust | replicate-noise and heteroskedastic MF paths have native MF-KG evidence | keep RRP × MF research-gated; validate existing public models rather than force composition |

### Source-of-truth corrections

Two older statements in the runtime documentation are now stale and must not guide new work:

- MultiFidelity × High-dimensional is now a public model surface.
- MultiFidelity × Robust is now a public model surface.

Their absence was true earlier in the audit but was closed by later implementation phases. Future
reviews must use the current registry, public exports, tests, and this matrix rather than historical
phase text.

### Priority after Phase 40

The highest-value remaining validation work is representative acquisition coverage for model
families still marked partial, not another broad model expansion. The next phase should inspect the
actual tests for:

1. high-dimensional MultiTask representatives, separated into deterministic reduction, learned
   neural reduction, fully Bayesian SAAS, and expressive / variational paths;
2. robust MultiTask and Mixed robust representatives with posterior sampling and one compatible MC
   acquisition;
3. Kronecker and Mixed × Kronecker acquisition evidence, adding tests only if the existing suite
   does not already establish the contract.

A missing Cartesian-product test is not by itself a gap. New tests should be added only when they
protect a materially different posterior, transform, sampler, or optimizer contract.


## Development-stream boundary and extension policy

This regression stream may add new regression models when an audit identifies a statistically
meaningful and practically useful gap. Validation is the first priority, not the terminal goal.
Candidates are evaluated by practical value, differentiation from existing models, statistical
contract, acquisition integration, reuse of public BoTorch APIs, implementation cost, maintenance
cost, and testability.

Use four decisions for expensive cross-capability candidates:

- **Implement**: value is high and implementation / maintenance cost is proportionate.
- **Prototype**: value is promising but statistical design or cost is still uncertain.
- **Hold**: useful, but current implementation or maintenance cost is disproportionate.
- **Reject**: insufficient statistical meaning, differentiation, or practical value.

A missing BoTorch public seam is not by itself a reason to reject a model. A small robotorchan-owned
kernel, likelihood, posterior adapter, transform, or covariance builder is acceptable when its
statistical contract is explicit and independently testable. Large copies of BoTorch private
implementation remain undesirable maintenance coupling.

Classification and ordinal models are intentionally developed in a separate stream. They must not
be pulled into this regression program merely because an acquisition or model audit discovers a
related future opportunity.


## Phase 2: validation-gap classification

This re-audit classifies missing evidence separately from missing model capability. It also restores
the intended Kronecker policy: practical Kronecker extensions remain an active implementation
target, while statistically weak Cartesian products are not added merely for symmetry.

The evidence codes used below are:

- **P**: posterior shape and finite moments or samples;
- **S**: posterior sampling through the model's supported sampler path;
- **MC**: representative Monte Carlo acquisition evaluation;
- **F**: fantasy-dependent workflow;
- **MF**: native multi-fidelity acquisition workflow;
- **AL**: regression Active Learning workflow;
- **OPT**: candidate optimization through the appropriate optimizer.

| Family | P | S | MC | F / MF | AL | OPT | Gap classification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard exact / Mixed exact | yes | yes | yes | yes | representative | yes | no material gap |
| Kronecker baseline | yes | yes | yes | acquisition-specific | not required | partial | validation gap |
| Mixed Kronecker | yes | partial | partial | acquisition-specific | not required | partial | validation gap |
| Nonstationary Kronecker | yes | yes | yes | not established | not required | not established | validation gap |
| High-dimensional MultiTask | yes | family-dependent | partial | partial | not required | partial | validation gap |
| Robust MultiTask | yes | family-dependent | partial | not generally required | not required | partial | validation gap |
| Mixed robust | yes | family-dependent | partial | not generally required | not required | partial | validation gap |
| Multi-fidelity high-dimensional | yes | yes | yes | yes | not required | representative | no core correctness gap |
| Multi-fidelity robust | yes | yes | yes | yes | not required | representative | no core correctness gap |
| Empirical non-GP ensemble | yes | yes | yes | unsupported | partial | partial | intentional AL limitation |
| NGBoost probabilistic surrogate | yes | yes | yes | unsupported | yes | acquisition path | no new model required |

### Gap categories

No current evidence establishes a major regression correctness defect. The actionable findings are
therefore validation and extension gaps rather than known numerical-correctness failures.

**Contract gaps.** No broad public-contract rewrite is justified in this phase. Capability metadata
must continue to describe structural support, while this matrix records executable evidence.

**Validation gaps.** High-dimensional MultiTask, robust MultiTask, Mixed robust, and the extended
Kronecker family need deeper representative acquisition and optimizer evidence. The missing tests
must be chosen by distinct posterior, transform, sampler, or optimizer semantics rather than by
class-name Cartesian products.

**Useful missing capability.** Kronecker is intentionally different from the other families. The
project goal is to provide a useful set of block-design Kronecker models, so later phases must audit
robust, high-dimensional, Mixed, and expressive Kronecker coverage for practical omissions. A
missing Kronecker extension may therefore become an implementation target when it preserves
block-design semantics and provides a use case not already served by an existing public model.

**Intentional limitations.** Ensemble Active Learning reductions that require an unsupported
epistemic / aleatoric decomposition remain unsupported. Research-gated multi-fidelity combinations
also remain outside the public surface until their statistical model is explicit.

### Phase 3 hand-off

Phase 3 should validate high-dimensional MultiTask representatives in four groups: deterministic
reduction, learned neural reduction, fully Bayesian SAAS, and expressive / variational models.
Posterior and sampler checks alone are insufficient where a representative MC acquisition or
candidate-optimization path is materially different.

The later Kronecker phases must not interpret this validation-first ordering as a freeze on model
development. They should actively identify practical missing Kronecker models, but use
**Implement**, **Prototype**, **Hold**, or **Reject** according to statistical value and maintenance
cost. Large copies of BoTorch private implementation are not an acceptable way to complete a
matrix cell.


## Next regression extension Phase 1: Kronecker extension matrix

This audit is based on main commit `1ee80b152e3f85203f442d3dbd549bad314dcd77`. The corresponding
main-branch CI push run completed successfully. This matrix distinguishes an existing public
implementation from a statistically meaningful future implementation target. A missing Cartesian
product is not automatically a defect.

| Kronecker family | Status | Evidence / decision |
| --- | --- | --- |
| Standard | Implemented | `KroneckerMultiTaskGP` preserves the BoTorch block-design contract. |
| Mixed | Implemented | `MixedKroneckerMultiTaskGP` keeps categorical data covariance separate from the output-task axis. |
| Reduced | Implemented | Shared `ReducedKroneckerMultiTaskGP` infrastructure is present. |
| PCA | Implemented | `PCAKroneckerMultiTaskGP` is public, registered, documented, and tested. |
| PLS | Implemented | `PLSKroneckerMultiTaskGP` is public, registered, documented, and tested. |
| Random Projection | Implemented | `RandomProjectionKroneckerMultiTaskGP` is public, registered, documented, and tested. |
| AE / VAE | Implemented | Frozen AE/VAE Kronecker variants exist; joint encoder/VAE block-design variants also exist. |
| Nonstationary | Implemented | `NonstationaryKroneckerMultiTaskGP` replaces only the data covariance with a Gibbs kernel. |
| Spectral Mixture | Implemented | `SpectralMixtureKroneckerMultiTaskGP` is public; the Mixed block-design variant is also implemented. |
| Infinite-width BNN | Implemented | `InfiniteWidthBNNKroneckerMultiTaskGP` is public; the Mixed block-design variant is also implemented. |
| Fully Bayesian SAAS | Hold | Phase 9 found no stable block-design Pyro/sample-loading path; a custom implementation would own substantial fully Bayesian internals. |
| Heteroskedastic | Prototype | Design documentation exists, but no public Kronecker implementation exists. Predicted noise must alter response observation covariance with an explicit task axis. |
| Robust Relevance Pursuit | Prototype | A dedicated block-design sparse observation/outlier model is required; the single-task `noise_covar` mixin is not a valid shortcut. |
| Student-t | Prototype | Meaningful heavy-tail extension, but it needs a block-design variational likelihood/posterior contract. |
| Contaminated | Prototype | Requires an explicit choice between shared and task-specific contamination parameters. |
| Replicate Noise | Prototype | Phase 11 validated aligned group × task variance-of-mean semantics; public implementation remains blocked on explicit fixed `G × m` Kronecker observation noise. |
| DeepGP | Hold | A probabilistic hierarchy does not reduce to a data-kernel substitution; a dedicated block-design variational architecture would be disproportionately complex now. |

### Phase 1 conclusions

The high-dimensional Kronecker surface is substantially more complete than a family-name-only audit
suggests: deterministic reduction, frozen neural reduction, and joint learned representations are
already present. The next implementation gap is therefore expressive covariance rather than generic
high-dimensional coverage.

`SpectralMixtureKroneckerMultiTaskGP` is the first implementation target. The existing spectral
mixture family already isolates the data kernel from long-format task covariance, so the Kronecker
variant should inject the same spectral data covariance into `KroneckerMultiTaskGP` without
encoding task identity in `train_X`.

`InfiniteWidthBNNKroneckerMultiTaskGP` is the second implementation target. Its
`InfiniteWidthReLUKernel` is already a reusable covariance module, making the intended model
`K_NNGP(X, X') × K_task` rather than a new neural inference architecture.

Robust extensions remain prototypes until their observation models are explicit. In particular,
heteroskedastic acceptance requires executable evidence that learned task-specific input-dependent
noise changes the response observation covariance. RRP, Student-t, contaminated, and replicate-noise
variants must not be produced by flattening block-design observations into long-format data.

DeepGP × Kronecker remains Hold. This is an implementation-cost decision, not a statistical rejection.

### Phase 2 gate

Phase 2 may proceed with `SpectralMixtureKroneckerMultiTaskGP`. It must preserve
`train_X[..., n, d]` / `train_Y[..., n, m]`, use spectral mixture only for the data covariance,
retain the existing raw-data and MLL contracts, and prove posterior sampling plus representative
scalarized MC acquisition and `optimize_acqf` execution before public registration.


### Phase 10 SAAS Kronecker disposition

Phase 10 closes the planned SAAS implementation slot without adding a runtime model. Phase 9's
`Hold` decision is the accepted implementation result under the current BoTorch dependency.

No public symbol, registry entry, capability flag, generated model-coverage entry, or compatibility
alias named `SaasFullyBayesianKroneckerMultiTaskGP` is introduced. This prevents documentation
from implying a block-design fully Bayesian capability that the runtime does not provide.

The extension matrix also records the expressive Kronecker work completed after the original Phase
1 snapshot: Spectral Mixture and Infinite-width BNN now have public continuous and Mixed
block-design implementations.

A future SAAS Kronecker phase may reopen only under the triggers documented in
`saas_kronecker_design.md`. Until then, ordinary fully Bayesian SAAS support remains the existing
single-task and long-format multi-task families; it must not be presented as Kronecker support.

# Mixed model support audit

## Scope

This document records the repository-wide Mixed-input audit required before the
Mixed model rework. The audit is pinned to `main` commit
`c75a98f25a48384b61cb21caefe8dd4fb9ed725a` (2026-09-19 JST).

The audit treats current `main` as authoritative. Historical PRs are evidence
about prior designs only; they are not treated as current implementation.

The public target remains raw mixed inputs plus `cat_dims`. Structural columns
(task, fidelity, context, hierarchy selectors) are not ordinary categorical
design variables. Model capability and acquisition-search capability are
tracked separately.

## Repository findings

Current `main` exposes Mixed support only for:

- `MixedSingleTaskGP`
- `MixedSingleTaskMultiFidelityGP`
- `MixedReducedGP`
- `MixedPCAGP`, `MixedPLSGP`, `MixedRandomProjectionGP`
- `MixedAutoEncoderGP`, `MixedVAEGP`
- `MixedSupervisedAutoEncoderGP`, `MixedSupervisedVAEGP`

Important audit findings:

1. `src/robotorchan/models/reduced/mixed.py` is a dedicated Mixed model file.
   This conflicts with the new family-colocation rule and should be removed by
   migration, not preserved through aliases.
2. `src/robotorchan/models/mixed_multi_fidelity.py` is also a dedicated Mixed
   model file. Its current implementation subclasses `MixedSingleTaskGP` and
   treats fidelity dimensions as ordinary ordered continuous coordinates. It
   therefore does **not** preserve the specialized
   `SingleTaskMultiFidelityGP` fidelity covariance semantics and requires
   redesign rather than simple relocation.
3. Current reduced Mixed models correctly keep categorical columns out of
   PCA/PLS/RP/AE/VAE reducers and append them after the continuous latent block.
   This is the preferred architecture for frozen reducers.
4. Current `MixedInputLayout.from_cat_dims` rejects negative indices, so the
   reduced Mixed family does not yet satisfy the repository-wide negative
   `cat_dims` contract.
5. Current `MixedSingleTaskMultiFidelityGP` likewise rejects negative
   categorical/fidelity indices.
6. `docs/model-design-guidelines.md` and `docs/mixed-model-support.md` were
   not present on the audited `main`; this file establishes the latter. The
   design-guideline document must be restored from current requirements rather
   than copied blindly from history.
7. No `MixedModelListGP` should be introduced. `ModelListGP` is a container;
   Mixed capability belongs to its children.

## Historical Mixed implementations no longer present on main

The following merged PRs demonstrate prior implementations that were later
removed or superseded. They must be re-evaluated against current model code
before reimplementation:

| Historical PR | Previously implemented decision | Current status |
| --- | --- | --- |
| #14 | `MixedMultiTaskGP`, native categorical data covariance with task-feature exclusion | Missing from main |
| #15 | `MixedKroneckerMultiTaskGP`, native categorical design covariance | Missing from main |
| #16 | `MixedSingleTaskVariationalGP`, native categorical covariance | Missing from main |
| #17 | Mixed multi-fidelity preserving native fidelity kernels on the non-linear-truncated path | Superseded by a semantically weaker current implementation |
| #18 | No `MixedModelListGP`; composition test only | Decision still valid |
| #21 | Fully Bayesian SAAS Mixed wrappers via internal one-hot | Missing; current feasibility docs intentionally defer it |
| #23 | MAP-SAAS Mixed wrappers via internal one-hot | Missing |
| #24 | Mixed robust relevance-pursuit with native mixed covariance | Missing; current PR #167 intentionally re-opened feasibility |
| #25 | Mixed LatentKronecker and Heterogeneous MTGP with native categorical covariance | Missing |
| #26 | Mixed HOGP native design covariance; Mixed OAK via one-hot fallback | Missing |
| #27 | Mixed LCEMGP and hierarchical models; SACGP/LCEAGP left without generic Mixed wrappers | Missing |

These historical designs are inputs to later phases, not restoration targets.

## Standard / structured model support matrix

Strategy values are recommendations for the next implementation phases, not
claims that the implementation already exists.

| Normal model | Mixed model | Main now | Past implementation | Recommended strategy | Audit decision |
| --- | --- | ---: | ---: | --- | --- |
| `SingleTaskGP` | `MixedSingleTaskGP` | Yes | Yes | Native | Keep and re-test; BoTorch-native mixed kernel |
| `MultiTaskGP` | `MixedMultiTaskGP` | No | Yes (#14) | Native | Reimplement against current task-feature handling |
| `KroneckerMultiTaskGP` | `MixedKroneckerMultiTaskGP` | No | Yes (#15) | Native | Reimplement; task identity is in output axis, not X |
| `SingleTaskMultiFidelityGP` | `MixedSingleTaskMultiFidelityGP` | Yes | Yes (#17) | Native | **Redesign current class** to preserve fidelity covariance; structural fidelity dims excluded from cat dims |
| `SingleTaskVariationalGP` | `MixedSingleTaskVariationalGP` | No | Yes (#16) | Native | Reimplement through covariance hook; preserve ELBO contract |
| `SaasFullyBayesianSingleTaskGP` | Mixed SAAS single-task | No | Yes (#21) | One-hot / investigation | NUTS/Pyro path prevents simple post-hoc categorical kernel; dedicated design required |
| `SaasFullyBayesianMultiTaskGP` | Mixed SAAS multi-task | No | Yes (#21) | One-hot / investigation | Same, with task feature kept structural |
| `AdditiveMapSaasSingleTaskGP` | Mixed additive MAP-SAAS | No | Yes (#23) | One-hot | Preserve SAAS covariance/prior semantics; model-owned transform |
| `EnsembleMapSaasSingleTaskGP` | Mixed ensemble MAP-SAAS | No | Yes (#23) | One-hot | Preserve batched MAP-SAAS semantics |
| `RobustRelevancePursuitSingleTaskGP` | Mixed robust relevance-pursuit | No | Yes (#24) | Needs investigation | Current #167 deliberately requires inference/benchmark verification before exposure |
| `LatentKroneckerGP` | `MixedLatentKroneckerGP` | No | Yes (#25) | Native | Replace only X covariance factor; preserve T covariance |
| `HeterogeneousMTGP` | `MixedHeterogeneousMTGP` | No | Yes (#25) | Native | Native per feature subset; task structure remains separate |
| `HigherOrderGP` | `MixedHigherOrderGP` | No | Yes (#26) | Native | Replace design-input covariance factor only |
| `OrthogonalAdditiveGP` | `MixedOrthogonalAdditiveGP` | No | Yes (#26) | One-hot | OAK quadrature is continuous; ordinary categorical kernel would change its mathematics |
| `SACGP` | none by default | No | No | Not applicable / investigation | Decomposition is contextual structure; do not create a nominal Mixed class without a design-variable use case |
| `LCEAGP` | none by default | No | No | Not applicable / investigation | Context embedding metadata is structural, not ordinary `cat_dims` |
| `LCEMGP` | `MixedLCEMGP` | No | Yes (#27) | Native | Design categorical dims may be mixed; task/context features remain structural |
| `HierarchicalConditionalKernelGP` | Mixed hierarchical GP | No | Yes (#27) | Native | Parent/branch selectors are structural and must not overlap `cat_dims` |
| `HierarchicalConditionalKernelMultiTaskGP` | Mixed hierarchical MTGP | No | Yes (#27) | Native | Same plus task-feature remapping |
| `PairwiseGP` | undecided | No | No | Needs investigation | Preference likelihood is compatible in principle, but duplicate consolidation/kernel semantics and mixed search must be verified |
| `ALEBOGP` | none | No | No | Not applicable | Model and search embedding are coupled; categorical discreteness is not preserved by ALEBO projection |
| `ModelListGP` | none | No | No (#18 explicitly avoided it) | Not applicable | Container composes Mixed-capable children |

## High-dimensional / reduced model support matrix

For reduced models, categorical integer codes must never be sent directly into
continuous reducers.

| Model | Reduction / representation | Categorical strategy | Mixed class now | Search compatible | Status |
| --- | --- | --- | ---: | --- | --- |
| `ReducedGP` | arbitrary frozen `InputReducer` | continuous-only reduction + native categorical GP | Yes (`MixedReducedGP`) | Partial | Keep concept; relocate/refactor family code |
| `PCAGP` | PCA | continuous-only PCA + native categorical GP | Yes | Partial | Preferred architecture already present |
| `PLSGP` | supervised PLS | continuous-only PLS + native categorical GP | Yes | Partial | Preferred architecture already present |
| `RandomProjectionGP` | Gaussian RP | continuous-only RP + native categorical GP | Yes | Partial | Preferred architecture already present |
| `AutoEncoderGP` | frozen AE | continuous-only encoder + native categorical GP | Yes | Partial | Current architecture is valid for frozen reducer semantics |
| `VAEGP` | frozen VAE mean | continuous-only encoder + native categorical GP | Yes | Partial | Current architecture is valid for frozen reducer semantics |
| `SupervisedAutoEncoderGP` | frozen supervised AE | continuous-only encoder + native categorical GP | Yes | Partial | Current architecture is valid |
| `SupervisedVAEGP` | frozen supervised VAE | continuous-only encoder + native categorical GP | Yes | Partial | Current architecture is valid |
| `JointEncoderGP` | jointly trained NN + GP | continuous encoder + categorical GP component, model-owned mapping | No | No | Dedicated Mixed joint architecture required |
| `HybridAutoEncoderGP` | joint encoder + reconstruction | continuous encoder + categorical GP component; reconstruction only over intended continuous representation unless explicitly categorical-aware | No | No | Dedicated Mixed joint architecture required |
| `JointVAEGP` | joint VAE + GP | continuous VAE + categorical GP component or categorical-aware encoder | No | No | Dedicated Mixed joint architecture required; preserve MLL gradient flow |
| `OutputPCAGP` | output PCA only | native categorical input GP + output reducer | No | Partial after model support | Needs a composition/generalization design; output reduction itself does not justify treating categories as continuous |
| `OutputPLSGP` | output PLS only | native categorical input GP + output reducer | No | Partial after model support | Same |
| SAAS high-dimensional models | sparse prior | see SAAS rows above | No | No | Do not conflate high-dimensional prior with Mixed support |
| `ALEBOGP` | learned Mahalanobis geometry | none in current design | No | No | Mixed model alone would be misleading; requires Mixed search strategy |
| Hybrid VAE GP | not present on current main | n/a | No | No | Not a current model; do not invent it during audit |

“Partial” search compatibility means the surrogate can consume raw mixed inputs,
but a generic continuous `optimize_acqf` path is insufficient. The search layer
must enumerate/fix categorical assignments or otherwise use a discrete-aware
optimizer.

## Search-layer audit

Current public search strategies are `OriginalSpaceStrategy`,
`LatentSpaceStrategy`, `RandomSearchStrategy`, `ALEBOStrategy`,
`REMBOStrategy`, `HeSBOStrategy`, `BAxUSStrategy`,
`BAxUSThompsonSamplingStrategy`, and `TuRBOStrategy`.

This Phase does not mark any of these as generally Mixed-compatible. A
Mixed-capable surrogate does not make a continuous embedding/trust-region
optimizer Mixed-safe. Later phases must explicitly verify categorical
feasibility, fixed-feature enumeration/alternation, inverse mapping, and
candidate reconstruction. ALEBO in particular is a model-plus-search method and
must not receive a `MixedALEBOGP` label without a discrete-aware search design.

## Phase consequences

The implementation order should follow the 12-phase plan in the project
instruction, with these concrete corrections discovered by the audit:

- Phase 2 must provide one canonical dimension-normalization utility with
  negative-index support and structural-dimension exclusion.
- Phase 2 should also provide reusable model-owned categorical encoding only
  where native categorical covariance is structurally inappropriate.
- Phase 4 must replace the current mixed multi-fidelity approximation rather
  than merely move its file.
- Phases 6-8 should preserve the current successful continuous-only reduction
  principle, but migrate `reduced/mixed.py` into model-family modules and add
  negative-index/state/device contract coverage.
- Joint/Hybrid models must keep raw-space training inputs and gradient flow
  through the encoder; they cannot reuse the frozen `MixedInputReducer`
  lifecycle mechanically.
- Phase 9 should use the old structured-model PRs as design evidence while
  re-checking the current upstream constructors and covariance hooks.
- Phase 10 should keep ALEBO Mixed support out of the public model namespace
  unless search semantics are solved.
- Phase 11 should test composability rather than create a Cartesian product of
  `Mixed × reduced × multitask × multifidelity × robust` classes.

## Phase 1 completion boundary

Phase 1 changes documentation only. No model implementation, compatibility
alias, deprecated wrapper, or monkey patch is introduced. The next phase may
begin from this matrix after re-checking the then-current `main`.


## Phase 2 infrastructure status

Phase 2 establishes one canonical raw-space dimension-normalization contract in
`models/base.py`. Mixed implementations should reuse it instead of implementing
per-model index validation. It supports Python-style negative indices, rejects
duplicates after normalization, and can reject overlap with structural columns.

The existing frozen-reducer Mixed infrastructure now uses the same normalization,
so categorical dimensions remain raw-space indices at the public API while the
reducer continues to fit continuous columns only. Phase 2 intentionally does not
add a generic kernel abstraction or a generic one-hot transform: current
`MixedSingleTaskGP` already delegates native covariance construction to BoTorch,
and the models that require one-hot fallback have different state/inference
requirements. Those transforms should be introduced only in the model-family
phase where they are actually required.


## Phase 3 standard exact / multi-task status

Phase 3 restores native categorical covariance for the current multi-task model
families without restoring historical compatibility helpers. `MixedMultiTaskGP`
and `MixedKroneckerMultiTaskGP` are colocated with their standard counterparts in
`models/multitask.py`.

`MixedMultiTaskGP` keeps the long-format task column structural: `task_feature`
is normalized in raw coordinates, excluded from the data covariance, and rejected
when it overlaps `cat_dims`. `MixedKroneckerMultiTaskGP` needs no structural task
column because task identity remains on the output axis. Both use the shared native
mixed covariance (continuous + categorical + interaction), accept negative raw-space
`cat_dims`, preserve raw training data, and retain the exact-GP `make_mll()` contract.

The existing `MixedSingleTaskGP` remains a thin wrapper over BoTorch's native
`MixedSingleTaskGP`; Phase 3 does not duplicate that upstream implementation.


## Phase 4 multi-fidelity / variational status

Phase 4 replaces the semantically weaker mixed multi-fidelity approximation with a
family-colocated implementation in `models/multi_fidelity.py`. Categorical design
variables use the shared native mixed covariance, while iteration/data fidelity
columns remain structural and are modeled by BoTorch's native fidelity kernels.
Categorical/fidelity overlap is rejected after raw-space negative-index normalization.
The mixed model deliberately requires `linear_truncated=False`: the linear-truncated
kernel owns the complete input covariance and cannot preserve a separate mixed design
covariance without changing the model semantics.

`MixedSingleTaskVariationalGP` is colocated in `models/variational.py` and injects the
shared native mixed covariance through the existing BoTorch covariance hook. Inducing
point allocation, variational strategy, likelihood, posterior behavior, raw-data
retention, and the `VariationalELBO` `make_mll()` contract remain owned by the standard
variational wrapper / BoTorch implementation. Both Phase 4 models accept raw mixed
inputs and Python-style negative `cat_dims`; no compatibility alias or deprecated
wrapper is retained.


## Phase 5 fully Bayesian / MAP-SAAS / robust status

Phase 5 resolves the historical feasibility gates against the current model
constructors rather than restoring old PRs verbatim.

Fully Bayesian SAAS uses a model-owned one-hot representation because BoTorch's
SAAS covariance is constructed inside the Pyro / NUTS model. Raw mixed inputs
remain the public API, observed category values are model state, unseen
categories are rejected, and multi-task task columns remain structural.
Categorical dimensions are never warped. These models remain non-MLL models and
continue to use BoTorch fully Bayesian fitting.

MAP-SAAS likewise keeps its specialized SAAS covariance and priors. Its Mixed
variants therefore use the shared model-owned one-hot InputTransform rather than
replacing the MAP-SAAS covariance with a nominal categorical kernel. The
transform is active for training, evaluation, and fantasization and owns
category state across dtype/device moves and serialization.

Robust relevance pursuit has a different architecture: BoTorch exposes the data
covariance directly while relevance pursuit specializes the likelihood/outlier
path. The Mixed robust wrapper therefore uses the native categorical covariance
(continuous + categorical + interaction) and leaves the relevance-pursuit
likelihood semantics unchanged. This closes the Phase 1 / PR #167 verification
gate with focused posterior, covariance, raw-data, and MLL contract tests.

No compatibility aliases, deprecated wrappers, or monkey patches are introduced.


## Phase 6 classical reduced GP status

PCA, PLS, and Random Projection Mixed wrappers are now colocated with their
standard model families in `models/reduced/base.py`. The shared
`MixedReducedGP` / `MixedInputReducer` infrastructure remains in
`models/reduced/mixed.py`; this is infrastructure rather than a parallel
family module.

For all three classical reducers, raw categorical integer codes never enter the
continuous reducer. `cat_dims` is normalized in the original raw coordinate
space, continuous columns alone are fitted/transformed by PCA / PLS / random
projection, and untouched categorical columns bypass the reducer. The reduced
continuous block plus categorical passthrough block is then modeled by
BoTorch's native MixedSingleTaskGP categorical covariance path.

Reducer lifecycle is unchanged: an unfitted reducer is fitted once at model
construction, an already fitted reducer is reused, state loading reconstructs
latent training inputs from retained raw training data, and public posterior /
conditioning APIs continue to accept original mixed-space inputs.

This phase intentionally does not change AE/VAE/supervised neural or joint /
hybrid representations; those are handled by Phases 7 and 8.


## Phase 7 frozen neural reduced GP status

Frozen AutoEncoder, VAE, supervised AutoEncoder, and supervised VAE Mixed
wrappers are colocated with their corresponding standard neural model families.
The shared `MixedReducedGP` infrastructure remains in `reduced/mixed.py`.

These models use the same categorical architecture as the classical reduced
family: categorical integer codes are never treated as continuous neural
features. The encoder/reducer is fitted only on continuous raw columns;
categorical columns bypass the frozen representation unchanged and enter the
downstream native Mixed GP categorical covariance. Supervised reducers receive
the outcome target during reducer fitting but still receive only continuous
input columns.

Negative `cat_dims`, raw training-data retention, raw-space posterior calls,
and the frozen reducer lifecycle are preserved. This phase does not introduce
categorical embeddings inside the neural encoder because that would define a
different representation model and would mix categorical semantics into the
continuous latent block.

JointEncoderGP, HybridAutoEncoderGP, and JointVAEGP are intentionally excluded:
their representation parameters participate in GP MLL optimization and require
the dedicated gradient-preserving Mixed design in Phase 8.


## Phase 8 joint / hybrid representation GP status

JointEncoderGP, HybridAutoEncoderGP, and JointVAEGP now have explicit Mixed
counterparts. Their categorical architecture differs deliberately from an
ordinary neural encoder over the full raw tensor.

Continuous raw columns alone enter the learnable encoder. Raw categorical
columns bypass the representation network and are appended after the latent
continuous coordinates. The GP covariance is a native mixed covariance over
the learned continuous latent block and categorical passthrough block. This
keeps category codes out of continuous neural geometry while preserving the
gradient path from exact GP marginal likelihood through the continuous
encoder.

MixedHybridAutoEncoderGP reconstructs only continuous raw columns; categorical
codes are not assigned an artificial Euclidean reconstruction loss.
MixedJointVAEGP likewise defines q(z|X) and reconstruction/KL losses only for
continuous inputs while the native categorical GP component handles category
identity.

All public prediction and training interfaces remain in original raw mixed
space, negative cat_dims are normalized there, and raw training data are
retained. These classes are intentionally distinct from the frozen neural
Mixed models in Phase 7.


## Phase 9 structured GP status

Structured Mixed support is restored against the current shared Mixed
infrastructure rather than by reviving historical private helper APIs.

- `MixedLatentKroneckerGP`: native categorical covariance is applied only to
  the design `X` factor; the `T` factor keeps LatentKroneckerGP semantics.
- `MixedHeterogeneousMTGP`: each heterogeneous feature subset receives a
  native continuous/categorical/mixed kernel according to its active features.
- `MixedHigherOrderGP`: only the design-input covariance factor is mixed;
  tensor-output Kronecker factors are unchanged.
- `MixedOrthogonalAdditiveGP`: model-owned one-hot fallback is used because
  the continuous Gauss-Legendre orthogonalization measure of OAK is not a
  categorical measure.
- `MixedLCEMGP`: ordinary design categories use native mixed covariance;
  task/context embedding features remain structural and separate.
- hierarchical single-task and multitask Mixed variants preserve hierarchical
  parent selectors as structural dimensions and reject overlap with
  `cat_dims`.

SACGP and LCEAGP do not receive generic Mixed wrappers in this phase: their
contextual decomposition / embedding semantics already assign special meaning
to contextual categorical information, so treating those coordinates as
ordinary categorical design variables would conflate two different roles.

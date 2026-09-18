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

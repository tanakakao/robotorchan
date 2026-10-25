# Acquisition theory–implementation mapping: Phase 3 inventory

## Sources and evidence boundary

- Baseline: `main` after the Phase 2 exit-check merge `b83afcc67e38d2b4dcc3c3a3ab794063f61705f6`.
- Public local exports: `src/robotorchan/acquisition/__init__.py`.
- Capability metadata: `src/robotorchan/acquisition/registry.py`.
- Repeatable signature inventory: `scripts/audit_acquisition_api.py`.
- Cross-layer compatibility: `acquisition/optimizer_compatibility.py`.
- Composition and semantic bridges: `acquisition/composition.py`.

This is a **source-inspected static inventory**, not a claim that
each acquisition has a numerically verified implementation or compatible
optimizer. Phase 5 adds per-method theory mapping; Phase 8 adds independent
mathematical checks.

## Distinct implementation paths

| Method group | Public/implementation ownership | Mathematical object or route | Evidence and limitation |
| --- | --- | --- | --- |
| `PosteriorVariance`, `PosteriorStd`, `Straddle`, `RandomizedStraddle`, `BoundaryVariance` | Locally exported classes; registered as active learning | GP marginal moments and boundary utilities | Registry declares `max_q=1`; verify individual formulas and output-axis semantics in Phase 5/8 |
| `ExpectedPredictiveInformationGain` | Local active-learning acquisition | Joint Gaussian information criterion | Registry explicitly requires single output and joint Gaussian posterior |
| `PredictiveEntropy`, `MarginUncertainty`, `ProbabilityVariance`, `BALD`, `LatentStraddle` | Local classification AL classes | Class probability / label uncertainty / latent score are **distinct targets** | Registered `max_q=1`, single output; probabilities cannot silently be replaced by latent GP moments |
| `qLogExpectedImprovement`, `qLogNoisyExpectedImprovement`, `qProbabilityOfImprovement`, `qUpperConfidenceBound`, `qSimpleRegret` | BoTorch-native; **metadata only**, no local class wrapper | Monte Carlo BO objectives | Presence in registry does not expose these as local public imports or verify optimizer capability |
| `qKnowledgeGradient`, `qMultiFidelityKnowledgeGradient` | BoTorch-native | Fantasy-based one-shot lookahead | Registry flags `one_shot=True`, `requires_fantasize=True`; multi-fidelity KG additionally needs fidelity-aware model |
| `qLogExpectedHypervolumeImprovement`, `qLogNoisyExpectedHypervolumeImprovement`, `qLogNParEGO` | BoTorch-native | Multi-objective / scalarized Monte Carlo BO | Constraints/multi-objective flags are metadata; inspect actual objective and partitioning in Phase 5 |
| `make_qei_acquisition`, `make_qnei_acquisition`, `make_qehvi_acquisition`, `make_qnehvi_acquisition` and `make_*constrained*` | Local construction helpers in `acquisition/composition.py`, not acquisition subclasses | Semantic model/objective/constraint adapters feeding BoTorch acquisitions | Helpers must not be counted as additional independent acquisition algorithms |
| `select_thompson_candidates` | `acquisition/sampling` exported helper | Posterior-sample selection | Search procedure, not a registered standalone MC acquisition class |

`scripts/audit_acquisition_api.py` distinguishes local acquisition
subclasses, supporting exports, composition helpers, and native BoTorch
registry entries. Its `gaps` keys reflect **registry/export mismatch**
rather than unsupported algorithms. The registry is not the exhaustive
inventory of BoTorch methods, and alternative choices (including native
upstream call sites) must be traced in Phase 3 before declaring coverage.

## Acquisition/optimizer interaction

`check_acquisition_optimizer_compatibility` checks registered `max_q`
and whether a one-shot acquisition is routed to `MIXED_ONE_SHOT`.
It is not an optimizer implementation, nor a proof of all nonlinear
candidate-constraint compatibility. In particular, a one-shot augmented
batch cannot safely be passed to the ordinary mixed optimizer as if its
categorical assignment were row-independent.

## Fixed Phase 3 completion criteria

1. Inventory local acquisition exports, capability registry entries,
   composition helpers and BoTorch-native call sites without conflation.
2. Inventory named dispatch optimizers, direct low-level backends, search
   strategies and standalone optimizer utilities beyond dispatch.
3. Trace candidate constraints (linear, nonlinear, intra-/inter-point),
   outcome constraints, mixed domains, q-batch and one-shot boundaries.
4. Resolve concrete implementation modules and representative tests,
   including unsupported and unverified behavior.
5. Run relevant inventory and CI checks, then make an explicit completion
   decision. Numerical identities belong to Phases 8–9.

## Inspected acquisition registry and assertion evidence

The following tests were inspected directly in source. They are named
assertion evidence; a successful repository CI is separate execution
evidence, and neither proves an independent mathematical derivation.

| Source of assertions | Confirmed behavior | Not established |
| --- | --- | --- |
| `tests/acquisition/test_api_inventory.py` | Audit inventory is JSON serializable; all registered local acquisition classes are exported; BoTorch-native entries are metadata-only; custom parameter names such as `output_index`, `target`, and `num_samples` survive introspection | Exact formulas, numerical convergence or completeness of the upstream BoTorch catalogue |
| `tests/acquisition/test_api_inventory.py` | `PosteriorVariance` produces finite `q=1` scores and rejects `q=2`; `PredictiveEntropy` returns finite classification scores and rejects `q=2` | Calibration of either score or equivalence of predictive label entropy to latent Gaussian entropy |
| `tests/acquisition/test_optimizer_compatibility.py` | Ordinary mixed route rejects qKG and multi-fidelity qKG; mixed one-shot route accepts them; non-one-shot methods reject one-shot routing; registered AL `q=1` limits are enforced; normal qLogEI accepts `q=3` | Success of an actual augmented one-shot optimization, feasibility of a candidate, compatibility with all concrete backends |

The compatibility test uses `CandidateOptimizer.MIXED_ONE_SHOT` as a
**semantic route**. It must not be confused with an additional dispatcher
optimizer name; the operational mixed one-shot implementation belongs to
`optim/one_shot/`. These targeted tests are distinct from later
formula-level Phase 8 checks.

## Phase 5 — regression active-learning acquisition mappings (source inspected)

This section maps the local `q=1` regression AL family
to the mathematical score actually computed by the
implementation. Let `mu(x)` and `v(x)` denote the
selected scalar output's posterior mean and variance,
`sigma(x)=sqrt(max(v(x),0))`, and `t` the target level.
All listed scores are **maximized**.

| Registered acquisition | Mathematical score / interpretation | Concrete implementation and ownership | Tests and verification boundary |
| --- | --- | --- | --- |
| `PosteriorVariance` | `v(x)`; marginal uncertainty sampling (not integrated batch variance). | `acquisition/active_learning/variance.py::PosteriorVariance.forward`; **Implemented** scalar posterior extraction, **Delegated** posterior moments to the model. | `tests/acquisition/test_api_inventory.py` and `tests/acquisition/test_public_api.py`; independent variance identity remains Phase 8. |
| `PosteriorStd` | `sigma(x)`; monotone in variance for nonnegative variance, but numerically different scale. | `variance.py::PosteriorStd.forward`; **Extended** `PosteriorVariance` with nonnegative clamp and square root. | Public API/AL contracts; independent numerical checks deferred. |
| `Straddle` | `beta * sigma(x) - abs(mu(x)-t)`, `beta >= 0`. Level-set boundary search balances uncertainty and proximity. | `acquisition/active_learning/straddle.py::Straddle.forward`; **Implemented** local formula, **Delegated** posterior moments. | `tests/acquisition/test_level_set_learning.py`; coefficient convention must not be confused with UCB's `sqrt(beta)`. |
| `RandomizedStraddle` | `sqrt(B) * sigma(x) - abs(mu(x)-t)`, where `B=-2 log(1-U)`, `U ~ Uniform(0,1)`; one random coefficient is cached per selection round until `resample()`. | `acquisition/active_learning/randomized_straddle.py`; **Extended** local Straddle with cached stochastic coefficient; no BoTorch-native acquisition delegation. | `tests/acquisition/test_randomized_straddle.py`; statistical distribution and reproducibility beyond named assertions require Phase 8. |
| `BoundaryVariance` | `v(x) * exp(-0.5 * ((mu(x)-t) / max(sigma(x),eps))^2)`; local Gaussian target-proximity heuristic, **not** a named upstream acquisition. | `straddle.py::BoundaryVariance.forward`; **Implemented** local heuristic and inherits scalar-output selection, not Straddle's `beta` weighting. | `tests/acquisition/test_boundary_variance_contract.py`; mathematical optimality claims are unsupported. |

**Shared domain and shape restrictions:** All five
registry entries declare `max_q=1` and require scalar
posterior output or an explicit `output_index`.
Structured-output posteriors without scalarization and
ensemble posteriors are rejected by the inspected local
implementations. The BoTorch
`average_over_ensemble_models` decorator does **not**
override those explicit ensemble guards. These are
per-candidate marginal scores, not joint q-batch
information criteria.

**Canonical theory and usage:** see
`docs/theory/acquisition/10_level_set.md`,
`docs/optimization/level_set_learning.md`,
`docs/optimization/boundary-variance.md` and
`examples/notebooks/28_regression_active_learning.ipynb`.
The named files are source/test references, not evidence
that every combination has been numerically verified.

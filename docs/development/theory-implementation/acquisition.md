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

## Phase 5 — classification AL and Gaussian EPIG (source inspected)

Classification AL distinguishes **predictive class
probability**, **label uncertainty** and **latent Gaussian
uncertainty**. These are not interchangeable posterior
quantities. The following five acquisitions are local
`AcquisitionFunction` subclasses, not BoTorch-native
classification acquisitions.

| Registered acquisition | Maximized score / target | Source ownership and evidence boundary |
| --- | --- | --- |
| `PredictiveEntropy` | `-sum_c p_c(x) log(p_c(x))`, with `p` returned by `predict_proba`; predictive **label entropy**. | `active_learning/classification.py::PredictiveEntropy`; **Implemented** score, **Delegated** class probabilities to the classification model. |
| `MarginUncertainty` | `1 - (p_(1)(x)-p_(2)(x))` for the two highest class probabilities; smaller margin scores higher. | `classification.py::MarginUncertainty`; **Implemented** local top-two operation, **Delegated** `predict_proba`. Requires at least two classes. |
| `ProbabilityVariance` | Mean over classes of the model-provided posterior variance of class probabilities; **not** variance of latent GP outputs. | `classification.py::ProbabilityVariance`; **Delegated** `probability_variance(X,num_samples)`, locally averages class axis. Requires `num_samples>=2`. |
| `BALD` | Model-provided mutual information between labels and latent parameters, commonly `H[Y|D,x] - E_theta H[Y|theta,x]`; not merely predictive entropy. | `classification.py::BALD`; **Delegated** `mutual_information(X,num_samples)`. Requires `num_samples>=2`; estimator depends on the model. |
| `LatentStraddle` | `beta * sqrt(max(v_latent(x),0)) - abs(mu_latent(x))`; zero latent decision boundary. | `classification.py::LatentStraddle`; **Implemented** local score, **Delegated** `latent_posterior`. Binary only, finite `beta>=0`. |

All five explicitly enforce `q=1`, and the registry
declares single-output classification targets. The
protocol checks the required prediction method at
construction, not its numerical calibration. Named
assertion evidence:
`tests/acquisition/test_classification_active_learning.py`,
`tests/acquisition/test_classification_calibration_integration.py`,
and `tests/acquisition/test_public_api.py`. The
classification-specific theoretical distinction is
documented in
`docs/theory/acquisition/classification-active-learning.md`.

### ExpectedPredictiveInformationGain (regression)

`ExpectedPredictiveInformationGain` is a separate,
locally implemented **Gaussian regression** AL
acquisition, not classification BALD or the general
non-Gaussian EPIG objective. For each candidate `x`
and finite target point `z_i`, it obtains a joint
two-point Gaussian posterior and computes

`I_i(x) = -0.5 * log(1-rho_i(x)^2)`,

where `rho_i^2 = cov(x,z_i)^2 /
((var(x)+noise)*(var(z_i)+noise))`. It maximizes
the arithmetic mean of `I_i` or a normalized
nonnegative weighted mean over the finite target set.
The correlation is clamped below one for stability,
and supplied observation noise is clamped nonnegative.

- **Implementation:** `active_learning/epig.py`;
  **Implemented** Gaussian information formula and
  finite target integration, **Delegated** joint
  covariance to the model's posterior.
- **Requirements:** `q=1`, single-output joint
  Gaussian `posterior.mvn.covariance_matrix`;
  ensemble posterior unsupported; `target_X` must
  be nonempty and two-dimensional.
- **Evidence:** `tests/acquisition/test_epig.py`
  and `docs/optimization/epig.md`; Gaussian
  assumptions and numerical identities still require
  independent Phase 8 verification.

**Phase boundary:** These are source-level formula and
ownership mappings, not proof of model calibration,
information-theoretic optimality or numerical identity.

## Phase 5 — single-objective Monte Carlo BO (source inspected)

These acquisition functions maximize **utility** after
the problem's objective direction has been normalized
to maximization. Let `F(Y)` be the scalar objective
on posterior samples and `f_best` the incumbent in
the same oriented scale.

| Method | Mathematical object / source-level route | Ownership and limitations |
| --- | --- | --- |
| `qExpectedImprovement` (qEI) | MC expectation of `max(max_i F(Y_i)-f_best,0)`; `composition.py::make_qei_acquisition` constructs the upstream class using `best_f` and an objective bridge. | **Delegated** acquisition formula to BoTorch; **Extended** local semantic direction and output bridge. Helper supports one unconstrained `RegressionObjective`. |
| `qNoisyExpectedImprovement` (qNEI) | MC expected improvement relative to jointly sampled baseline; `composition.py::make_qnei_acquisition` passes `X_baseline`, `prune_baseline`, `cache_root` and the objective bridge. | **Delegated** to BoTorch; local helper requires nonempty 2-D baseline, one unconstrained regression objective. No `best_f` parameter. |
| `qLogExpectedImprovement` (qLogEI) | Log-domain expected improvement for improved numerical behavior; constructed directly in `benchmarks/botorch_strategy.py`. | **Delegated** to BoTorch. Registered metadata is not a local wrapper; no general-purpose `make_qlogei_acquisition` composition helper is exported. |
| `qLogNoisyExpectedImprovement` (qLogNEI) | Log-domain noisy expected improvement; benchmark strategy supplies `X_baseline=X`. | **Delegated** to BoTorch; distinguish from the composition module's ordinary qNEI constructor. |
| `qProbabilityOfImprovement` (qPI) | MC probability that the best sampled batch utility exceeds `f_best`; benchmark strategy passes `best_f=train_Y.max()`. | **Delegated** to BoTorch; registry metadata does not implement the score. |
| `qUpperConfidenceBound` (qUCB) | MC upper-confidence utility; benchmark strategy passes `beta=0.2`. | **Delegated** to BoTorch; do not conflate this MC acquisition with analytic single-point UCB or Straddle. |
| `qSimpleRegret` | MC expected best sampled objective value, with no improvement threshold. | **Delegated** to BoTorch by registry metadata; no use of this class was established in the inspected benchmark strategy. Do not claim an executable benchmark route from registration alone. |

The benchmark route
`benchmarks/botorch_strategy.py::botorch_gp_candidates`
currently accepts `qEI`, `qLogEI`, `qNEI`,
`qLogNEI`, `qPI` and `qUCB`; it requires a
continuous, unconstrained, single-objective problem.
It orients training targets by the problem direction,
fits a BoTorch `SingleTaskGP` (or supplied factory)
and delegates candidate optimization to
`botorch.optim.optimize_acqf`. The fixed
`beta=0.2` in this benchmark is an experiment
configuration, not a universal qUCB default.

**Composition vs. algorithm:** The public
`make_qei_acquisition` and
`make_qnei_acquisition` are construction helpers,
not independent acquisition algorithms. They resolve
heterogeneous model ownership, apply semantic objective
direction and pass the resulting BoTorch objective to
native qEI/qNEI. Their continuous and classification
constraint variants are distinct composition policies
and will be mapped under constrained BO rather than
silently attributed to unconstrained EI.

**Evidence and limits:** Source references:
`src/robotorchan/acquisition/registry.py`,
`src/robotorchan/acquisition/composition.py`,
`src/robotorchan/benchmarks/botorch_strategy.py`;
tests:
`tests/acquisition/test_standard_bo_acquisition.py`,
`tests/acquisition/test_phase5_native_contracts.py`,
`tests/acquisition/test_phase5_native_integration.py`.
See `docs/optimization/standard_acquisition.md`
for usage. Exact MC estimator equivalence, log-domain
stability and noisy-baseline behavior remain Phase 8
numerical verification topics.

## Phase 5 — multi-objective MC Bayesian optimization (source inspected)

For an objective vector oriented to maximization,
hypervolume improvement measures the gain in
dominated volume relative to a **directed** reference
point. These are BoTorch algorithms; the local
composition functions are semantic construction
bridges, not reimplementations of hypervolume.

| Method | Mathematical object | Concrete route and ownership |
| --- | --- | --- |
| `qExpectedHypervolumeImprovement` (qEHVI) | Expected batch hypervolume improvement over a reference point and a nondominated partition. | `composition.py::make_qehvi_acquisition` returns native BoTorch `qExpectedHypervolumeImprovement`; **Delegated** hypervolume computation, **Extended** directed multi-output bridge and argument validation. |
| `qNoisyExpectedHypervolumeImprovement` (qNEHVI) | MC noisy expected hypervolume improvement using a sampled baseline. | `composition.py::make_qnehvi_acquisition` returns native BoTorch `qNoisyExpectedHypervolumeImprovement`; **Delegated** noisy hypervolume algorithm, **Extended** directed bridge and baseline validation. |
| `qLogExpectedHypervolumeImprovement` (qLogEHVI) | Log-domain expected hypervolume improvement. | `registry.py` lists native BoTorch capability metadata; tests/examples reference the upstream class. No local `make_qlogehvi_acquisition` composition helper was found in the inspected public exports. |
| `qLogNoisyExpectedHypervolumeImprovement` (qLogNEHVI) | Log-domain noisy expected hypervolume improvement. | `registry.py` lists native BoTorch capability metadata; no local `make_qlognehvi_acquisition` helper is exported. Do not equate registry support with a composed runtime route. |
| `qLogNParEGO` (scalarized log-NEI policy) | Random-weight multi-objective scalarization followed by log-domain noisy improvement. | `composition.py::make_qlognparego_acquisition` **constructs native `qLogNoisyExpectedImprovement`** with a directed Chebyshev `GenericMCObjective`. Weights are supplied by the caller; this helper does **not** resample them or instantiate a distinct BoTorch `qLogNParEGO` class. |

### Verified construction contracts

- **qEHVI:** `make_botorch_multiobjective_bridge`
  requires compatible objective outputs from one
  shared regression posterior. `ref_point` must be
  finite, have one component per objective and match
  the provided `BoxDecomposition.ref_point`
  exactly. The caller supplies partitioning already
  constructed in the directed objective space.
- **qNEHVI:** `X_baseline` is a nonempty `n x d`
  tensor of **input locations**, not Pareto objective
  values; `ref_point` must be finite and
  dimensionally consistent. `prune_baseline` and
  `cache_root` are forwarded to native BoTorch.
- **qLogNParEGO composition:** `X_baseline` and
  `Y_baseline` have aligned rows; `Y_baseline`
  must be finite floating-point values already
  oriented to maximization. `weights` must be
  finite, nonnegative, nonzero in sum and match the
  objective count. Chebyshev scalarization is built
  with BoTorch `get_chebyshev_scalarization`.
- These three composition helpers reject feasibility
  constraints; constrained qEHVI is a separate policy
  documented under the constrained acquisition
  mapping. There is no implicit cross-model joint
  posterior reconstruction.

**Evidence:** `src/robotorchan/acquisition/composition.py`,
`src/robotorchan/acquisition/registry.py`,
`tests/acquisition/test_multiobjective.py`,
`tests/acquisition/test_phase5_native_integration.py`,
`tests/acquisition/test_composition.py`, and
`docs/optimization/multiobjective.md`.
Source/contract evidence does not establish independent
hypervolume estimator accuracy, reference-point
sensitivity or log-domain numerical equivalence;
those belong to Phase 8.

## Phase 5 — outcome-constrained BO composition (source inspected)

These policies impose **outcome feasibility** on
posterior predictions; they are not candidate-space
linear/nonlinear constraints passed to an optimizer.
Let `A(X)` denote the native acquisition, and
`p_j(x)` a classifier's posterior-predictive
probability of feasibility. Continuous outcome
constraints are represented as sample-wise residuals
in the same BoTorch posterior as the objective;
classifier feasibility is a separate marginal factor.

| Local constructor | Native acquisition and constraint semantics | Ownership and limits |
| --- | --- | --- |
| `make_continuous_constrained_qei_acquisition` | `qExpectedImprovement` with BoTorch `ConstrainedMCObjective` and same-posterior continuous residuals. | **Extended** objective/constraint bridging; **Delegated** MC improvement and sample feasibility to BoTorch. Exactly one regression objective, at least one continuous constraint. |
| `make_mixed_constrained_qei_acquisition` | Continuous residuals enter `ConstrainedMCObjective`; separate classifier PoFs weight the resulting qEI via `FeasibilityWeightedAcquisition`. | **Extended** local hybrid composition; requires both constraint kinds and distinct classification outputs. |
| `make_continuous_constrained_qnei_acquisition` | `qNoisyExpectedImprovement` with `ConstrainedMCObjective` and a nonempty `X_baseline`. | **Extended** same-posterior continuous constraints; **Delegated** noisy improvement to BoTorch. |
| `make_multiple_learned_constrained_qnei_acquisition` | Native qNEI with optional continuous residual constraints, then independent classifier PoF weighting when classifiers are present. | **Extended** heterogeneous composition. Marginal classifier probabilities are not a sampled joint posterior or baseline feasibility correction. |
| `make_constrained_qehvi_acquisition` | Native `qExpectedHypervolumeImprovement` with sample-wise continuous `constraints` and optional external classifier PoF factors. | **Extended** multi-objective feasibility bridge; **Delegated** HV improvement to BoTorch. Requires directed `ref_point` matching partitioning and `eta>0`. |
| `make_deterministic_pof_acquisition` | Multiplies an existing objective acquisition by one posterior-predictive PoF through `FeasibilityWeightedAcquisition`. | **Implemented** local weighting adapter; does not create an independent EI/HV acquisition or joint classifier samples. |

### Interpretation and boundaries

- For the default `q_reduction="product"`,
  classifier feasibility weighting uses a product
  over the q candidates. This is a **factorization
  assumption** across marginal feasibility terms,
  not a proof that outputs or candidates are
  probabilistically independent.
- The composition constructors reject unsupported
  feasibility representations, repeated classifier
  outputs, and continuous constraints whose model
  entry differs from the objective's shared
  posterior. They do not invent cross-entry
  covariance.
- `infeasible_cost` controls BoTorch
  `ConstrainedMCObjective` for continuous qEI/qNEI
  routes. It is not equivalent to multiplying a
  native acquisition by a classifier's mean PoF.
- `make_sample_classification_feasibility_bridge`
  is a separate capability-gated path requiring
  epistemic probability samples. The deterministic
  PoF routes above do not silently use that path.
- No local constrained qNEHVI constructor is
  exported in the inspected composition API.
  Constrained qEHVI support does not imply
  constrained qNEHVI support.

**Evidence:** `src/robotorchan/acquisition/composition.py`,
`src/robotorchan/acquisition/classification_constraints.py`,
`tests/acquisition/test_composition.py`,
`tests/acquisition/test_phase5_native_integration.py`,
`tests/benchmarks/test_heterogeneous_classification_constrained_e2e.py`.
These sources establish construction paths and
named test evidence; statistical validity of
factorization, calibration, and numerical constrained
acquisition identities remain Phase 8 checks.

## Phase 5 — information-theoretic and lookahead acquisitions (source inspected)

The following methods optimize different **information
targets**. Their presence in theory documentation
does not imply a local acquisition class, registry
entry, or optimizer route.

| Method | Theory / optimization target | Implementation route and status |
| --- | --- | --- |
| `qKnowledgeGradient` (qKG) | One-step value of information: expected improvement in the optimal value after a fantasy observation. | **Delegated** to native `botorch.acquisition.knowledge_gradient.qKnowledgeGradient`; registry marks `one_shot=True`, `requires_fantasize=True`. No local wrapper. |
| `qMultiFidelityKnowledgeGradient` | Fantasy-based lookahead across fidelity levels, with a fidelity-aware model and appropriate cost/value policy. | **Delegated** to native BoTorch; registry additionally declares `requires_multi_fidelity=True`. Capability metadata alone does not construct a cost-aware policy. |
| `qMultiStepLookahead` | Multi-stage fantasy decision tree; optimizes future decisions jointly rather than only one-step value. | **Delegated** directly to BoTorch per `docs/optimization/lookahead.md`; not a named local registry entry in the inspected `registry.py`. |
| `qMaxValueEntropy` (MES) | Information gain about the maximum objective value, approximated using a finite `candidate_set`. | **Delegated** directly to BoTorch; `tests/acquisition/test_information_theoretic.py` checks finite scores with a local `SingleTaskGP`. No local wrapper/registry entry. |
| `qLowerBoundMaxValueEntropy` (GIBBON) | Lower-bound approximation to max-value entropy search. | **Delegated** directly to BoTorch; same native integration test checks finite values. No local wrapper/registry entry. |
| Predictive Entropy Search (PES) | Information about the optimizer location. | **Unverified** as a specific executable route in the inspected local registry/tests; theoretical coverage does not establish a supported public constructor. |
| Joint Entropy Search (JES) | Joint information about optimizer location and optimum value. | **Unverified** as a specific executable route in the inspected local registry/tests; no local wrapper is claimed. |

### One-shot and information-target boundaries

- qKG's optimization input contains `q` actual
  candidates **plus fantasy decision variables**.
  Native BoTorch initialization and candidate
  extraction preserve that augmented-batch
  contract. The registered compatibility check
  rejects ordinary mixed optimization for one-shot
  acquisitions and permits the semantic
  `MIXED_ONE_SHOT` route; this check is not an
  end-to-end optimizer execution.
- `qMultiStepLookahead` requires explicit
  augmented initial conditions or an appropriate
  `ic_generator` when the default initializer
  does not account for the full fantasy tree.
  Its computational cost grows with horizon and
  fantasy branching.
- MES/GIBBON require a finite candidate set to
  approximate optimum-value information; their
  finite-output tests do not verify information
  estimates or optimization performance.
- The local regression
  `ExpectedPredictiveInformationGain` computes
  Gaussian information between candidate and
  **target observations**. Classification `BALD`
  targets label/parameter mutual information.
  Neither is interchangeable with KG, MES, PES,
  or JES.
- A multi-fidelity model's availability alone
  does not prove a complete cost-aware KG
  optimization path.

**Evidence:** `src/robotorchan/acquisition/registry.py`,
`src/robotorchan/acquisition/optimizer_compatibility.py`,
`tests/acquisition/test_lookahead.py`,
`tests/acquisition/test_optimizer_compatibility.py`,
`tests/acquisition/test_information_theoretic.py`,
`docs/optimization/lookahead.md`,
`docs/optimization/multifidelity-cost-aware.md`,
`docs/optimization/information_theoretic.md`.
Numerical information identities and fantasy
optimization behavior remain Phase 8.

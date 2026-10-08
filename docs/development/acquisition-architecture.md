# Acquisition architecture

robotorchan follows a **BoTorch-first** acquisition design.

## Scope

The acquisition package contains only behavior that adds robotorchan-specific value.
BoTorch acquisition classes that already satisfy the required contract are imported and
constructed directly from BoTorch. robotorchan does not create aliases, compatibility
wrappers, or duplicated implementations merely to provide a unified namespace.

Current robotorchan-owned acquisition code covers:

- regression active-learning and level-set criteria that are not provided directly by BoTorch;
- compatibility checks between registered model and acquisition capabilities;
- acquisition / optimizer semantic compatibility checks;
- posterior-sampler selection from registered model capabilities;
- non-GP Monte Carlo acquisition validation;
- finite-set Thompson-style candidate selection.

The acquisition registry is metadata for workflow recommendation and compatibility dispatch.
It does not replace BoTorch constructors and is not a string-based acquisition factory.

## Ownership rules

An acquisition implementation belongs in `robotorchan.acquisition` only when at least one
condition holds:

- robotorchan adds model-specific validation that BoTorch does not provide;
- the method itself is not provided by the supported BoTorch version;
- robotorchan needs additional semantics that cannot be expressed by normal BoTorch
  constructor arguments.

Otherwise users should instantiate the BoTorch acquisition directly. BoTorch-native
acquisitions may still have registry metadata when workflow dispatch needs to reason about
their capabilities.

## Active learning

`active_learning/` owns robotorchan acquisition implementations for regression active
learning and level-set estimation. These classes follow BoTorch acquisition conventions so
that normal BoTorch optimizers can consume them when their semantics permit.

Classification active-learning criteria are robotorchan-owned when they add semantics not
provided directly by BoTorch. They remain separate from Bayesian optimization
outcome-constraint composition.

## Capability and compatibility metadata

`capabilities.py` defines acquisition metadata such as posterior requirements, batch
semantics, multi-output support, fantasization requirements, and one-shot behavior.

`registry.py` records both robotorchan-owned acquisitions and selected BoTorch-native
acquisitions needed by workflow recommendation or compatibility dispatch. Registry entries
describe capabilities; they do not wrap or reconstruct the BoTorch implementation.

`compatibility.py` checks model / acquisition compatibility using model and acquisition
registry metadata. `optimizer_compatibility.py` separately checks whether an optimization
strategy can preserve acquisition semantics such as one-shot augmented batches.

Candidate/input-space feasibility constraints remain an optimizer responsibility and must
not be inferred from acquisition output-constraint metadata.

## Sampling responsibilities

Sampling has two distinct responsibilities and they intentionally remain separate:

- `samplers.py` selects a native BoTorch `MCSampler` from registered posterior-sampling
  capabilities. It is capability dispatch, not a candidate-generation algorithm.
- `sampling/` owns posterior-sampling candidate-selection algorithms. For example,
  `sampling/thompson.py` uses BoTorch `MaxPosteriorSampling` to select candidates from a
  finite set.

Do not merge these responsibilities merely because both involve posterior samples.

## Non-GP surrogates

Non-GP surrogates use BoTorch Monte Carlo acquisition functions when their posterior
contract supports sampling. `validate_non_gp_acquisition` validates the acquisition and
sampler against registered posterior-sampling capabilities rather than assuming every
non-GP model is an empirical ensemble.

`make_non_gp_acquisition` remains a small construction helper because it combines
construction with robotorchan-specific compatibility validation. It must not grow into a
general acquisition factory.

## Heterogeneous semantic composition

The semantic layer owns model-specific interpretation of heterogeneous outputs. Acquisition
composition consumes semantic representations instead of branching on concrete model classes.

The semantic boundary distinguishes feasibility representations by the meaning of their
runtime values, not merely by tensor shape:

- **sample-residual feasibility**: a callable over outcome samples with values <= 0 denoting feasibility;
- **posterior-predictive PoF**: deterministic P(feasible | X, D), already marginalized over
  predictive uncertainty;
- **probability-residual feasibility**: a deterministic residual from thresholding a predictive
  probability, again using <= 0 as the feasibility convention;
- **sample-wise feasibility**: `SampleProbabilityOfFeasibility` retains an explicit
  Monte Carlo sample dimension and is derived from classifier
  `sample_class_probabilities()` rather than deterministic mean PoF;
- **hard feasibility**: a Boolean/indicator decision, which is distinct from both residuals
  and probabilities and must only be introduced by an explicit policy that requires it.

These representations must not be silently converted into one another. In particular,
thresholding mean PoF does not create sample-wise feasibility, and testing a sample residual
does not create a posterior-predictive PoF. Regression outcome constraints naturally expose
sample residuals, while classification feasibility can retain probability semantics.

The composition boundary is split into four responsibilities:

1. `ProblemSemantics` declares user intent without acquisition-specific behavior.
2. `resolve_acquisition_composition()` resolves output ownership and preserves the semantic
   objective and feasibility representations in an immutable `AcquisitionCompositionPlan`.
3. Later composition policies adapt compatible bindings to native BoTorch objective,
   constraint, sampler, and acquisition constructor interfaces.
4. The resulting BoTorch acquisition remains responsible for acquisition evaluation and is
   passed to the existing optimization layer.

The resolved plan is descriptive rather than executable: it must not create a shared posterior,
choose a sampler, aggregate feasibility, or instantiate an acquisition function. Those choices
depend on the requested acquisition family and predictive capabilities and belong to later
composition policies.

Standard Bayesian optimization remains BoTorch-native. Semantic composition should follow the
smallest sufficient integration level:

1. use a native BoTorch constructor directly when semantic inputs already match its contract;
2. add a thin adapter when only objective, constraint, sampler, or transform conversion is
   required;
3. add an explicit robotorchan composition policy when heterogeneous predictive
   representations must be combined;
4. implement a custom acquisition only when the required semantics cannot be expressed by the
   supported BoTorch interfaces.

Existing `FeasibilityWeightedAcquisition` is a deterministic probability-of-feasibility
weighting utility. It multiplies an already evaluated acquisition value by reduced classifier
feasibility and therefore must not be treated as a general sample-wise constrained Monte Carlo
contract. Joint Monte Carlo composition must preserve sample-level utility and feasibility
semantics before their required reductions.

Sampler selection remains governed by posterior-sampling capabilities. Heterogeneous
composition must reuse that contract rather than choose samplers from concrete model classes.

`HeterogeneousModel` deliberately has no shared `posterior()` contract. Regression posteriors
and classification latent posteriors have different meanings, so acquisition composition must
preserve entry ownership and native predictive representations rather than concatenate
heterogeneous outputs into a synthetic joint posterior or sample tensor solely to satisfy a
BoTorch constructor.

Semantic objective and constraint adapters are therefore entry-local at the BoTorch boundary.
A regression objective or continuous outcome constraint can be converted to a callable over
samples from its owning entry. Classification probability objectives expose both
posterior-predictive evaluation and, when the classifier supports epistemic sampling,
class-probability samples induced by its latent posterior.

Classification feasibility exposes posterior-predictive probability of feasibility and an
optional thresholded probability residual. Sample-wise classification feasibility is represented
separately by `SampleProbabilityOfFeasibility` and is derived from the classifier's
`sample_class_probabilities()` contract without reinterpreting latent-function samples as
outcome samples. Models that do not provide epistemic probability samples may support
deterministic probability weighting but must not be treated as sample-wise feasible by
implicit fallback.


Outcome or black-box constraints and candidate/input-space constraints remain distinct.
Outcome constraints belong to acquisition composition. Candidate constraints belong to
acquisition optimization through `robotorchan.optim.CandidateConstraints`.

Regression level-set estimation is not classification and remains represented by the
regression active-learning package.

## Current package shape

Only robotorchan-owned implementations or concrete cross-layer metadata / dispatch
responsibilities create modules:

```text
acquisition/
├── __init__.py
├── capabilities.py
├── classification_constraints.py
├── compatibility.py
├── non_gp.py
├── optimizer_compatibility.py
├── registry.py
├── samplers.py
├── active_learning/
│   ├── classification.py
│   ├── epig.py
│   ├── randomized_straddle.py
│   ├── straddle.py
│   └── variance.py
└── sampling/
    └── thompson.py
```

Empty scaffolding is avoided.

## Public API contract

- BoTorch-native acquisitions are not re-exported from robotorchan.
- The acquisition registry is capability metadata, not a string-based factory.
- Registry entries for BoTorch-native acquisitions must not create robotorchan wrapper
  classes solely for namespace uniformity.
- No legacy aliases or deprecated wrappers are added.
- Custom acquisitions follow BoTorch's `AcquisitionFunction` conventions whenever
  practical so that BoTorch optimizers can consume them directly.
- Model capability checks fail explicitly and close to construction, recommendation, or
  evaluation.
- Candidate feasibility remains owned by `robotorchan.optim`; acquisition metadata only
  describes acquisition semantics.

This keeps acquisition functions composable with BoTorch while allowing workflow and
optimization layers to reason about concrete compatibility requirements without building a
second acquisition API.


### Sample-shape contract

Acquisition composition preserves BoTorch's shape semantics instead of repairing
shape mismatches by implicit squeezing or broadcasting. Once the sampler's
`sample_shape` and the posterior's fully broadcast `batch_shape` are resolved,
one scalar sample-wise objective, residual, or feasibility value has shape

```text
sample_shape + batch_shape + (q,)
```

The source output/class axis is removed only when selecting the semantic scalar.
The sample dimensions, t-batch dimensions, and q dimension remain explicit.
Deterministic posterior-predictive PoF has no Monte Carlo sample dimensions and
therefore remains a separate representation.

`SampleShapeContract` records this boundary contract from explicit, already-resolved
`sample_shape`, posterior `batch_shape`, and `q`. It intentionally does not infer
posterior batches from `X`, because model batches may add dimensions, and it does
not assign a meaning to `sample_shape=None`, because sampler backends may define
different defaults. Composition code must not
silently squeeze q=1, collapse t-batches, copy deterministic mean PoF across a
sample dimension, or rely on accidental broadcasting to reconcile independently
sampled heterogeneous outputs. Cross-model sample alignment and joint sampling
are separate composition policies and are not implied by matching tensor shapes.


### Native qEI integration

Single-objective unconstrained expected improvement is composed with BoTorch's
native `qExpectedImprovement`; robotorchan does not implement a parallel qEI
class. The composition builder resolves exactly one `RegressionObjective`,
reuses its owning BoTorch model and `MCAcquisitionObjective`, transforms
`best_f` into the same semantic maximize-space as the objective, and forwards
an optional native `MCSampler`.

Phase 11 deliberately rejects feasibility bindings. Continuous and mixed learned
constraints are composed in the following phases so unconstrained qEI does not
silently apply deterministic PoF weighting or invent cross-model sample coupling.


### qEI with continuous constraints

Continuous outcome constraints can be composed into native BoTorch qEI when the
objective and every constraint are outputs of the same owning BoTorch model
entry. In that case they share one posterior sample tensor, so robotorchan builds
BoTorch's `ConstrainedMCObjective` from the semantic objective and
sample-residual constraint callables and passes it to native
`qExpectedImprovement`.

Phase 12 does not pretend that independent heterogeneous model entries form one
joint posterior. If a continuous constraint belongs to another entry, constrained
qEI rejects the composition rather than independently sampling tensors and
treating matching sample indices as joint draws. Classification feasibility is
also excluded here and is composed in the mixed-constraint phase.


### qEI with mixed constraints

The Phase 13 mixed baseline keeps the two established feasibility semantics
distinct. Continuous sample residuals that share the objective model entry are
handled inside BoTorch's `ConstrainedMCObjective`. Classification constraints
remain posterior-predictive `ProbabilityOfFeasibility` values and weight the
resulting native qEI outside its objective posterior.

This deliberately does not pair classifier probability samples with regression
posterior samples by matching Monte Carlo indices. Separate heterogeneous model
entries do not provide a joint posterior merely because their tensors have the
same shape. Sample-wise cross-model aggregation and correlation policy belong to
the later feasibility-aggregator and correlation phases.


### Native qNEI integration

Phase 14 exposes `make_qnei_acquisition` for a single unconstrained
`RegressionObjective`. It delegates noisy-baseline improvement, posterior
sampling, and baseline handling to BoTorch's native
`qNoisyExpectedImprovement`. The semantic objective bridge preserves output
ownership and maximize/minimize direction. `X_baseline` is supplied explicitly
and must be a nonempty two-dimensional tensor; no observed best value is
required. Constraint feasibility composition is deferred to Phase 15.


### qNEI continuous feasibility semantics

The Phase 15 native qNEI baseline accepts only sample-residual continuous
constraints sharing the regression objective's BoTorch model entry. The
`ConstrainedMCObjective` is evaluated on the same outcome samples for both
candidate and baseline values by native `qNoisyExpectedImprovement`. Baseline
feasibility must not be replaced by an externally supplied unconstrained
`best_f`, nor may an independently trained classifier be treated as part of
this joint posterior. Classification and multi-entry aggregation are deferred
to later composition phases.


### Multiple learned feasibility factors

Phase 16 composes same-posterior continuous residual constraints into the
native qNEI Monte Carlo objective and weights the resulting acquisition by
one or more classifier predictive feasibility factors. Multiplying marginal
classifier probabilities is an **explicit independence approximation**, not
joint feasibility sampling; it does not correct baseline feasibility for
independently learned classifiers. Correlated constraint aggregation and
sample-wise cross-model semantics remain separate design work in Phases 17–18.


### Phase 17: explicit feasibility aggregation

`IndependentFeasibilityAggregator` combines distinct classifier marginal
probabilities **per candidate** and retains the q dimension. The outer
`FeasibilityWeightedAcquisition` applies the selected q reduction once,
avoiding nested reductions when multiple classification constraints exist.
The product across classifiers is an independence approximation; correlated
constraints, shared output references, and joint posterior semantics are not
inferred. Phase 18 audits those assumptions.


### Phase 18: constraint correlation audit

The existing aggregator computes `prod_i P(C_i | X)` for distinct
classifier outputs. This is a **conditional independence assumption** and
must not be interpreted as `P(all C_i | X)` without supporting dependence
information. For example, two events with marginal probability 0.6 yield
0.36 under independence, but 0.6 when the events are identical. In general,
for two events with probabilities p and q, the joint probability is bounded
by `max(0, p + q - 1) <= P(A and B) <= min(p, q)`.

| Representation | Cross-constraint dependence | Supported behavior |
| --- | --- | --- |
| Continuous residuals from one BoTorch model entry | Shared posterior samples | Native `ConstrainedMCObjective` evaluates all residuals on the same draws |
| Distinct classifier marginal PoFs | Unknown | Explicit independent product approximation |
| Same classifier output repeated | Not independent | Rejected before aggregation |
| Residual constraints from separate model entries | Unknown | Rejected by qNEI composition |
| Sample-wise classifier probabilities | Requires explicit coupling | Not silently combined with regression posterior samples |
| Baseline classifier feasibility for qNEI | Unknown | Not corrected by outer candidate PoF weighting |

A minimum across marginal probabilities is an upper bound on a conjunction,
not a joint probability estimator. Multiplying marginal factors across
candidate points also assumes independence across the q-batch; posterior
correlation within a classifier is not recovered from pointwise
`predict_proba`. The `minimum` q reduction is a heuristic and must not
be labeled joint PoF.

The acquisition implementation does **not** currently estimate dependence,
construct a joint classifier posterior, or provide calibrated lower/upper
joint-feasibility bounds. Future extensions must require explicit dependence
representations, sampling alignment and batch-shape contracts rather than
inferring correlations from heterogeneous model ownership.


### Phase 19: multi-objective bridge

`make_botorch_multiobjective_bridge` converts at least two distinct
`RegressionObjective` bindings from **one shared BoTorch model entry**
into a native `GenericMCMultiOutputObjective`. Each output retains its
declared direction, and the bridge exposes ordered global output indices.
It does not build an acquisition function, reference point, or partitioning.

Separate heterogeneous model entries do not expose a coupled posterior.
Consequently, cross-entry objectives are rejected rather than silently
combining independent draws. Mixed regression/classification objectives
and `ProbabilityObjective` remain future integration work. Phase 20
will connect compatible multi-output objectives to native qEHVI.


### Phase 20: native qEHVI integration

`make_qehvi_acquisition` constructs BoTorch's
`qExpectedHypervolumeImprovement` using the Phase 19 multi-output bridge.
It requires at least two regression objectives owned by one shared BoTorch
posterior and accepts an explicit `NondominatedPartitioning`, reference
point and optional native MC sampler. The partitioning's observed outcomes
and reference point **must already be transformed to the directed
(maximization) objective space**, including sign changes for minimized
outputs. No partitioning or joint posterior is fabricated.

This phase is unconstrained: semantic feasibility bindings are rejected.
Constrained qEHVI belongs to Phase 21. Cross-entry multiobjective models
and classification probability objectives remain unsupported.

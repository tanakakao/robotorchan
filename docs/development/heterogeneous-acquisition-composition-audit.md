# Heterogeneous Acquisition Composition: Phase 1 Architecture Audit

## Scope

This audit records the current acquisition architecture before heterogeneous semantic
composition is added. The repository `main` branch is the source of truth.

The goal is to preserve the existing BoTorch-first ownership boundary and identify where
robotorchan must bridge semantic objectives and constraints without duplicating BoTorch
acquisition algorithms.

## Current ownership boundary

| Concern | Current owner | Phase 1 decision |
| --- | --- | --- |
| Standard single-objective BO | BoTorch | Keep native |
| Noisy / batch improvement | BoTorch | Keep native |
| Multi-objective BO | BoTorch | Keep native |
| Lookahead / KG | BoTorch | Keep native |
| Outcome-constraint MC composition | BoTorch | Prefer native composition |
| Candidate/input constraints | robotorchan optimizer + BoTorch | Keep separate |
| Classification probability semantics | robotorchan | Reuse as semantic input |
| Heterogeneous objective/constraint semantics | robotorchan semantics | Bridge only |
| Active-learning acquisitions | robotorchan where distinct | Keep separate |
| Sampler selection | robotorchan utility over BoTorch | Reuse |

The acquisition registry already marks representative BO acquisitions with
`implementation_strategy="BoTorch-native acquisition; no robotorchan wrapper"`.
This remains the governing rule for the composition cycle.

## Existing standard acquisition surface

The documented standard path uses native BoTorch classes. In particular, robotorchan
recommends the log-space improvement implementations rather than duplicating legacy EI
implementations.

The current representative registry includes:

- `qLogExpectedImprovement`;
- `qLogNoisyExpectedImprovement`;
- `qUpperConfidenceBound`;
- `qKnowledgeGradient`;
- `qMultiFidelityKnowledgeGradient`;
- `qLogExpectedHypervolumeImprovement`;
- `qLogNoisyExpectedHypervolumeImprovement`;
- `qLogNParEGO`.

The registry is compatibility metadata, not a local acquisition catalog.

## Existing semantic inputs

The completed semantic layer already exposes the main inputs required by the next phases:

- `RegressionObjective.to_botorch(...)` creates a BoTorch MC objective for regression
  samples.
- `ProbabilityObjective` evaluates and samples directed class probabilities.
- `ContinuousConstraint.to_botorch(...)` creates a sample residual using the BoTorch
  outcome-constraint sign convention: values less than or equal to zero are feasible.
- `ClassificationConstraint` exposes classifier-backed probability of feasibility and an
  explicit feasibility representation.
- `ProblemSemantics.feasibility_representations(...)` preserves declared constraint order.

The semantic layer therefore already contains the model-specific knowledge. Acquisition
composition must consume these representations rather than inspect concrete model classes.

## Feasibility representation boundary

The semantic layer currently distinguishes:

- `SampleResidualFeasibility`;
- `ProbabilityOfFeasibility`;
- `ProbabilityResidualFeasibility`.

This is the correct extension boundary for heterogeneous constraints. Acquisition code must
not silently convert one representation into another.

A continuous regression constraint naturally maps to sample residuals. A classification
constraint without an explicit probability threshold retains posterior-predictive probability
semantics instead of pretending to be a regression residual.

## Existing deterministic classification weighting

`acquisition/classification_constraints.py` contains
`FeasibilityWeightedAcquisition`, which evaluates an objective acquisition and multiplies it
by classifier probability of feasibility after a q reduction.

This utility is useful as the deterministic PoF reference path, but it must not become the
general heterogeneous constrained-BO contract.

Reasons:

1. it composes already-reduced acquisition values with feasibility after acquisition
   evaluation;
2. q-batch feasibility is reduced by an explicit product or minimum policy;
3. it does not express sample-wise joint utility of the form
   `E[U(X) F(X)]`;
4. it cannot by itself preserve correlated objective/constraint MC semantics.

The heterogeneous composition cycle should therefore treat this implementation as a baseline
and compatibility utility, not as the architecture to generalize.

## Sampler boundary

`make_model_sampler` already selects BoTorch samplers from posterior sampling capabilities:

- Gaussian posterior -> `SobolQMCNormalSampler`;
- empirical ensemble posterior -> `IndexSampler`;
- stochastic posterior -> `StochasticSampler`.

Heterogeneous acquisition composition should reuse the posterior/sampling contract. It must
not add acquisition-specific sampler selection based on concrete model classes.

## Outcome constraints versus candidate constraints

The existing architecture correctly separates two unrelated constraint spaces.

Outcome or black-box constraints operate on uncertain model outputs and belong to acquisition
composition. Candidate constraints operate directly on proposed inputs and belong to
acquisition optimization through `CandidateConstraints`.

The heterogeneous acquisition work must not merge these contracts.

## BoTorch-first classification of required work

The next phases should classify each requested behavior into four levels.

### Level A: direct BoTorch path

Use BoTorch without a robotorchan wrapper when semantic inputs already match the native
constructor contract.

### Level B: thin semantic adapter

Add only conversion from robotorchan semantic declarations to native BoTorch objective,
constraint, sampler, or transform inputs.

### Level C: heterogeneous composition policy

Add robotorchan logic only when regression and classification predictive representations must
be combined while preserving their distinct semantics.

### Level D: custom acquisition implementation

Create a new acquisition algorithm only when the required semantics cannot be represented by
the supported BoTorch interfaces. No Phase 1 requirement currently justifies Level D.

## Acquisition-specific audit decisions

| Family | Current status | Composition direction |
| --- | --- | --- |
| qLogEI | Native and registered | semantic objective + constraint bridge |
| qLogNEI | Native and registered | baseline-aware semantic bridge |
| qUCB | Native and registered | audit feasibility semantics before composing |
| qKG | Native and registered | audit fantasy/one-shot semantics before composing |
| qLogEHVI | Native and registered | multi-objective semantic bridge |
| qLogNEHVI | Native and registered | noisy multi-objective semantic bridge |
| qLogNParEGO | Native and registered | avoid EHVI-specific composition assumptions |
| AL acquisitions | Locally owned where distinct | extension point only in this cycle |

The implementation phases should use the log-space BoTorch variants as the preferred
production path. References to qEI/qNEI/qEHVI/qNEHVI in the development plan describe the
algorithm families, not a requirement to prefer their legacy non-log classes.

## Risks to control in Phases 2-10

The highest-risk boundaries are:

1. sample shape, t-batch shape, q dimension, output dimension, and constraint dimension;
2. mixing deterministic mean PoF with sample-wise feasibility;
3. accidentally treating classifier probabilities as regression outcomes;
4. reducing q feasibility before utility when MC semantics require sample-wise composition;
5. assuming independence across learned constraints without an explicit aggregation policy;
6. losing output-owner mapping when heterogeneous outputs live in different model entries;
7. confusing acquisition outcome constraints with optimizer candidate constraints;
8. wrapping native BoTorch acquisitions merely to expose a robotorchan name.

## Phase 1 conclusion

The current repository is already aligned with the intended BoTorch-first architecture.
No new qEI, qNEI, qEHVI, qNEHVI, or qLogNParEGO implementation is required by this audit.

The main missing layer is not another acquisition function. It is a composition bridge that
translates `ProblemSemantics` and its explicit objective/feasibility representations into
the appropriate BoTorch-native acquisition inputs while preserving heterogeneous prediction
semantics.

Phase 2 should therefore audit the exact Semantic -> Acquisition contract and identify which
semantic information is directly consumable, which needs a thin adapter, and which requires
an explicit heterogeneous composition policy.

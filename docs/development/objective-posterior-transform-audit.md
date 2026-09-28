# Objective / PosteriorTransform audit inventory

This document records Phase 1 of the Objective / PosteriorTransform compatibility audit.

Source of truth:

- repository: `tanakakao/robotorchan`
- base branch: `main`
- audited commit: `75f681e007b9a70f82008e67af4e1276b452318f`
- supported BoTorch range: `>=0.18.1,<0.19`

The inventory is descriptive. It does not declare a capability supported until later phases
validate its runtime contract.

## Responsibility boundary

The audit keeps these concepts separate:

1. `PosteriorTransform` acts on a posterior distribution.
2. `MCAcquisitionObjective` acts on posterior samples.
3. Outcome constraints act on modeled outcome samples inside an acquisition.
4. Candidate constraints act on known input-space feasibility during optimization.
5. Input perturbations generate scenarios before posterior evaluation.
6. Risk measures aggregate scenario outcomes after objective evaluation.

No robotorchan-specific Objective framework is introduced by this phase.

## Public robotorchan objective surface

The current `robotorchan.objectives` package contains robust scenario aggregations:

- `Expectation`
- `MeanVariance`
- `WorstCase`
- `VaR`
- `CVaR`
- `SNRatio`
- `make_risk_measure`

These objects reduce the final scenario dimension. They are not subclasses or aliases of
BoTorch `MCAcquisitionObjective` or `PosteriorTransform`.

Current tests cover basic values, parameter validation, scenario-axis reduction, and gradient
propagation through perturbation, posterior sampling, and risk aggregation.

## BoTorch Objective usage

robotorchan currently relies on native BoTorch objective objects rather than wrapping them.

Observed paths include:

- `GenericMCObjective` in outcome-constrained optimization tests.
- Native multi-objective acquisitions and qLogNParEGO scalarization.
- `MCAcquisitionObjective | None` accepted by Thompson candidate selection.
- Documentation examples using `GenericMCObjective` and
  `IdentityMCMultiOutputObjective`.

No robotorchan implementation of `GenericMCObjective`, `LinearMCObjective`,
`MCAcquisitionObjective`, or `ScalarizedPosteriorTransform` was found.

## PosteriorTransform inventory

### Direct pass-through paths

The following model families expose a `posterior_transform` argument and apply or forward it:

- NGBoost posterior adapter
- bootstrap ensemble posterior adapter
- random-forest posterior adapter
- DeepGP posterior adapter
- uncertain-input models
- reduced mixed models through their delegated posterior path

These paths require runtime validation in later phases. Presence of the argument alone is not
evidence of full BoTorch compatibility.

### Explicitly restricted paths

ALEBO currently raises `NotImplementedError` when `posterior_transform` is supplied.

Reduced high-dimensional output models reject `posterior_transform` while output reduction is
active. This is intentional in the current implementation because the posterior must first be
reconstructed into the original output space.

These restrictions must not be silently converted to "supported" metadata.

## Multi-output and multi-objective inventory

Standard multi-objective BO is intentionally BoTorch-native. Existing tests cover:

- qLogExpectedHypervolumeImprovement
- qLogNoisyExpectedHypervolumeImprovement
- qLogNParEGO
- ModelListGP multi-output posterior sampling
- reference-point construction
- optimizer execution for qLogEHVI

The acquisition registry marks the validated native BoTorch paths and does not create objective
wrappers.

The current compatibility layer treats structured outputs separately and reports that
scalarization is required when an acquisition cannot consume the structured posterior.

## Constraint inventory

Outcome and candidate constraints already have separate ownership.

Outcome constraints are passed to compatible BoTorch acquisitions as sample-space callables.
An E2E test combines:

`ModelListGP -> SobolQMCNormalSampler -> GenericMCObjective -> outcome constraint
-> qLogExpectedImprovement -> optimize_acqf_botorch`.

Candidate constraints belong to `robotorchan.optim`. Acquisition capability metadata explicitly
states that `supports_constraints` refers to output / black-box constrained BO and must not be
used to infer candidate-constraint support.

## Robust and input-perturbation inventory

Input scenarios are generated in `robotorchan.uncertainty`, including Gaussian, uniform, and
correlated Gaussian perturbations.

Risk aggregation is implemented in `robotorchan.objectives.risk`. The current E2E contract keeps
input scenario sampling, posterior sampling, and risk aggregation as distinct axes and verifies
candidate gradients for representative risk measures.

This separation is consistent with the intended audit boundary. Later phases must determine
whether any risk aggregation should use a native BoTorch risk-measure API instead.

## Capability metadata inventory

The acquisition registry currently describes properties such as:

- posterior requirement
- multi-output support
- structured-output support
- ensemble support
- outcome-constraint composition
- multi-objective support
- Monte Carlo requirement
- fantasize requirement
- multi-fidelity requirement

There are no dedicated `supports_objective` or `supports_posterior_transform` flags.

Phase 1 does not add them. Later phases should add metadata only if runtime behavior cannot be
represented or validated cleanly without it.

## Phase 1 gaps carried forward

The following are audit targets, not yet confirmed defects:

1. Validate direct `PosteriorTransform` pass-through against BoTorch's current public contract.
2. Determine whether ALEBO's transform restriction is fundamental or removable.
3. Validate transform semantics for reconstructed high-dimensional outputs.
4. Test native MC objectives across q, batch, and output shapes.
5. Separate task dimensions from objective/output dimensions for MultiTask and Kronecker models.
6. Validate native multi-output objectives, subset selection, and nonlinear objectives.
7. Validate outcome-constraint sign and shape conventions.
8. Compare robotorchan risk aggregations with BoTorch `0.18.x` risk-measure APIs.
9. Confirm dtype, device, and autograd preservation for objective and transform paths.
10. Confirm native BoTorch Objective / PosteriorTransform objects pass through without conversion.

## Phase 1 conclusion

The current architecture is already predominantly BoTorch-first: standard objectives and
multi-objective acquisitions are consumed directly, while robotorchan-specific objective code is
limited to scenario-risk aggregation.

The main compatibility risk is not duplicate Objective wrappers. It is uneven
`PosteriorTransform` behavior across specialized posterior implementations and the need to prove
shape semantics for MultiTask, Kronecker, structured-output, robust, and non-GP paths.

No production behavior is changed in Phase 1.

## Phase 2: BoTorch 0.18.1 API mapping

robotorchan declares `botorch>=0.18.1,<0.19`. The compatibility target for this audit is
therefore the public BoTorch 0.18.x API, with 0.18.1 as the minimum supported contract.

The mapping below is normative for later phases: use the BoTorch object directly unless a
robotorchan-specific semantic gap is demonstrated.

| Concern | BoTorch-native surface | robotorchan status | Audit rule |
| --- | --- | --- | --- |
| MC objective base | `MCAcquisitionObjective` | accepted by Thompson selection | pass through |
| ad-hoc scalar objective | `GenericMCObjective` | used directly in tests | do not wrap |
| linear MC scalarization | `LinearMCObjective` | no duplicate found | use native API |
| identity MC objective | `IdentityMCObjective` | implicit native acquisition behavior | use native API |
| posterior transform | `PosteriorTransform` | model-dependent pass-through | validate per model |
| affine posterior scalarization | `ScalarizedPosteriorTransform` | no duplicate found | use native API |
| multi-output MC objective | `MCMultiOutputObjective` family | documented native usage | use native API |
| output subset selection | `IdentityMCMultiOutputObjective` | documented directly | use native API |
| constrained MC composition | acquisition `constraints` and native objective utilities | native path tested | do not merge with candidate constraints |
| multi-objective BO | native qLogEHVI / qLogNEHVI / qLogNParEGO | native paths tested | preserve vector objective where required |
| risk over environmental scenarios | BoTorch risk-measure MC objective APIs | robotorchan has scenario aggregators | compare semantics before retaining or replacing |
| input perturbation | input/scenario transform responsibility | `robotorchan.uncertainty` | keep outside generic Objective |

### PosteriorTransform versus MC Objective

A `PosteriorTransform` transforms the posterior distribution before sampling or analytic
acquisition evaluation. This makes it the appropriate standard mechanism when an acquisition
needs a transformed posterior, including affine scalarization compatible with analytic
acquisitions.

An `MCAcquisitionObjective` transforms posterior samples. Its core shape contract is conceptually

`sample_shape x batch_shape x q x m -> sample_shape x batch_shape x q`.

Nonlinear sample objectives therefore belong on the MC-objective path rather than being presented
as posterior transforms.

robotorchan must not introduce an abstraction that hides this distinction.

### Scalarization mapping

For affine scalarization of a posterior, the preferred standard API is
`ScalarizedPosteriorTransform`. It propagates the posterior mean and covariance under the linear
map and can be consumed by compatible analytic as well as MC acquisition functions.

For scalarization defined on posterior samples, use a native MC objective. For nonlinear
scalarization, `GenericMCObjective` is the default extension point unless a reusable native
BoTorch objective already expresses the operation.

For multi-objective acquisitions, scalarization is not the default. qLogEHVI and qLogNEHVI retain
the objective vector required for hypervolume calculations. qLogNParEGO is a distinct
scalarization-based multi-objective strategy and should not be used as evidence that all
multi-objective paths should be scalarized.

### Constraint mapping

Outcome constraints and candidate constraints remain separate.

Outcome constraints are functions of modeled outcome samples and are composed with compatible
BoTorch acquisitions. Their exact sign, sample shape, and feasibility semantics are deferred to
the dedicated constraint phase.

Candidate constraints are known functions of `X` and remain an acquisition-optimization
responsibility under `robotorchan.optim`.

No shared `constraints` abstraction should combine these two concepts.

### Risk-measure mapping

BoTorch provides risk-measure MC-objective machinery for robust BO with environmental variables.
This overlaps conceptually with part of `robotorchan.objectives.risk`.

Phase 2 does not replace the existing risk classes because API-name similarity is insufficient to
establish semantic equivalence. Phase 12 must compare at least:

- expected scenario-axis layout and `n_w` semantics,
- maximization/minimization and tail conventions,
- VaR/CVaR definitions and empirical estimators,
- differentiability,
- preprocessing / objective composition,
- compatibility with native robust acquisition workflows,
- whether `SNRatio` has a meaningful BoTorch-native equivalent.

Until that comparison is complete, the robotorchan risk classes are classified as
`robotorchan extension under review`, not as duplicates.

### Pass-through policy established by Phase 2

The following are design requirements for subsequent phases:

1. Accept native BoTorch Objective objects where the target acquisition accepts them.
2. Accept native BoTorch PosteriorTransform objects where the model posterior contract supports
   them.
3. Do not convert either object into a robotorchan-specific type.
4. Do not add aliases for native BoTorch objective classes.
5. Use `GenericMCObjective` as the ordinary custom-objective extension point.
6. Preserve native multi-output objective classes for multi-objective acquisitions.
7. Keep input perturbation generation outside generic MC objectives.
8. Add a robotorchan-specific objective only when a demonstrated semantic gap survives the later
   compatibility phases.

### Phase 2 classification

- **BoTorch standard:** Objective base classes, generic/linear MC objectives, posterior transforms,
  scalarized posterior transforms, native multi-output objectives, and native constrained
  acquisition composition.
- **robotorchan extension:** current scenario-risk and Taguchi S/N aggregation.
- **duplicate:** none confirmed in Phase 2.
- **legacy:** none confirmed in Phase 2.
- **conditional / model-specific:** PosteriorTransform support for specialized posterior
  implementations.
- **requires semantic comparison:** robotorchan risk aggregation versus BoTorch risk-measure
  objectives.

Phase 2 changes documentation only. Runtime support remains unclaimed until the corresponding
compatibility and E2E phases pass.


## Phase 3: responsibility-boundary audit

Phase 3 audits ownership rather than runtime compatibility. The current codebase keeps the major
layers separate; no production boundary violation was found that requires an implementation
change in this phase.

### Canonical pipeline

The audited composition is:

`X -> candidate feasibility -> model.posterior(X) -> PosteriorTransform -> Posterior
-> Sampler -> posterior samples -> MC Objective -> objective samples
-> outcome feasibility / acquisition utility -> acquisition optimization`.

For robust candidate evaluation, input scenarios are introduced before posterior evaluation:

`X -> perturbation / environmental scenarios -> X' -> posterior -> sampling
-> objective -> scenario risk aggregation -> acquisition utility`.

The second pipeline does not make perturbation generation a responsibility of the model,
posterior sampler, or generic objective.

### Model and PosteriorTransform boundary

`posterior_transform` belongs to the model-posterior boundary. A model may apply the supplied
BoTorch transform to the posterior it constructs or may explicitly reject transforms whose
semantics it cannot preserve.

This boundary is currently visible in native-style posterior methods for non-GP adapters,
DeepGP, uncertain-input models, and reduced-model delegation. ALEBO and active output reduction
have explicit restrictions rather than silently changing transform semantics.

A `PosteriorTransform` must not be used as a replacement for a nonlinear sample objective.
Conversely, an MC objective must not be used to claim support for analytic posterior
transformation.

Phase 4 and Phase 5 will validate whether the current model-specific implementations satisfy the
actual BoTorch transform contract.

### Sampling and MC Objective boundary

Posterior sampling produces samples without deciding their optimization meaning. A native
`MCAcquisitionObjective` consumes those samples when an acquisition requires a scalar sample
utility.

The Thompson sampling path accepts a native `MCAcquisitionObjective | None`; it does not define a
parallel robotorchan objective interface. Existing constrained-BO tests likewise construct
`GenericMCObjective` directly.

The classes in `robotorchan.objectives.risk` are scenario-axis aggregators. They are not currently
`MCAcquisitionObjective` subclasses and must not be documented or passed as though they were
drop-in values for an acquisition's `objective=` argument. Their relationship to BoTorch
risk-measure objectives remains a Phase 12 question.

### Outcome-constraint boundary

Outcome constraints express unknown feasibility through modeled outcomes. They are evaluated on
posterior-derived samples by compatible acquisition functions.

The acquisition registry's `supports_constraints` capability refers to this output / black-box
constraint composition. It does not describe candidate-space feasibility.

The existing outcome-constrained E2E path correctly composes a model, posterior sampler,
`GenericMCObjective`, outcome constraint callables, a constrained MC acquisition, and
acquisition optimization without moving candidate constraints into the acquisition objective.

### Candidate-constraint boundary

Known candidate/input-space constraints belong to `robotorchan.optim`.
`CandidateConstraints` explicitly describes candidate coordinates and separates itself from
output constraints used by constrained acquisition functions.

Linear equality, linear inequality, and nonlinear inequality constraints are therefore optimizer
inputs. They must not be represented as posterior-sample constraints merely to reuse an
acquisition API.

This separation also applies to mixed-variable search: categorical/integer domain structure and
candidate feasibility are optimization-space concerns, not MC-objective semantics.

### Input-perturbation and uncertain-input-model boundary

Two different uncertainty concepts remain separate:

1. uncertain-input surrogate models represent input uncertainty inside the statistical model;
2. candidate-time perturbation generates decision scenarios around candidate inputs.

Candidate perturbation utilities live under `robotorchan.uncertainty`. The uncertain-input model
package explicitly states that it does not own candidate perturbation scenarios.

The robust workflow therefore remains compositional. An objective receives modeled outcomes; it
does not generate perturbed candidate inputs.

### Risk-aggregation boundary

The current robust pipeline distinguishes posterior-sampling axes from scenario axes. Risk
aggregation reduces the scenario dimension after objective evaluation.

This is a useful semantic boundary, but the current risk API is not yet certified as the final
public abstraction. BoTorch 0.18.x also has risk-measure MC-objective machinery. Phase 12 must
decide whether the robotorchan classes are complementary, redundant, or should be adapted.

Until then:

- do not merge posterior sampling and scenario sampling,
- do not make perturbation generation part of a generic objective,
- do not advertise robotorchan risk aggregators as native MC objectives,
- do not add a second risk-objective wrapper layer.

### Boundary matrix

| Concept | Owns | Must not own | Current status |
| --- | --- | --- | --- |
| Model / posterior | predictive posterior | optimization utility | separated |
| PosteriorTransform | posterior-level transformation | candidate perturbation | separated |
| Sampler | posterior sample generation | objective semantics | separated |
| MC Objective | sample-to-utility transformation | known candidate feasibility | separated |
| Outcome constraint | modeled unknown feasibility | input-space constraints | separated |
| Candidate constraint | known feasibility of `X` | posterior-sample feasibility | separated |
| Input perturbation | candidate scenario generation | posterior transformation | separated |
| Risk aggregation | scenario utility aggregation | scenario generation | separated |
| Acquisition | utility of candidate evaluations | optimizer feasibility mechanics | separated |
| Optimizer | candidate search and known constraints | outcome-model semantics | separated |

### Phase 3 findings

No architectural boundary violation requiring production-code changes was found.

The remaining risks are compatibility questions rather than ownership errors:

1. whether specialized posterior implementations correctly support native
   `PosteriorTransform`;
2. whether sample, task, output, q, batch, and scenario dimensions remain unambiguous;
3. whether robotorchan risk aggregation overlaps BoTorch risk-measure objectives;
4. whether all documentation consistently uses `supports_constraints` only for outcome
   constraints;
5. whether full E2E paths preserve these boundaries under mixed and constrained optimization.

These are carried into the dedicated later phases. Phase 3 introduces no wrapper, alias, or
production API.


## Phase 4: PosteriorTransform compatibility

Phase 4 adds runtime contract coverage for representative model families using BoTorch's native
`ScalarizedPosteriorTransform`.

The compatibility categories are:

| Model path | Phase 4 status | Reason |
| --- | --- | --- |
| standard exact GP | supported | native BoTorch posterior path |
| mixed exact GP | supported | native BoTorch posterior path |
| Kronecker multi-task | supported | native multi-output posterior transform |
| input-reduced GP | supported | transform is applied after input reduction |
| empirical non-GP ensemble | supported when transform accepts its posterior | direct native transform call |
| DeepGP empirical posterior | supported when transform accepts its posterior | direct native transform call |
| output-reduced GP | explicitly restricted | original-output reconstruction changes semantics |
| ALEBO metric-marginal model | explicitly restricted | specialized posterior path not certified |

The tests intentionally pass a BoTorch-created transform directly. No robotorchan adapter or
conversion layer is introduced.

For single-output Gaussian paths, the test also verifies the affine transform contract:
`mean' = offset + w * mean` and `variance' = w^2 * variance`.

For Kronecker multi-task output, the transform uses a weight for each task/output and verifies
that the transformed posterior is scalar-valued.

Output reduction remains an explicit unsupported composition in this phase. Applying a transform
in latent output coordinates would not generally equal applying the same transform after
reconstruction in the original output coordinates. Supporting that composition requires a
mathematically explicit original-output transform path rather than silently forwarding the
transform.

ALEBO remains explicit rather than falsely advertised as compatible. Its metric-marginal
posterior path requires a separate design decision if transform support is later justified.

Phase 4 therefore strengthens the compatibility contract with tests but does not add a wrapper or
change production semantics.

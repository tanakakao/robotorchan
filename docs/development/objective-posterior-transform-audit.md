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
| Kronecker multi-task | explicitly restricted | BoTorch 0.18.1 rejects posterior transforms |
| input-reduced GP | supported | transform is applied after input reduction |
| empirical non-GP ensemble | transform-dependent | scalarized transform rejects EnsemblePosterior |
| DeepGP empirical posterior | transform-dependent | scalarized transform rejects empirical posterior |
| output-reduced GP | explicitly restricted | original-output reconstruction changes semantics |
| ALEBO metric-marginal model | explicitly restricted | specialized posterior path not certified |

The tests intentionally pass a BoTorch-created transform directly. No robotorchan adapter or
conversion layer is introduced.

For single-output Gaussian paths, the test also verifies the affine transform contract:
`mean' = offset + w * mean` and `variance' = w^2 * variance`.

BoTorch 0.18.1 explicitly rejects posterior transforms for `KroneckerMultiTaskGP`. robotorchan
preserves that upstream restriction rather than introducing a compatibility implementation.

The empirical non-GP and DeepGP posterior adapters still forward a supplied transform, but native
`ScalarizedPosteriorTransform` rejects these posterior types because they do not expose the
distribution structure required by BoTorch's scalarization utility. Phase 4 records that behavior
as transform-dependent rather than claiming scalarized-transform support.

Output reduction remains an explicit unsupported composition in this phase. Applying a transform
in latent output coordinates would not generally equal applying the same transform after
reconstruction in the original output coordinates. Supporting that composition requires a
mathematically explicit original-output transform path rather than silently forwarding the
transform.

ALEBO remains explicit rather than falsely advertised as compatible. Its metric-marginal
posterior path requires a separate design decision if transform support is later justified.

Phase 4 therefore strengthens the compatibility contract with tests but does not add a wrapper or
change production semantics.


## Phase 5: ScalarizedPosteriorTransform compatibility

Phase 5 treats `posterior_transform=` as a common model API wherever the model can preserve the
meaning of the supplied BoTorch transform. Uniform syntax does not imply that every posterior
representation can support every transform.

### Scalarization contract

For a Gaussian multi-output posterior and affine scalarization

`Z = offset + w^T Y`,

the required moments are

`E[Z] = offset + w^T E[Y]`

and

`Var[Z] = w^T Cov[Y] w`.

The second expression requires cross-output covariance. Summing only marginal variances is valid
only when the relevant outputs are independent. Phase 5 therefore does not manufacture
scalarized posteriors from marginal means and variances alone.

### API consistency policy

Models should expose the standard `posterior_transform=` argument whenever their public
`posterior()` API is under robotorchan control. The implementation should then:

1. delegate directly to BoTorch when the underlying BoTorch model supports the transform;
2. apply the transform only after any robotorchan-owned output reconstruction when the transform
   is defined in the public/original output space;
3. preserve the transform object unchanged;
4. raise a clear unsupported error when the resulting Posterior type lacks the mathematical
   information required by that transform.

No robotorchan-specific transform DSL, alias, or compatibility wrapper is introduced.

### Output reduction

The previous output-reduction path rejected every non-null `posterior_transform` before
constructing the public posterior. Phase 5 removes that API-level rejection.

The new order is:

`latent posterior -> original-output reconstruction -> supplied PosteriorTransform`.

This order is necessary because public transform weights refer to original outputs, not latent
PCA/PLS coordinates.

The current `LinearOutputPosterior` preserves sampling through the latent posterior and exposes
restored marginal moments, but it does not expose a GPyTorch distribution containing the complete
joint covariance required by BoTorch's `ScalarizedPosteriorTransform`. Consequently the standard
scalarized transform still raises an explicit unsupported error after reconstruction. This is a
Posterior capability limitation, not an inconsistent model signature.

A temporary empirical approximation was considered and rejected: replacing the transformed
posterior with a finite ensemble would change the posterior representation and inject arbitrary
Monte Carlo error into a deterministic affine transform.

### Kronecker multi-task

BoTorch 0.18.1 explicitly rejects `posterior_transform` in
`KroneckerMultiTaskGP.posterior`. robotorchan does not bypass that guard in Phase 5. Correct
scalarization requires preserving the joint task and q covariance, so an implementation must be
based on that covariance structure rather than marginal moments.

The public method signature remains BoTorch-compatible. This is a capability gap to revisit only
if robotorchan can implement the exact transform without changing Kronecker semantics.

### Empirical non-GP and DeepGP posteriors

These models already expose `posterior_transform=` and pass the supplied object to the produced
Posterior. This keeps the API uniform.

BoTorch's current `ScalarizedPosteriorTransform` is distribution-oriented and does not
scalarize `EnsemblePosterior` or `DeepGPPosterior`. Phase 5 does not special-case the transform
inside each model. Sample-space scalarization remains naturally available through MC objectives,
which is the correct path for empirical posterior representations.

### Phase 5 decision

The common API is strengthened without claiming false universal capability:

- standard / mixed / compatible exact-GP posteriors: native scalarized transform;
- input reduction: native scalarized transform after input mapping;
- output reduction: common transform entry point, applied after output reconstruction;
- Kronecker: common BoTorch signature, upstream scalarized transform unsupported;
- ensemble non-GP: common entry point, transform-dependent;
- DeepGP empirical posterior: common entry point, transform-dependent.

Exact support for additional posterior types is desirable, but only when their full covariance or
sample semantics allow the requested transform to be implemented without approximation.


## Phase 6: MC Objective compatibility

Phase 6 verifies that robotorchan posterior families use BoTorch MC objectives in sample space
without model-specific objective adapters.

The canonical contract is:

`Posterior -> Sampler -> samples[..., q, m] -> MCAcquisitionObjective -> values[..., q]`.

Representative runtime coverage now includes:

| Posterior family | Sampler | Native MC Objective | Status |
| --- | --- | --- | --- |
| Kronecker multi-task | `SobolQMCNormalSampler` | `LinearMCObjective` | supported |
| empirical ensemble | `IndexSampler` | `GenericMCObjective` | supported |
| DeepGP trajectory | `StochasticSampler` | `GenericMCObjective` | single-output supported |
| single-output samples | sampler-independent | `IdentityMCObjective` | supported |
| output-reduced GP | normal sampler | `GenericMCObjective` | already E2E covered |

This distinction is important for models whose posterior representation cannot satisfy a
distribution-level `ScalarizedPosteriorTransform`. Once those posteriors produce correctly
shaped samples, MC objectives operate on tensors and therefore do not require a GPyTorch
distribution.

For Kronecker multi-task models, Phase 6 verifies linear scalarization in sample space and
candidate-gradient propagation. This provides the BoTorch-native scalar-objective path while
preserving the joint task covariance during posterior sampling.

For empirical ensemble and DeepGP posteriors, the model-specific sampler remains responsible for
drawing valid samples. The objective receives ordinary tensors and uses the same BoTorch API as
Gaussian models. The current DeepGP contract is single-output (`m=1`); Phase 6 therefore verifies
`GenericMCObjective` on that supported shape rather than implying unsupported multi-output
DeepGP behavior.

No robotorchan MC Objective wrapper, registry, conversion layer, or compatibility alias is added.
`GenericMCObjective` remains the extension point for arbitrary differentiable sample-space
objectives, and `LinearMCObjective` is preferred for ordinary weighted scalarization.

Phase 6 found no production-code gap that requires a custom Objective implementation. The
compatibility requirement is instead enforced by posterior sampling shape and sampler selection,
which were established by the preceding Sampling / Posterior audit and are now connected
explicitly to native MC Objective tests.


## Phase 7: sample, batch, q, and output shape contract

Phase 7 makes the MC Objective axis contract executable rather than relying on examples with only
one sample dimension or no t-batch.

The canonical tensor layout is:

`sample_shape x batch_shape x q x m`.

A scalar `MCAcquisitionObjective` removes only the final output dimension:

`sample_shape x batch_shape x q x m -> sample_shape x batch_shape x q`.

Neither the sampler nor the objective may silently flatten, squeeze, or reinterpret sample,
t-batch, q, task, or output axes.

### Audited cases

- multi-dimensional `sample_shape=[5, 7]`, proving that sample shape is not assumed to be one
  leading dimension;
- t-batch shape `[2]`;
- `q=1` and `q=3`;
- `m=1` and `m=2`;
- Kronecker task outputs, where the final task/output axis remains distinct from q;
- independent Gaussian external posterior sampling;
- empirical ensemble sampling;
- DeepGP single-output trajectory sampling;
- output-reducer inverse transforms from latent `m=2` to original `m=5`.

For example, a two-dimensional sample shape, t-batch 2, q=3, and m=2 must remain
`[5, 7, 2, 3, 2]` before scalarization and `[5, 7, 2, 3]` afterward.

### Task and output semantics

For block-design `KroneckerMultiTaskGP`, tasks occupy the final posterior output axis. A q=1
query with two tasks is therefore `[..., 1, 2]`, not `[..., 2, 1]`. Phase 7 covers q=1 and
q>1 separately to prevent accidental interchange of q and task/output axes.

Long-format MultiTask models can encode task identity in X instead. Their task-feature input
dimension is not itself an MC output axis. Downstream code must follow the returned Posterior
shape rather than infer axes from the model family name.

### Reduced outputs

Output reconstruction operates only on the final latent-output dimension. All leading sample,
batch, and q dimensions are preserved. MC objectives are then defined in the restored original
output space.

### Result

No production implementation change is required by the audited representative paths. The
existing posterior and reducer implementations preserve the BoTorch axis contract. Phase 7 adds
regression coverage so future posterior adapters or objectives cannot accidentally collapse q,
batch, or output dimensions.


## Phase 8: multi-output Objective compatibility

Phase 8 separates three operations that are easy to conflate:

1. output selection while preserving a vector objective;
2. component-wise weighting while preserving a vector objective;
3. scalarization of multiple outputs into one MC utility.

For BoTorch 0.18.1, robotorchan uses the native Objective APIs directly.

| Intent | Native API | Output semantics |
| --- | --- | --- |
| select/reorder outputs | `IdentityMCMultiOutputObjective` | vector preserved |
| select and weight outputs | `WeightedMCMultiOutputObjective` | vector preserved |
| affine scalarization | `LinearMCObjective` | scalar |
| nonlinear scalarization | `GenericMCObjective` | scalar |

No robotorchan output-selection DSL, objective registry, adapter, or compatibility alias is
required.

### Output selection and ordering

`IdentityMCMultiOutputObjective(outcomes=[...])` indexes the final posterior-sample output axis.
The requested order is semantically meaningful and is preserved. Phase 8 tests this on
Kronecker multi-task samples and empirical ensemble samples.

For output-reduced models, selection occurs only after samples have been reconstructed into the
original output space. Public outcome indices therefore refer to the original Y columns, not
latent PCA/PLS coordinates.

### Weighted vector objectives versus scalarization

`WeightedMCMultiOutputObjective` performs component-wise weighting and retains a final objective
dimension. It must not be treated as a scalar weighted sum.

A true weighted scalar utility uses `LinearMCObjective`, which reduces the final output
dimension. General nonlinear scalar utilities use `GenericMCObjective`.

This distinction is important for acquisition compatibility: scalar MC acquisitions consume
scalar objective samples, whereas hypervolume-based multi-objective acquisitions require vector
objective samples.

### Multi-objective preservation

qEHVI/qNEHVI-style workflows must preserve the selected objective vector through the
multi-output Objective. Scalarization before hypervolume computation would change the problem
from multi-objective BO into scalar BO.

qLogNParEGO is intentionally different: its scalarization is part of the acquisition strategy
and is handled by BoTorch rather than by a robotorchan Objective wrapper.

### Representative model coverage

Phase 8 verifies the native API on:

- Kronecker multi-task samples with three outputs;
- empirical ensemble multi-output samples;
- output-reduced samples restored from latent m=2 to original m=5.

Existing acquisition tests additionally cover original-space output selection with
`IdentityMCMultiOutputObjective` through qEHVI/qNEHVI on output-reduced GP models.

### Result

No production implementation gap was found. Multi-output Objective compatibility follows from
the common posterior-sample contract established in Phases 6 and 7. The correct extension point
remains the native BoTorch Objective API.


## Phase 9: MultiTask and Kronecker Objective semantics

Phase 9 verifies that Objective code consumes posterior outputs rather than interpreting model
input conventions.

### Long-format MultiTaskGP

A long-format `MultiTaskGP` stores task identity in a structural input feature during training.
For example, `[x, task]` identifies which task generated a scalar observation.

When prediction X includes an explicit task feature, the returned posterior is single-output:
the task feature chooses the modeled function but is not itself an Objective output dimension.

When prediction X omits the task feature and `output_indices=[...]` requests tasks, BoTorch
constructs those requested tasks on the final posterior output axis. Only at this point do
multi-output Objectives index or scalarize the tasks.

Therefore:

`task feature in X != posterior output axis`.

### Block-design KroneckerMultiTaskGP

For `KroneckerMultiTaskGP`, task identity is represented directly by the final training-Y and
posterior output dimension. With two tasks, a q=2 posterior has event/output shape `[2, 2]`
without a task feature in candidate X.

Thus:

`Kronecker task axis == posterior output axis`.

### Common Objective boundary

Although the model representations differ, the Objective boundary is the same after posterior
construction. If both posteriors expose two requested tasks as `[..., q, 2]`, the same native
`LinearMCObjective(weights=[...])` can reduce both to `[..., q]`.

Likewise, `IdentityMCMultiOutputObjective` can select/reorder tasks only when those tasks are
actually present on the posterior sample output axis.

Objective implementations must not inspect `task_feature`, infer task count from X, or special
case Kronecker model classes. Their contract starts at posterior samples.

### Result

No production implementation change is required. robotorchan preserves BoTorch's distinct
long-format and block-design representations while converging on the same native Objective
interface after posterior construction.


## Phase 10: multi-objective Objective compatibility

Phase 10 verifies the boundary between model outputs, multi-output Objectives, reference points,
Pareto partitioning, and acquisition-owned scalarization.

### Hypervolume acquisitions operate in Objective space

A surrogate may expose more outputs than the optimization problem uses. For example, a
three-output model can optimize only outputs `[2, 0]` through
`IdentityMCMultiOutputObjective(outcomes=[2, 0])`.

For qLogEHVI/qLogNEHVI, all hypervolume geometry must then use that transformed Objective space:

- posterior samples: model output space;
- multi-output Objective: select/reorder the optimization objectives;
- reference point: one entry per transformed objective;
- partitioning Y: transformed objective values in the same order;
- hypervolume utility: transformed Objective space.

The reference point is therefore not defined by the raw model output count. Its dimension and
ordering must match the final multi-output Objective.

### Output ordering is part of the optimization problem

If an Objective changes `[y0, y1, y2]` to `[y2, y0]`, both the reference point and
partitioning observations must use `[y2, y0]`. Reordering only posterior samples while retaining
a raw-output reference point would define inconsistent hypervolume geometry.

Phase 10 adds a regression test that a three-dimensional reference point is rejected for a
two-dimensional Objective/partitioning pair.

### qLogEHVI and qLogNEHVI preserve vector objectives

qLogEHVI and qLogNEHVI consume vector-valued multi-output Objectives. They must not be preceded
by a scalar `LinearMCObjective` or `GenericMCObjective` when the intended problem is
hypervolume-based multi-objective optimization.

This is a semantic distinction rather than a robotorchan compatibility restriction.

### qLogNParEGO owns its scalarization

qLogNParEGO is a scalarization-based multi-objective acquisition. Its
`scalarization_weights` are acquisition inputs and BoTorch constructs the scalarized utility
internally.

robotorchan therefore does not insert an additional scalar Objective in front of qLogNParEGO.
Doing so would collapse the multi-output posterior before the acquisition applies its intended
scalarization.

### Result

Native BoTorch multi-objective Objective composition is sufficient. No robotorchan Objective
wrapper or multi-objective scalarization layer is required. The main contract is that reference
points, partitioning observations, and vector Objectives all describe the same transformed
Objective space.


## Phase 11: outcome Constraint compatibility

Phase 11 verifies unknown outcome constraints as sample-space acquisition inputs and keeps them
strictly separate from known candidate/input-space constraints.

### Feasibility sign convention

BoTorch outcome-constraint callables follow the convention:

`constraint(samples) <= 0`

means feasible.

For a physical requirement `g(x) <= limit`, a natural callable is therefore
`g_sample - limit`. Reversing the sign silently reverses feasibility.

### Shape contract

Given posterior samples with shape

`sample_shape x batch_shape x q x m`,

each scalar outcome-constraint callable returns

`sample_shape x batch_shape x q`.

The callable removes only the output dimension used to compute that constraint. It must not
reduce sample, t-batch, or q dimensions. Multiple constraints are represented by multiple
callables rather than by reinterpreting one of these structural axes as a constraint axis.

### Objective and constraint composition

Scalar MC Objectives and outcome constraints consume the same raw posterior samples for different
purposes:

- Objective: raw model samples -> utility samples;
- outcome constraint: raw model samples -> signed feasibility values.

Outcome constraints do not consume the already scalarized Objective output. This permits a
multi-output model to expose, for example, one utility output and multiple modeled constraint
outputs without coupling their transformations.

Phase 11 verifies native `qLogExpectedImprovement` composition with one scalar
`GenericMCObjective` and two outcome constraints.

### Candidate constraints remain separate

Outcome constraints describe uncertain black-box quantities modeled by the surrogate and are
integrated probabilistically by compatible acquisitions.

`robotorchan.optim.CandidateConstraints` instead describes known feasibility in X-space and is
owned by acquisition optimization. Candidate constraints do not receive posterior samples, and
outcome constraints are not forwarded to optimizer constraint backends.

The two APIs must not be merged merely because both use the word constraint.

### Result

The native BoTorch outcome-constraint callable contract is sufficient. No robotorchan
`ConstrainedMCObjective` wrapper or second constraint language is required. The existing
acquisition capability metadata meaning of `supports_constraints` remains output/black-box
constraint compatibility, not candidate-space constraint support.


## Phase 12: Risk Measure compatibility

Phase 12 compares robotorchan's scenario aggregators with BoTorch's
`RiskMeasureMCObjective` family. Similar names do not imply identical APIs or finite-sample
estimators.

### Two distinct contracts

robotorchan currently provides lightweight scenario-axis aggregators:

`values[..., n_w] -> robust_values[...]`

They make the scenario axis explicit as the final tensor dimension and are used after
robotorchan perturbation and posterior sampling.

BoTorch risk measures are MC Objectives. With input perturbations, `n_w` environmental
realizations are represented in the acquisition sample layout and the Objective reconstructs the
candidate/scenario structure before reducing the environmental axis.

The robotorchan classes therefore must not be advertised as drop-in replacements for
`botorch.acquisition.risk_measures.RiskMeasureMCObjective`.

### Equivalent measures

For the same empirical scenarios and maximization convention:

- `Expectation` agrees with BoTorch `Expectation`;
- `WorstCase` agrees with BoTorch `WorstCase`.

These are straightforward reductions and do not depend on quantile interpolation conventions.

### VaR is not currently estimator-equivalent

Both APIs use the lower tail for a maximization objective, but the finite-sample estimators differ.

robotorchan `VaR` uses `torch.quantile(values, 1 - alpha)`, which linearly interpolates by
default. BoTorch's risk Objective uses its empirical scenario/order-statistic convention.

For scenarios `[1, 2, 3, 4, 5]` at `alpha=0.8`, the current robotorchan result is `1.8`,
while BoTorch 0.18.1 returns `2.0` for its empirical VaR convention.

This difference is semantically material. The classes must not be treated as interchangeable
merely because both are named VaR.

### CVaR is also estimator-sensitive

robotorchan `CVaR` first computes an interpolated quantile threshold and then averages scenario
values at or below that threshold. BoTorch's `CVaR` follows its empirical tail estimator within
the `RiskMeasureMCObjective` contract.

These formulations are not API-equivalent, even though they can produce the same value for some
finite scenario sets. For `[1, 2, 3, 4, 5]` at `alpha=0.7`, both currently return `1.5`.
Phase 12 therefore tests the observed agreement without making the stronger and incorrect claim
that the estimators must differ for every fractional empirical tail.

A future API decision may choose to align the robotorchan scenario aggregators with BoTorch, but
that would be a numerical contract change and must be handled deliberately rather than as a
compatibility alias.

### MeanVariance and SNRatio

`MeanVariance` and `SNRatio` remain robotorchan scenario aggregators.

- `MeanVariance` implements mean minus a configurable population-variance penalty.
- `SNRatio` implements Taguchi larger-is-better, smaller-is-better, and nominal-is-best forms.

They should be evaluated by their documented mathematical definitions, not mapped to a BoTorch
class solely to obtain a common name.

### Input perturbation responsibility

Risk aggregation does not generate environmental scenarios.

The robust pipeline remains:

`X -> perturbation/scenario generation -> posterior -> posterior sampling -> objective -> risk`

For the native BoTorch path, the input transform / perturbation layout and
`RiskMeasureMCObjective(n_w=...)` must agree on `n_w`. For the robotorchan lightweight path,
the scenario axis is already explicit and the aggregator reduces only that axis.

### Gradient, dtype, and device

The existing robotorchan E2E tests verify candidate gradients through perturbation, posterior
sampling, and risk aggregation for Expectation, MeanVariance, and CVaR. The implementations use
PyTorch tensor operations and preserve input dtype/device. SNRatio constructs its optional target
with the values' dtype/device.

Phase 14 will perform the cross-cutting gradient/dtype/device audit; Phase 12 establishes the
risk-specific contract needed for it.

### Result

BoTorch-native risk Objectives are the preferred path when the acquisition uses BoTorch's
environmental-scenario layout. robotorchan's existing risk classes remain useful explicit-axis
scenario aggregators and quality-engineering utilities.

No compatibility wrapper is introduced. In particular, VaR/CVaR numerical differences are
documented and tested rather than hidden behind aliases.


## Phase 13: Input Perturbation / Robust Objective compatibility

Phase 13 separates scenario generation from robust Objective evaluation and verifies both the
BoTorch-native and robotorchan explicit-axis pipelines end to end.

### BoTorch-native pipeline

BoTorch `InputPerturbation` is a one-to-many input transform. For
`X: batch_shape x q x d` and `n_w` perturbations it produces

`batch_shape x (q * n_w) x d`.

The perturbations belonging to one candidate remain adjacent on the posterior q-batch. This
layout is intentional: the environmental realizations are evaluated on the same GP sample path.

A native `RiskMeasureMCObjective(n_w=n_w)` reconstructs the candidate/scenario structure from
that flattened posterior-sample layout and reduces the environmental dimension.

The verified native path is therefore:

`X -> InputPerturbation -> posterior(q*n_w) -> sampler -> RiskMeasureMCObjective -> utility(q)`.

The model's `posterior` applies its configured input transform. The risk Objective does not
generate perturbations.

### robotorchan explicit-axis pipeline

robotorchan `ScenarioGenerator` uses a different, explicit representation:

`X[..., q, d] -> scenarios[..., q, n_w, d]`.

The posterior and sampler preserve the explicit candidate and scenario axes, after which the
lightweight robotorchan risk aggregator reduces only the final scenario axis.

The verified path is:

`X -> ScenarioGenerator -> posterior(q,n_w) -> sampler -> objective values -> risk -> utility(q)`.

This representation is useful when scenarios are generated externally or when callers want direct
control over the scenario tensor. It is not shape-compatible with BoTorch `InputPerturbation`
and must not be passed to a `RiskMeasureMCObjective` as though the layouts were identical.

### Gradient contract

Both paths preserve gradients from the final robust utility to the original candidate tensor in
the Phase 13 E2E tests.

The current Gaussian scenario generator constructs perturbations with PyTorch operations and adds
them to an expanded clone of `X`; this preserves the derivative of the perturbed points with
respect to the candidate.

### Responsibility boundary

The canonical robust decomposition is:

1. perturbation / environmental scenario generation,
2. posterior evaluation,
3. posterior sampling,
4. optional sample-space Objective transformation,
5. risk aggregation,
6. acquisition utility.

A risk measure must not own random input perturbation generation. Conversely, a scenario
generator must not choose or apply the risk measure.

### API decision

No robotorchan wrapper around BoTorch `InputPerturbation` or `RiskMeasureMCObjective` is
introduced. Native BoTorch objects should be used directly when their flattened `q*n_w`
contract is desired.

The existing robotorchan scenario generators and explicit-axis risk aggregators remain a
complementary lower-level path rather than compatibility aliases for the BoTorch classes.


## Phase 14: Gradient / dtype / device audit

Phase 14 audits the differentiable Objective and robust-Objective paths for accidental graph breaks,
dtype coercion, and device migration.

### Static audit

The Objective/risk path does not require a `.detach()`, NumPy conversion, or CPU round trip.
The explicit scenario generators construct random tensors using the input tensor's dtype and
device and add them to an expanded clone of the candidate tensor, preserving candidate gradients.

Repository-wide detach/CPU conversions found in non-differentiable external optimizers,
scikit-learn-style estimators, persisted metadata, or fitted-state snapshots are separate
responsibilities and are not Objective graph breaks.

### Runtime contracts

Phase 14 adds runtime coverage for both `torch.float32` and `torch.float64`:

- posterior samples -> `GenericMCObjective` -> candidate gradient,
- `GaussianPerturbation` -> posterior -> sampler -> robotorchan `Expectation`,
- BoTorch `InputPerturbation` -> posterior -> sampler -> BoTorch `Expectation`.

Each path must preserve the originating dtype and device through the final objective value and
candidate gradient.

A CUDA contract is also exercised when CUDA is available. CPU-only CI skips that test rather than
pretending to validate CUDA behavior.

### Construction rule

New Objective, PosteriorTransform, perturbation, and risk code must prefer tensor-derived
construction such as `new_tensor`, `zeros_like`, or explicit `dtype=X.dtype, device=X.device`
when creating tensors that participate in the differentiable path.

Using `torch.tensor(existing_tensor)`, implicit CPU constants, or an unconditional
`.cpu().numpy()` conversion inside that path is incompatible with the contract.


## Phase 15: Sampling -> Objective integration

Phase 15 verifies the integration boundary between posterior sampling and MC Objectives.

The contract is intentionally sampler-agnostic at the Objective boundary:

`Posterior -> compatible BoTorch sampler -> samples[..., q, m] -> MC Objective`.

No robotorchan conversion layer is required between a compatible sampler and the Objective.

Representative integration tests cover:

- Gaussian posterior + `SobolQMCNormalSampler`,
- Kronecker multi-task posterior + `SobolQMCNormalSampler`,
- ensemble posterior + `IndexSampler`,
- DeepGP posterior + `StochasticSampler`.

The tests use a multidimensional `sample_shape=[4, 5]` to ensure that Objective evaluation treats
all leading sample dimensions transparently rather than assuming one Monte Carlo dimension.

For differentiable Gaussian and Kronecker paths, gradients are also checked from the final
Objective values back to candidate inputs.

Sampler selection remains a posterior capability concern. The Objective consumes the resulting
sample tensor and should not branch on the model, posterior, or sampler class.


## Phase 16: Objective -> Acquisition integration

Phase 16 verifies that native BoTorch Objective objects connect directly to representative
acquisition families without a robotorchan adapter.

The scalar MC Objective integration matrix now includes:

- improvement: `qLogExpectedImprovement`,
- noisy improvement: `qLogNoisyExpectedImprovement`,
- confidence bound: `qUpperConfidenceBound`,
- constrained improvement: scalar Objective plus raw-sample outcome constraints,
- risk-aware noisy improvement: `InputPerturbation` plus native `Expectation`.

The existing Phase 10 and Phase 11 tests provide the corresponding multi-objective and
outcome-constraint coverage for qLogEHVI, qLogNEHVI, qLogNParEGO, and constrained qLogEI.

The integration boundary remains:

`samples -> Objective -> acquisition utility`.

Outcome constraints continue to receive raw model samples according to BoTorch's acquisition
contract; they are not evaluated on already scalarized Objective values.

For differentiable improvement and risk-aware paths, Phase 16 also verifies gradients from the
acquisition value back to candidate inputs.

No acquisition-specific Objective wrapper or conversion registry is required.


## Phase 17: Full E2E through acquisition optimization

Phase 17 extends the integration boundary through candidate generation:

`Model -> Posterior -> Sampler -> Objective -> Acquisition -> Acquisition Optimization -> Candidate`.

Dedicated E2E tests cover:

- a multi-output model with `GenericMCObjective` and `optimize_acqf`,
- the same Objective path with a known linear candidate constraint,
- a native categorical mixed model with `GenericMCObjective` and `optimize_acqf_mixed`.

The constraint test deliberately keeps known candidate/input-space feasibility in the optimizer.
It is not moved into the sample-space Objective or outcome-constraint layer.

The mixed test likewise keeps categorical enumeration in `optimize_acqf_mixed`; the Objective
continues to consume posterior samples without knowledge of categorical dimensions.

Existing runtime E2E coverage already exercises standard, Kronecker, and Mixed Kronecker
optimization. Phase 17 adds the Objective-specific missing combinations instead of duplicating
those model-level tests.

No robotorchan Objective-to-optimizer adapter is required.


## Phase 18: BoTorch object pass-through and custom Objective audit

Phase 18 verifies that Objective compatibility is object-level compatibility, not merely matching
tensor shapes.

BoTorch-created objects pass through robotorchan workflows directly:

- `ScalarizedPosteriorTransform` is passed to `model.posterior(..., posterior_transform=...)`,
- `LinearMCObjective` is retained as the exact Objective instance owned by the acquisition,
- `GenericMCObjective(user_function)` requires no robotorchan registration or conversion.

The custom Objective test also verifies acquisition gradients back to candidate inputs.

The source audit found no robotorchan Objective registry, Objective DSL, or conversion layer that
users must opt into. The existing `robotorchan.objectives.risk` package remains a separate
explicit-scenario aggregation utility as documented in Phase 12; it is not an Objective registry.

Therefore the public extension point for arbitrary differentiable scalar MC objectives remains
BoTorch's `GenericMCObjective`.

PosteriorTransform support is still posterior-dependent. Pass-through does not imply that every
BoTorch PosteriorTransform can operate on every posterior family; the unsupported combinations
identified in Phases 4 and 5 remain explicit rather than being hidden behind adapters.


## Phase 19: gap implementation decision

Phase 19 re-evaluated the remaining Objective and PosteriorTransform gaps using the implementation
criteria defined by this audit.

One concrete API gap is closed: `ALEBOMetricMarginalModel.posterior` now applies a supplied
BoTorch `posterior_transform` after constructing its moment-matched `GPyTorchPosterior`. This is
an exact pass-through at the model API boundary and introduces no additional approximation beyond
ALEBO's existing metric-marginal moment matching.

The other known scalarized PosteriorTransform gaps are deliberately not filled with robotorchan
adapters:

- `LinearOutputPosterior` preserves latent sampling and restored marginal moments but does not
  expose the complete joint GPyTorch distribution required by the native scalarized transform.
- `EnsemblePosterior` and `DeepGPPosterior` are empirical posterior representations. Scalar
  sample-space objectives such as `LinearMCObjective` remain the natural path.
- Kronecker scalarized PosteriorTransform support remains limited by the upstream posterior /
  transform contract. robotorchan does not replace that contract with an approximate transform.

No custom Objective registry, scalarization DSL, posterior adapter, or compatibility alias is added.
The Phase 19 implementation is therefore limited to the ALEBO pass-through gap that can be closed
without changing posterior semantics.


## Phase 20: final compatibility matrix and close audit

Phase 20 closes the Objective / PosteriorTransform audit against the current public API, runtime
tests, documentation, and CI.

Legend: **O** supported, **C** conditional / representation-dependent, **X** intentionally
unsupported, **N/A** not a distinct model-side concern.

| Capability | Standard | Multi-output | MultiTask | Kronecker | Mixed | MultiFidelity | Robust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PosteriorTransform | O | O | O | C | O | O | C |
| MC Objective | O | O | O | O | O | O | O |
| Scalarization | O | O | O | C | O | O | C |
| Outcome Constraint | O | O | O | O | O | O | O |
| Risk Objective | O | O | O | O | O | O | O |

The conditional PosteriorTransform / scalarization cells are deliberate:

- Kronecker accepts MC sample-space Objectives, while native scalarized PosteriorTransform remains
  limited by the upstream Kronecker posterior / transform contract.
- empirical `EnsemblePosterior` and `DeepGPPosterior` representations do not pretend to be
  Gaussian distributions; scalarization belongs in the MC Objective path.
- restored `LinearOutputPosterior` supports sampling in original output space but does not expose
  the complete joint GPyTorch distribution required by native posterior scalarization.
- robust workflows may use either the BoTorch `InputPerturbation` /
  `RiskMeasureMCObjective` contract or robotorchan's explicit scenario-axis contract. These
  layouts are complementary, not interchangeable aliases.

ALEBO is no longer an API gap: its metric-marginal acquisition model applies a supplied BoTorch
PosteriorTransform after constructing the moment-matched `GPyTorchPosterior`.

### Final responsibility boundaries

The final supported composition is:

`X -> model posterior -> PosteriorTransform -> sampler -> MC Objective -> outcome feasibility /
risk aggregation -> acquisition -> candidate optimization`.

Known candidate/input-space constraints remain optimizer responsibilities. Input perturbation /
scenario generation remains upstream of posterior sampling and risk aggregation.

### Final API decision

No robotorchan Objective registry, Objective DSL, scalarization wrapper, PosteriorTransform alias,
or compatibility conversion layer is required. Native BoTorch Objective and PosteriorTransform
objects remain the primary public API.

robotorchan-specific risk classes remain explicit scenario-axis aggregators rather than aliases for
BoTorch `RiskMeasureMCObjective` implementations.

Dedicated capability flags such as `supports_objective` or `supports_posterior_transform` are
not added. Runtime behavior is sufficiently represented by posterior type, acquisition capability,
and explicit compatibility tests; a broad boolean would hide transform-specific limitations.

### Close conditions

The audit is considered closed because:

1. BoTorch Objective and PosteriorTransform APIs were mapped against the supported BoTorch version.
2. Posterior, sampling, Objective, constraint, risk, acquisition, and optimizer responsibilities
   are separated and tested.
3. scalar, nonlinear, multi-output, multi-objective, MultiTask, Kronecker, Mixed, robust, and
   constrained paths have representative runtime coverage.
4. gradient, dtype, device, sample-shape, batch-shape, q, and output-axis contracts are covered.
5. BoTorch-created objects pass through without robotorchan registration or conversion.
6. the only justified model-side API gap found in Phase 19, ALEBO PosteriorTransform pass-through,
   was implemented.
7. remaining unsupported combinations have representation or upstream-contract reasons and are not
   hidden behind approximate adapters.
8. documentation, tests, and public API describe the same responsibility boundaries.
9. the Phase 19 merged head passed the repository CI matrix before this final documentation audit.

No major Objective / PosteriorTransform compatibility gap remains that justifies additional
production code in this flow.

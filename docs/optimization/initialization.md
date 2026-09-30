# Acquisition initialization compatibility

robotorchan keeps acquisition initialization aligned with BoTorch instead of maintaining a
parallel initialization framework. The optimizer should receive the same public concepts that a
BoTorch user expects: `raw_samples`, `num_restarts`, `batch_initial_conditions`,
`fixed_features`, candidate constraints, and an optional `ic_generator`.

## Compatibility matrix

| Optimization case | Automatic initialization | Explicit initial conditions | Notes |
| --- | --- | --- | --- |
| Continuous q=1 / q-batch | BoTorch native | Supported | Standard `num_restarts x q x d` contract. |
| Sequential q-batch | BoTorch native per step | Outer q-batch IC is not reused | Each greedy q=1 step initializes after pending-state update. |
| Linear inequality / equality | BoTorch polytope path | Supported | Joint inter-point constraints are supported. |
| Nonlinear inequality | Generator required when automatic feasible IC is needed | Supported | Uses BoTorch's nonlinear optimizer contract; `batch_limit=1`. |
| Mixed continuous / categorical | BoTorch mixed path | Supported | Categorical assignments remain optimizer-space concepts. |
| Fixed features / task / fidelity | BoTorch native | Supported | Fixed structural coordinates use `fixed_features`. |
| MultiTask / Kronecker | BoTorch native | Supported | No model-specific initializer is required. |
| Multi-objective | BoTorch native | Supported | Acquisition output structure does not change IC axes. |
| Input perturbation / robust | BoTorch native | Supported | Scenario axes are not appended to candidate q. |
| Model-owned dimension reduction | BoTorch native in public X | Supported | The model owns the transform; optimization remains in public X. |
| REMBO / HeSBO | BoTorch native in embedded space | Strategy-owned | Initialization follows the optimizer's embedded coordinates. |
| ALEBO | Feasible polytope sampler | Strategy-owned | Special initializer is required by the embedded polytope. |
| qKG / qMFKG | BoTorch one-shot initializer | Augmented-q IC | Keep BoTorch's dedicated KG initialization. |
| qMultiStepLookahead | robotorchan augmented-q helper | Augmented-q IC | Uses BoTorch standard initialization over the full tree batch. |
| Async `X_pending` | Same as synchronous case | Same q-shape | Pending points are acquisition context, not IC rows. |
| Fantasy model batch | Same as base acquisition | Same q-shape | Model fantasy batch is not a restart or q dimension. |
| Hybrid local refinement | Shared BoTorch backend | Global result becomes IC | Continuous local refinement uses the common backend contract. |

## Unsupported or conditional combinations

Sequential optimization with inter-point linear or nonlinear constraints is not advertised as an
automatic-initialization feature because the greedy q=1 subproblems do not preserve the original
joint-q constraint semantics. Mixed optimization does not advertise inter-point nonlinear
constraints. These cases fail explicitly instead of silently changing the mathematical problem.

For nonlinear constraints, robotorchan accepts either suitable explicit
`batch_initial_conditions` or an `ic_generator`. The generator owns feasible-start generation;
robotorchan does not add a hidden rejection sampler or repair heuristic.

## One-shot acquisitions

BoTorch already selects specialized initializers for Knowledge Gradient acquisitions. robotorchan
does not wrap or replace those paths. `qMultiStepLookahead` is different: its public requested
`q` is smaller than the augmented decision-tree batch evaluated by the acquisition. Use
`gen_augmented_one_shot_initial_conditions` as `ic_generator`, or provide explicit initial
conditions with

`acqf.get_augmented_q_batch_size(q)`

candidate rows per restart.

The helper delegates sampling and restart selection to BoTorch's
`gen_batch_initial_conditions`; it only maps public q to augmented q.

## Capability metadata

Initialization requirements are deliberately not duplicated into
`OptimizerCapabilities`. Most initialization behavior depends on the acquisition, constraint
shape, or search strategy rather than on the scalar optimizer backend alone. Adding flags such as
`supports_auto_initialization` would therefore collapse distinct contracts and could advertise
invalid combinations.

The existing optimizer capability registry continues to describe optimizer-backend behavior.
Initialization compatibility is enforced by the concrete strategy/backend APIs and executable
tests.

## Performance contract

robotorchan does not maintain a second raw-sample pool or restart-ranking implementation for the
standard BoTorch paths. `raw_samples`, `num_restarts`, and initialization options are forwarded
to BoTorch. CI checks runtime correctness rather than wall-clock thresholds, which would be
runner-dependent and unstable.

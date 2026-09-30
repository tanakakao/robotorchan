# Acquisition initialization compatibility

robotorchan keeps acquisition initialization aligned with BoTorch instead of maintaining a
parallel initialization framework. The optimizer should receive the same public concepts that a
BoTorch user expects: `raw_samples`, `num_restarts`, `batch_initial_conditions`,
`fixed_features`, candidate constraints, and an optional `ic_generator`.

## Compatibility matrix

The standard continuous, constrained, mixed, MultiTask, Kronecker, multi-objective, robust, and
model-owned dimension-reduction paths use BoTorch-native initialization. Explicit initial
conditions use the ordinary `num_restarts x q x d` shape. Fixed task or fidelity coordinates are
handled with `fixed_features`.

Sequential q-batch optimization initializes each greedy q=1 step after the pending-state update;
an outer joint-q initial-condition tensor is not reused. Joint inter-point linear constraints are
supported, while sequential inter-point constraints are conditional as described below.

Optimizer-owned embeddings initialize in their optimizer coordinates. REMBO and HeSBO use
BoTorch-native box initialization in embedded space. ALEBO is the justified exception: its
embedded feasible region is a polytope, so it uses a feasible polytope sampler.

qKG and qMFKG keep BoTorch's dedicated one-shot initialization and use augmented-q explicit
initial conditions. qMultiStepLookahead uses robotorchan's augmented-q helper because the public
requested q differs from the full decision-tree batch evaluated by the acquisition.

Async `X_pending` points remain acquisition context and are not appended to initial-condition
rows. Likewise, a fantasy model batch is a model batch dimension, not a restart or candidate-q
dimension. Hybrid continuous local refinement uses the shared BoTorch backend and receives the
global-stage result as explicit initial conditions.

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

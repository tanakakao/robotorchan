# Acquisition initialization

robotorchan keeps acquisition initialization aligned with BoTorch instead of defining a parallel
initialization framework. The optimizer owns restart and candidate axes; the acquisition and model
may add their own batch or sampling axes without changing the ordinary initial-condition contract.

## Standard contract

For ordinary acquisition optimization, automatic initialization is delegated to BoTorch.
Explicit initial conditions use shape `num_restarts x q x d`. The public optimizer arguments
`num_restarts`, `raw_samples`, `batch_initial_conditions`, `fixed_features`, and supported
candidate constraints retain their BoTorch meaning.

| Case | Initialization contract |
| --- | --- |
| Continuous q-batch | BoTorch standard initialization |
| Sequential q | Fresh q=1 initialization at each greedy step |
| Linear constraints | BoTorch feasible initialization |
| Nonlinear constraints | Explicit initial conditions or a suitable `ic_generator` |
| Mixed space | BoTorch mixed optimization initialization |
| Fixed task/fidelity | Ordinary fixed-feature initialization |
| MultiTask / Kronecker | No model-specific initializer |
| Multi-objective | No objective-specific initializer |
| Input perturbation / robust objective | Initialize nominal `q x d`; scenario axes stay internal |
| `X_pending` | Pending points are acquisition context and are not appended to initial conditions |
| Fantasy model | Fantasy batch dimensions are model batch dimensions, not restart/q axes |

Inter-point constraints are supported only where the selected BoTorch optimization mode can
preserve their joint-q semantics. In particular, sequential optimization cannot preserve an
inter-point constraint by optimizing independent q=1 subproblems.

## One-shot acquisitions

Knowledge Gradient and Multi-Fidelity Knowledge Gradient use BoTorch's specialized one-shot
initialization. Explicit initial conditions for these acquisitions use the augmented q returned by
`acq_function.get_augmented_q_batch_size(q)`, while the optimized result contains only the
requested actual candidates.

For a `OneShotAcquisitionFunction` without a dedicated BoTorch initializer, such as
`qMultiStepLookahead`, use `gen_augmented_one_shot_initial_conditions` as `ic_generator`.
The helper computes augmented q and delegates sampling and restart selection to BoTorch's
`gen_batch_initial_conditions`; it does not implement a separate heuristic.

Mixed one-shot optimization is a specialized path because categorical assignments may differ
between actual and fantasy rows. See
[mixed one-shot optimization](mixed-one-shot-optimization.md) for its narrower contract.

## High-dimensional search

Model-owned dimensionality reduction does not change optimizer coordinates: initialization remains
in the model's public/raw input space. Optimizer-owned embeddings such as REMBO and HeSBO initialize
in their search coordinates and project candidates back to the public space.

ALEBO is the intentional exception to box initialization. Its embedded feasible region is a
polytope, so it generates feasible embedded initial conditions with a polytope sampler. Sequential
ALEBO uses an initialization generator so each greedy q=1 step receives fresh feasible starts.

## Hybrid optimization

The hybrid backend passes candidates from its global stage to the common BoTorch backend as
explicit `batch_initial_conditions` for local refinement. This keeps constraint handling,
initial-condition validation, dtype/device behavior, and local gradient optimization on the same
backend contract as ordinary BoTorch optimization.

## Capability metadata

Initialization-specific capability flags are intentionally not part of `OptimizerCapabilities`.
For standard paths, initialization support follows the optimizer's existing q-batch, sequential,
mixed, fixed-feature, and constraint capabilities. Adding duplicate flags would allow metadata to
contradict the executable optimizer contract.

Specialized requirements are expressed by the API itself: nonlinear optimization may require an
`ic_generator` or explicit initial conditions, one-shot acquisitions may require augmented-q
initial conditions, and embedded strategies own initialization only when their search geometry
requires it.

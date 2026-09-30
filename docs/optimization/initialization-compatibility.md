# Acquisition initialization compatibility

This matrix is the closeout contract for acquisition initialization. It complements
[Acquisition initialization](initialization.md) by recording the executable coverage expected from
the optimizer stack.

| Problem | Initialization | Runtime contract |
| --- | --- | --- |
| Continuous q=1 / q>1 | BoTorch standard | supported |
| Explicit initial conditions | `num_restarts x q x d` | supported |
| Sequential q | fresh q=1 starts per step | supported |
| Linear intra-point constraints | BoTorch feasible starts | supported |
| Linear inter-point constraints | joint-q feasible starts | joint only |
| Nonlinear intra-point constraints | explicit starts or `ic_generator` | supported |
| Nonlinear inter-point constraints | joint feasible starts | joint only |
| Mixed continuous/categorical | BoTorch mixed initialization | supported |
| Fixed features / task / fidelity | standard fixed-feature path | supported |
| MultiTask / Kronecker | standard optimizer coordinates | supported |
| Multi-objective | standard q-batch initialization | supported |
| Robust input perturbation | nominal `q x d` starts | supported |
| Model-owned reduction | public/raw model input space | supported |
| REMBO / HeSBO | embedded box coordinates | supported |
| ALEBO | feasible polytope sampling | specialized |
| qKG / qMFKG | BoTorch one-shot initializer | specialized |
| qMultiStepLookahead | augmented-q helper or explicit augmented starts | specialized |
| Mixed one-shot | row-specific categorical enumeration | q=1 public candidate |
| `X_pending` | not part of initial-condition rows | supported |
| Fantasy model batch | not part of restart/q axes | supported |
| Hybrid | global candidate becomes local explicit start | supported |

## Required invariants

The ordinary explicit initial-condition shape is `num_restarts x q x d`. Pending observations,
fantasy-model batch dimensions, robust perturbation scenarios, and model output dimensions do not
become optimizer restart or q axes.

One-shot acquisitions are the deliberate exception: auxiliary decision variables are part of the
optimization variable, so explicit starts use the acquisition's augmented q. Candidate extraction
returns only the requested public candidates.

Sequential optimization is not a valid fallback for an inter-point constraint because independent
q=1 subproblems cannot preserve a joint-q relation. Unsupported combinations must fail explicitly
rather than silently dropping the relation.

Initialization must preserve the bounds/search-coordinate dtype and device. Gradient-based local
optimization must retain an acquisition gradient path. Geometry-specific strategies may own their
initialization only when standard box initialization cannot represent their feasible search region.

## Regression evidence

The closeout suite keeps representative runtime evidence for:

- standard and explicit q-batch optimization;
- linear and nonlinear intra/inter-point constraints;
- mixed optimization and fixed features;
- MultiTask, Kronecker, multi-fidelity, and multi-objective acquisitions;
- robust input perturbation;
- model-owned reduction and embedded search;
- ALEBO feasible initialization and gradient flow;
- qKG, qMFKG, and multi-step lookahead augmented initialization;
- pending candidates and fantasy-model batches;
- hybrid global-to-local explicit initialization.

The suite intentionally does not add timing thresholds. Initialization performance remains governed
by BoTorch's `raw_samples`, `num_restarts`, batching options, and the selected search strategy.
A wall-clock threshold in CI would be environment-dependent and would not define a stable API
contract.

## Closeout

No general robotorchan initialization framework is required. Standard initialization remains a
BoTorch responsibility. robotorchan owns only the compatibility boundary and the small number of
cases where its search geometry or a generic one-shot augmented decision space requires additional
coordination.

Future optimizer or acquisition additions should be checked against this matrix before being marked
supported. A feature is not complete when its acquisition evaluates successfully but its claimed
optimization problem cannot produce valid initial conditions.

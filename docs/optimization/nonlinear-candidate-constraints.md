# Nonlinear candidate constraints

This guide is the user-facing contract for nonlinear constraints on candidate inputs.
The detailed mathematical background is in
[Candidate Constraint Optimization](../theory/optimization/05_constraints.md).

## Candidate constraints and outcome constraints

These two constraint layers answer different questions.

| Layer | Space | Convention | Owner |
| --- | --- | --- | --- |
| Candidate constraint | proposed input X | `g(X) >= 0` | acquisition optimizer |
| Outcome constraint | modeled/sample outcome Y | `c(Y) <= 0` | constrained acquisition function |

`CandidateConstraints` represents only the first row. It does not model unknown feasibility,
replace a constrained acquisition function, or change a multi-objective reference point.

## BoTorch to robotorchan mapping

robotorchan intentionally preserves the BoTorch candidate-constraint vocabulary.

| BoTorch concept | robotorchan public surface |
| --- | --- |
| linear inequality tuple | `CandidateConstraints.inequality_constraints` |
| linear equality tuple | `CandidateConstraints.equality_constraints` |
| nonlinear `(callable, is_intrapoint)` | `CandidateConstraints.nonlinear_inequality_constraints` |
| feasible nonlinear value | `constraint(X) >= 0` |
| explicit restart points | `batch_initial_conditions` |
| custom initial-condition generator | `ic_generator` and `ic_gen_kwargs` |

`CandidateConstraints` is a container for BoTorch-compatible candidate constraint definitions.
The tuple semantics are unchanged; robotorchan does not introduce a second nonlinear DSL.

## Intra-point and inter-point callables

An intra-point nonlinear constraint receives one candidate `[d]`:

```python
def inside_circle(x):
    return x.new_tensor(1.0) - x.square().sum()


constraints = CandidateConstraints(
    nonlinear_inequality_constraints=((inside_circle, True),),
)
```

An inter-point constraint receives the complete joint q-batch `[q, d]`:

```python
def separated(X):
    return (X[0] - X[1]).square().sum() - X.new_tensor(0.01)


constraints = CandidateConstraints(
    nonlinear_inequality_constraints=((separated, False),),
)
```

The same callable shape contract is used by the BoTorch-native and common derivative-free
constraint paths. Gradient-based optimization additionally requires a differentiable callable.

## Initialization

BoTorch nonlinear candidate optimization requires feasible starting points. Supply either
`batch_initial_conditions` with shape `[num_restarts, q, d]` or a suitable `ic_generator`.

Explicit initial conditions are checked for shape, finite values, bounds, dtype, device, and
nonlinear feasibility before optimization. robotorchan does not silently drop the constraint or
fall back to unconstrained optimization when initialization fails.

## Composition

Linear inequality, linear equality, nonlinear inequality, box bounds, and fixed features may be
combined when the selected backend supports them. Candidate constraints can also be used while an
acquisition function independently applies outcome constraints.

For TuRBO, the feasible candidate set is the intersection of global bounds, current trust-region
bounds, and `CandidateConstraints`. For robust input perturbation, candidate constraints apply to
the nominal candidate; they do not imply feasibility of every perturbed scenario.

## Compatibility boundary

| Combination | Current contract |
| --- | --- |
| q=1 intra-point nonlinear | supported |
| joint q>1 intra-point nonlinear | supported |
| joint q>1 inter-point nonlinear | supported on compatible joint backends |
| greedy sequential + intra-point | supported on compatible paths |
| greedy sequential + inter-point | unsupported |
| BoTorch mixed + intra-point | supported |
| BoTorch mixed + inter-point | unsupported |
| fixed features | supported on compatible backends |
| MultiTask task feature in X | supported |
| Kronecker output tasks | no artificial task coordinate is added |
| MultiFidelity fidelity feature in X | supported |
| qLogEHVI / qLogNEHVI | compatible with candidate constraints |
| robust input perturbation | nominal-X feasibility only |
| TuRBO | local bounds and candidate constraints are intersected |
| latent / embedded search | rejected unless an exact constraint mapping exists |

Capability metadata describes backend-level support. Runtime validation remains the final authority
for conditional combinations such as inter-point nonlinear constraints with sequential search.

## dtype, device, and gradients

A nonlinear callable should keep computation in PyTorch and preserve the incoming Tensor's dtype
and device. For gradient-based optimization, do not insert `.detach()`, NumPy conversion, Python
`float` conversion, or an unconditional `.cpu()` in the constraint path. Constants can be
created with `x.new_tensor(...)`.

The BoTorch/SciPy constrained optimizer is not advertised as an end-to-end GPU optimizer merely
because a callable can evaluate CUDA Tensors.

## Failure semantics

Malformed constraint structures, unsupported combinations, invalid explicit initial conditions,
and downstream optimizer failures are distinct errors. There is no silent unconstrained fallback.

An arbitrary nonlinear feasible set is not proven non-empty in advance. A custom initializer must
still be capable of producing feasible restart points for the requested problem.

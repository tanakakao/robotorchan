# Multi-fidelity and cost-aware acquisition functions

This guide uses BoTorch's native multi-fidelity and cost-aware acquisition infrastructure.
robotorchan does not add aliases around these APIs.

## Multi-fidelity Knowledge Gradient

For continuous fidelity variables, BoTorch provides `qMultiFidelityKnowledgeGradient` (MF-KG).
The acquisition combines value of information with a projection to the target fidelity and,
optionally, a cost-aware utility.

```python
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel

cost_model = AffineFidelityCostModel(
    fidelity_weights={fidelity_dim: 1.0},
    fixed_cost=0.1,
)
cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    num_fantasies=64,
    current_value=current_value,
    cost_aware_utility=cost_utility,
    project=project_to_target_fidelity,
)
```

The projection function is part of the optimization problem: it maps a candidate to the target
fidelity used to value the final decision. It should not be hidden inside a generic acquisition
factory.

## Cost models

`AffineFidelityCostModel` is appropriate when evaluation cost is known approximately as an
affine function of fidelity. When cost must be learned, a positive cost surrogate can be used
with `InverseCostWeightedUtility`. The cost objective passed to `InverseCostWeightedUtility` must be strictly positive. A learned
cost surrogate therefore needs predictions on a positive cost scale, or an explicit positive
cost objective / transform before inverse weighting.

## Discrete fidelity

When fidelity is a finite set of allowed values, keep the acquisition native and enforce the
allowed fidelity choices in candidate optimization, for example with BoTorch mixed/discrete
optimization utilities. Do not relax a categorical fidelity into an unconstrained continuous
variable accidentally.

## Runtime-validated contract

The current integration coverage includes:

- `SingleTaskMultiFidelityGP` and `MixedSingleTaskMultiFidelityGP`;
- target-fidelity projection used by qMFKG;
- target-fidelity incumbent optimization for `current_value`;
- `AffineFidelityCostModel` with `InverseCostWeightedUtility`;
- learned robotorchan cost surrogates used by `InverseCostWeightedUtility`;
- continuous qMFKG evaluation and mixed qMFKG candidate generation.

The actual candidate fidelity remains an optimization variable. Projection to target fidelity is
used to value the terminal decision and must not silently fix the evaluated candidate itself.

## Multi-fidelity max-value entropy

Multi-fidelity MES is not part of the current runtime-validated contract. MF-KG already covers the
validated non-myopic cost-aware workflow, while adding MF-MES would require a separate executable
integration for its candidate set, fidelity semantics, cost-aware composition, and optimizer
behavior. It remains a future extension rather than a missing correctness requirement.

## Scope

This guide documents the validated acquisition integration and cost-aware utilities. The
surrogate-model layer remains responsible for choosing an appropriate multi-fidelity model and
declaring the fidelity dimensions. Multi-objective multi-fidelity lookahead and MF-MES are
intentionally not presented as validated robotorchan integration paths.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/11_multifidelity_cost_aware.md).

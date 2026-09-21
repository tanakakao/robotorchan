# Multi-fidelity and cost-aware acquisition functions

Phase 10 uses BoTorch's native multi-fidelity and cost-aware acquisition infrastructure.
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

## Scope

This phase validates acquisition integration and cost-aware utilities. The surrogate-model
layer remains responsible for choosing an appropriate multi-fidelity model and declaring the
fidelity dimensions. Multi-objective multi-fidelity lookahead is intentionally not wrapped.

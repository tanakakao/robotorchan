# Lookahead acquisition functions

This guide keeps lookahead Bayesian optimization on BoTorch's native APIs. robotorchan does not
wrap or re-export these acquisition classes because the surrogate models already satisfy the
BoTorch model contract.

## Knowledge Gradient

For one-step value-of-information optimization, use `qKnowledgeGradient` directly.

```python
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

acqf = qKnowledgeGradient(model=model, num_fantasies=64)
```

The candidate tensor passed directly to the acquisition contains the actual q candidates plus
the fantasy points required by Knowledge Gradient. In normal optimization code, prefer
BoTorch's dedicated KG optimization utilities rather than manually constructing that augmented
tensor.

## Multi-step lookahead

For explicit non-myopic decision trees, use `qMultiStepLookahead`.

```python
import torch
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.sampling.normal import SobolQMCNormalSampler

samplers = [
    SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    SobolQMCNormalSampler(sample_shape=torch.Size([16])),
]
acqf = qMultiStepLookahead(
    model=model,
    batch_sizes=[1, 1],
    samplers=samplers,
)
```

`batch_sizes` defines the future decision stages and `samplers` defines fantasy branching.
The augmented q-batch therefore grows quickly with horizon and fantasy count. Multi-step
lookahead should be reserved for cases where the additional non-myopic value justifies its
substantially higher optimization cost.

## Scope

Phase 8 validates the native BoTorch lookahead path rather than adding a robotorchan-specific
lookahead abstraction. Multi-objective and multi-fidelity lookahead variants belong to their
respective dedicated integration paths.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/06_lookahead.md).

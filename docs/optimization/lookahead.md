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

During `optimize_acqf`, BoTorch detects qKG as a one-shot acquisition and generates initial
conditions for the full augmented q-batch. robotorchan does not replace that initializer. An
explicit `batch_initial_conditions` tensor must likewise use
`acqf.get_augmented_q_batch_size(q)` rows, while the optimized result contains only the requested
candidate rows after BoTorch applies `extract_candidates`.

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

Unlike qKG, BoTorch does not automatically select a dedicated initializer for
`qMultiStepLookahead`. The ordinary initializer receives the requested public `q`, while the
acquisition evaluates the full augmented decision tree. Use
`gen_augmented_one_shot_initial_conditions` as `ic_generator`, or provide explicit
`batch_initial_conditions` with `acqf.get_augmented_q_batch_size(q)` rows. The helper only
reuses BoTorch's standard `gen_batch_initial_conditions` over the augmented batch; it does not
introduce a separate sampling heuristic.

The same helper is appropriate for custom `OneShotAcquisitionFunction` implementations that
use standard box / linear-constraint initialization but require an augmented q-batch and do not
already have a specialized BoTorch initializer. Specialized acquisitions such as qKG should keep
their native initializer instead.

## Scope

robotorchan validates the native BoTorch lookahead path rather than adding a robotorchan-specific
lookahead abstraction. Multi-objective and multi-fidelity lookahead variants belong to their
respective dedicated integration paths.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/06_lookahead.md).

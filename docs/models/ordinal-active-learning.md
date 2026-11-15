# Ordinal GP: Active Learning and candidate generation

Ordinal labels (0, 1, ..., C-1) have a **known order**. Unlike ordinary
multiclass classification, the ordered-probit model uses a single latent
Gaussian process and increasing thresholds. `OrdinalSingleTaskGPClassifier`
and `MixedOrdinalSingleTaskGPClassifier` expose the existing classification
prediction contract, so existing acquisitions can be reused without new
ordinal-specific acquisition implementations.

| Acquisition | Standard | Mixed | Batch behavior |
| --- | --- | --- | --- |
| PredictiveEntropy | Yes | Yes | q=1 only |
| MarginUncertainty | Yes | Yes | q=1 only |
| ProbabilityVariance | Yes | Yes | q=1 only |
| BALD | Yes | Yes | q=1 only |
| qPredictiveEntropy | Yes | Yes | Mean marginal entropy |
| qMarginUncertainty | Yes | Yes | Mean marginal top-two uncertainty |
| qProbabilityVariance | Yes | Yes | Mean marginal epistemic variance |
| qBALD | Yes | Yes | Joint information via label enumeration; C**q <= 256 |
| LatentStraddle / qLatentStraddle | No | No | Binary zero-boundary only |

The three marginal q-acquisitions **do not** model redundancy between
candidates. qBALD evaluates joint label information using correlated latent
posterior samples, but its cost grows exponentially with the number of
classes and candidates. For three classes, q=5 is allowed (243 label
vectors), whereas q=6 is rejected (729 vectors).

## Minimal continuous-input example

```python
import torch

from robotorchan.acquisition.active_learning.classification import qPredictiveEntropy
from robotorchan.models.classification import OrdinalSingleTaskGPClassifier
from robotorchan.optim.strategies.sobol import SobolSearchStrategy

X = torch.linspace(0, 1, 15, dtype=torch.double).unsqueeze(-1)
y = torch.bucketize(X[:, 0], torch.tensor([0.35, 0.7], dtype=torch.double))
model = OrdinalSingleTaskGPClassifier(X, y, num_classes=3)
model.fit(epochs=20, lr=0.01)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
result = SobolSearchStrategy(bounds, num_samples=128, seed=11).optimize(
    qPredictiveEntropy(model), q=2
)
print(result.candidates)
```

For Mixed inputs, specify `cat_dims` when constructing the model **and**
provide a corresponding `MixedVariableSpace` to the sampling optimizer.
The model's categorical dimensions do not automatically configure candidate
generation. `CandidateConstraints` constrain **inputs**, not ordinal
class probabilities.

Phase 3 tests cover continuous Sobol candidate generation, Mixed Sobol
sampling with a linear input constraint, q=1 and q-batch acquisitions,
and the qBALD enumeration limit. They do not establish compatibility with
every optimization backend or guarantee better selection performance.

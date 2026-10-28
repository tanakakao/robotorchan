# Heterogeneous E2E Phase 4 — Binary GP classifier

Phase 4 checks the existing `BinarySingleTaskGPClassifier` on the
Phase 2 Pass/Fail labels, without introducing a new classifier.

The benchmark provides a **true conditional probability** `p(x)` and a
distinct stochastic training label `Pass ~ Bernoulli(p(x))`. The tests
use only observed labels to fit the Bernoulli variational ELBO; the true
probability is used only for held-out diagnostics.

The integration tests cover:
- Native variational ELBO and 12 Adam optimization steps, with finite loss.
- Preservation of the original training inputs and binary labels.
- Latent posterior mean/variance, distinct from class probabilities.
- `predict_proba(X)` as `[P(Fail), P(Pass)]`, normalized and bounded.
- `predict_class(X)` returning canonical 0/1 labels.
- Sample-wise class probabilities and batched `q=3` shape contracts.
- A finite Brier-style squared probability error against held-out truth.

The Brier-style metric is a diagnostic sanity check, **not** a calibration
or superiority claim. Rigorous calibration, repeated-seed confidence
intervals, and model-comparison benchmarks are deferred to later phases.

Run:

```bash
pytest -q tests/benchmarks/test_heterogeneous_binary_classifier_e2e.py
```

# Batch / Async / Fantasization final audit

This document closes the Batch / Async / Fantasization audit against the current main branch.
Support labels are based on executable runtime evidence and explicit optimizer/model contracts.

## Model matrix

| Model family | q-batch | Fantasize | Fantasy posterior | Async E2E |
| --- | --- | --- | --- | --- |
| SingleTask exact GP | Supported | Supported | Supported | Supported |
| MultiTaskGP | Supported | Supported | Supported | Supported through compatible acquisition |
| Kronecker MultiTask GP | Supported | Unsupported | Unsupported | Partial: pending acquisition is possible, fantasy-model async is not |
| Mixed exact GP | Supported | Supported where model advertises it | Supported where fantasize is supported | Supported |
| Multi-fidelity GP | Supported | Supported where advertised | Supported where advertised | Supported |
| Reduced / high-dimensional exact GP | Supported | Supported where advertised | Supported | Supported in original input coordinates |
| Robust exact GP / input perturbation | Supported | Model-dependent | Model-dependent | Supported through compatible acquisition |
| Empirical tree ensembles | Supported | Unsupported | Not applicable | Partial: posterior-sampling batch path, no GP fantasy semantics |
| NGBoost | Posterior-path dependent | Unsupported | Not applicable | Partial |
| Deep / stochastic posterior models | Model-dependent | Model-dependent | Model-dependent | Partial |

## Acquisition matrix

| Acquisition family | q > 1 | X_pending | Joint | Sequential | Async |
| --- | --- | --- | --- | --- | --- |
| qLogEI / qEI | Supported | Acquisition-dependent | Supported | Supported | Use noisy/pending-aware acquisition when required |
| qLogNEI | Supported | Supported | Supported | Supported | Supported |
| qUCB | Supported | Acquisition-dependent | Supported | Supported | Acquisition-dependent |
| qLogEHVI | Supported | Acquisition-dependent | Supported | Supported | Acquisition-dependent |
| qLogNEHVI | Supported | Supported | Supported | Supported | Supported |
| qKG | One-shot | Separate from internal fantasy rows | Supported one-shot | Not ordinary sequential q | Supported only when acquisition/model contract permits |
| qMFKG | One-shot | Separate from internal fantasy rows | Supported one-shot | Not ordinary sequential q | Model/acquisition-dependent |
| qMultiStepLookahead | One-shot tree | Separate from lookahead fantasies | Supported by BoTorch semantics | Not ordinary sequential q | Model/acquisition-dependent |
| robotorchan q=1 AL acquisitions | Unsupported unless documented | Not generally claimed | Unsupported | Can use external sequential composition only when semantically valid | Not generally claimed |

## Search-space matrix

| Search space | Batch | Async / pending | Fantasize |
| --- | --- | --- | --- |
| Continuous | Supported | Supported | Model-dependent |
| Mixed | Supported | Supported | Model-dependent |
| Linear candidate constraints | Supported | Supported | Orthogonal to model fantasy |
| Nonlinear candidate constraints | Supported for continuous / compatible backends | Supported where optimizer supports them | Orthogonal |
| Inter-point nonlinear + native BoTorch mixed | Unsupported | Unsupported at optimizer layer | Orthogonal |
| Multi-fidelity | Supported | Supported | Model-dependent |
| Robust input perturbation | Supported | Supported | Independent perturbation and fantasy axes |
| Mixed one-shot qKG / qMFKG | q=1 actual candidate only | Model/acquisition-dependent | Acquisition-internal fantasies supported |

## Final decisions

- No custom asynchronous scheduler is added.
- No model-level `supports_X_pending` flag is added.
- Kronecker fantasization remains explicitly unsupported instead of relying on private
  GPyTorch prediction-cache manipulation.
- Mixed one-shot optimization remains correctness-first at `q=1`; larger q requires a scalable
  categorical search design rather than unbounded exact enumeration.
- Non-GP models are not required to emulate ExactGP fantasy conditioning.
- BoTorch-native batch, pending, fantasy, and one-shot semantics remain the default path.

The audit is closed when CI for this documentation state passes. Future work should be opened as a
new capability project rather than silently widening these contracts.

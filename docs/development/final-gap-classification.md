# Final gap classification

This document closes the repository-wide audit by separating correctness work from deliberate
limitations and future extensions. It does not expand the public API.

## Classification

| Area | Class | Final status | Evidence / boundary |
| --- | --- | --- | --- |
| Model registry and public exports | Complete | No open A/B gap | Registry, public API tests, and generated documentation coverage are synchronized. |
| Raw training-data and training API contracts | Complete | No open A/B gap | Exact, variational, fully Bayesian, pairwise, and non-GP training paths have explicit contracts. |
| Cross-capability model metadata | Complete | No open A/B gap | Mixed, MultiTask, high-dimensional, robust, multi-fidelity, structured, preference, and non-GP semantics are explicit. |
| Acquisition compatibility | Complete | No open A/B gap | Output arity, posterior sampling, fantasy, multi-fidelity, ensemble, and structured-output requirements are checked explicitly. |
| Candidate optimization | Complete for current public contract | No open A/B gap | Continuous, mixed, qKG, and qMFKG representative candidate-generation paths are executable. |
| Multi-objective / constraints | Complete for current public contract | No open A/B gap | qLogEHVI, qLogNEHVI, and qLogNParEGO integration is covered; constraint support remains BoTorch composition. |
| Ensemble / non-GP acquisition | Complete for current public contract | No open A/B gap | Gaussian and empirical ensemble posterior sampling semantics are separated; empirical ensembles use IndexSampler semantics. |
| Multi-fidelity advanced workflow | Complete for current public contract | No open A/B gap | MF-KG, cost-aware utility, target projection, and mixed qMFKG semantics are documented and tested. |
| Runtime E2E validation | C | Representative coverage complete | Robust and broad multi-output combinations are not exhaustively optimizer-tested. This is validation depth, not a known defect. |
| q > 1 mixed one-shot optimization | D | Intentionally unsupported | Exact row-wise categorical fantasy enumeration currently has a q=1 public contract. |
| Large mixed one-shot assignment spaces | D | Intentionally unsupported | Enumeration is bounded and fails explicitly beyond the configured correctness-first limit. |
| Empirical-ensemble local Active Learning | D | Intentionally unsupported | Local AL acquisitions reject EnsemblePosterior until an explicit reduction semantics is defined. |
| Multi-fidelity × high-dimensional model | D | Intentionally unsupported | No combined public model contract is declared. |
| Multi-fidelity × robust model | D | Intentionally unsupported | No combined public model contract is declared. |
| Hypervolume Knowledge Gradient integration | E | Future extension | Theory/native capability does not imply robotorchan integration support. |
| Multi-fidelity MES / MUMBO integration | E | Future extension | MF-KG is the validated current multi-fidelity lookahead path. |
| Broader constrained/custom-objective MO examples | E | Future extension | Current compatibility supports BoTorch composition without owning additional wrappers. |
| Broader robust / MultiTask optimizer E2E matrix | C | Optional validation extension | Add when a concrete workflow needs stronger regression protection. |

## Definitions

- **A — correctness issue:** implemented behavior is wrong. Must be fixed before audit closure.
- **B — contract inconsistency:** public API, metadata, documentation, or runtime behavior disagree.
  Must normally be fixed before audit closure.
- **C — validation gap:** supported behavior lacks exhaustive validation. Prioritize by workflow
  importance; representative coverage can be sufficient.
- **D — explicit limitation:** deliberately unsupported behavior. Explicit failure plus
  documentation is considered complete.
- **E — future extension:** useful new capability that is not required for correctness or current
  contract completeness.

## Closure decision

No known class A or class B gaps remain after the repository-wide audit.

The remaining class C items are coverage-depth improvements. They do not currently justify
creating additional model classes, acquisition wrappers, or optimizer APIs. The class D items are
deliberate public boundaries and must remain explicit rather than being inferred as supported.
Class E items belong to future feature work and should receive their own design and runtime
validation when implemented.

The audit can therefore close after this classification if CI remains green. Future models,
acquisitions, and optimizers should be integrated through the existing registry, compatibility,
runtime-test, documentation, and generated-coverage contracts instead of reopening completed
phases wholesale.

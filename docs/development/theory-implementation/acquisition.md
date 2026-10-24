# Acquisition theory–implementation mapping: Phase 3 inventory

## Sources and evidence boundary

- Baseline: `main` after the Phase 2 exit-check merge `b83afcc67e38d2b4dcc3c3a3ab794063f61705f6`.
- Public local exports: `src/robotorchan/acquisition/__init__.py`.
- Capability metadata: `src/robotorchan/acquisition/registry.py`.
- Repeatable signature inventory: `scripts/audit_acquisition_api.py`.
- Cross-layer compatibility: `acquisition/optimizer_compatibility.py`.
- Composition and semantic bridges: `acquisition/composition.py`.

This is a **source-inspected static inventory**, not a claim that
each acquisition has a numerically verified implementation or compatible
optimizer. Phase 5 adds per-method theory mapping; Phase 8 adds independent
mathematical checks.

## Distinct implementation paths

| Method group | Public/implementation ownership | Mathematical object or route | Evidence and limitation |
| --- | --- | --- | --- |
| `PosteriorVariance`, `PosteriorStd`, `Straddle`, `RandomizedStraddle`, `BoundaryVariance` | Locally exported classes; registered as active learning | GP marginal moments and boundary utilities | Registry declares `max_q=1`; verify individual formulas and output-axis semantics in Phase 5/8 |
| `ExpectedPredictiveInformationGain` | Local active-learning acquisition | Joint Gaussian information criterion | Registry explicitly requires single output and joint Gaussian posterior |
| `PredictiveEntropy`, `MarginUncertainty`, `ProbabilityVariance`, `BALD`, `LatentStraddle` | Local classification AL classes | Class probability / label uncertainty / latent score are **distinct targets** | Registered `max_q=1`, single output; probabilities cannot silently be replaced by latent GP moments |
| `qLogExpectedImprovement`, `qLogNoisyExpectedImprovement`, `qProbabilityOfImprovement`, `qUpperConfidenceBound`, `qSimpleRegret` | BoTorch-native; **metadata only**, no local class wrapper | Monte Carlo BO objectives | Presence in registry does not expose these as local public imports or verify optimizer capability |
| `qKnowledgeGradient`, `qMultiFidelityKnowledgeGradient` | BoTorch-native | Fantasy-based one-shot lookahead | Registry flags `one_shot=True`, `requires_fantasize=True`; multi-fidelity KG additionally needs fidelity-aware model |
| `qLogExpectedHypervolumeImprovement`, `qLogNoisyExpectedHypervolumeImprovement`, `qLogNParEGO` | BoTorch-native | Multi-objective / scalarized Monte Carlo BO | Constraints/multi-objective flags are metadata; inspect actual objective and partitioning in Phase 5 |
| `make_qei_acquisition`, `make_qnei_acquisition`, `make_qehvi_acquisition`, `make_qnehvi_acquisition` and `make_*constrained*` | Local construction helpers in `acquisition/composition.py`, not acquisition subclasses | Semantic model/objective/constraint adapters feeding BoTorch acquisitions | Helpers must not be counted as additional independent acquisition algorithms |
| `select_thompson_candidates` | `acquisition/sampling` exported helper | Posterior-sample selection | Search procedure, not a registered standalone MC acquisition class |

`scripts/audit_acquisition_api.py` distinguishes local acquisition
subclasses, supporting exports, composition helpers, and native BoTorch
registry entries. Its `gaps` keys reflect **registry/export mismatch**
rather than unsupported algorithms. The registry is not the exhaustive
inventory of BoTorch methods, and alternative choices (including native
upstream call sites) must be traced in Phase 3 before declaring coverage.

## Acquisition/optimizer interaction

`check_acquisition_optimizer_compatibility` checks registered `max_q`
and whether a one-shot acquisition is routed to `MIXED_ONE_SHOT`.
It is not an optimizer implementation, nor a proof of all nonlinear
candidate-constraint compatibility. In particular, a one-shot augmented
batch cannot safely be passed to the ordinary mixed optimizer as if its
categorical assignment were row-independent.

## Fixed Phase 3 completion criteria

1. Inventory local acquisition exports, capability registry entries,
   composition helpers and BoTorch-native call sites without conflation.
2. Inventory named dispatch optimizers, direct low-level backends, search
   strategies and standalone optimizer utilities beyond dispatch.
3. Trace candidate constraints (linear, nonlinear, intra-/inter-point),
   outcome constraints, mixed domains, q-batch and one-shot boundaries.
4. Resolve concrete implementation modules and representative tests,
   including unsupported and unverified behavior.
5. Run relevant inventory and CI checks, then make an explicit completion
   decision. Numerical identities belong to Phases 8–9.

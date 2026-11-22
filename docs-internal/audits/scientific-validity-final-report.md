# Scientific validity audit — consolidated final report

Status: **scoped audit complete with explicit limitations** (2026-11-22).

This report closes the **bounded scientific-validity audit**, not a claim that every
combination of models, acquisition functions, optimizers, constraints and datasets
has been proven correct. It consolidates *existing* evidence; no new benchmark,
implementation change, or test expansion is required to issue this assessment.

## Evidence and dispositions

| Domain | Disposition | Evidence | Limit |
| --- | --- | --- | --- |
| GP surrogate posterior / likelihood / variational fitting | **Targeted verification; defects fixed** | [Phase 2](scientific-validity-phase2.md); PRs #1954–#1958 | No universal convergence or uncertainty-calibration certification |
| Binary GP classification | **Targeted verification; empirical limitation characterized** | Phase 2 binary probit reference; [ten-seed study](scientific-validity-phase5-gp-al-full-results.md); [epoch sensitivity](scientific-validity-phase5-gp-al-epoch-sensitivity-results.md) | One synthetic AL boundary; fitting budget materially changes Brier |
| Multiclass GP classification | **Targeted verification** | Phase 2 class-wise posterior and synthetic learning references (PRs #1957, #1958); [Phase 3](scientific-validity-phase3.md) qBALD joint-label reference (PR #1961) | Not a universal calibration or all-class-count guarantee |
| Ordinal GP regression | **Limited / targeted coverage** | Phase 2 ordinal probability-posterior references and `tests/models/classification/test_ordinal_gp.py` | Full likelihood quadrature, convergence and calibration across settings not established; no defect inferred |
| BO and AL acquisition mathematics | **Targeted verification** | [Phase 3](scientific-validity-phase3.md); analytic Gaussian, MC, regression/classification AL, EPIG and joint-entropy references | Marginal q aggregations are not joint information gain; MC error and model assumptions remain |
| Optimizers, mixed spaces, q-batch and candidate constraints | **Targeted verification; error contracts checked** | [Phase 4](scientific-validity-phase4.md); PRs #1965–#1969 | Sampled-pool optimum is not global optimum; no full optimizer × constraint Cartesian certification |
| Closed-loop BO empirical reliability | **Measured with scoped fairness/provenance checks** | [Phase 5 inventory](scientific-validity-phase5-inventory.md); [cross-q pairing](scientific-validity-phase5-all-strategy-pairing.md) | Branin/Hartmann6 only; cross-q performance comparison remains exploratory |
| Closed-loop GP active learning | **Measured; performance trade-off characterized** | [full study](scientific-validity-phase5-gp-al-full-results.md), [epoch sensitivity](scientific-validity-phase5-gp-al-epoch-sensitivity-results.md) | Accuracy and Brier favor different policies; no general ranking or defect conclusion |
| Advanced extensions (Safe, Transfer, Cost-aware/Multi-Fidelity, TuRBO, Preference, Causal, etc.) | **Existing implementation/CI evidence; no new blanket certification** | Feature-specific development and regression records; not independently certified as a whole by Phases 2–5 | Individual guarantees and empirical performance must not be inferred from unrelated representative tests |
| Exhaustive numerical stability, OOD calibration, hardware/version portability | **Unverified as universal claims** | Representative tests and benchmark limitations above | Not a release-blocking defect absent a concrete failing contract |

## Findings that required correction

- Multiclass inherited MLL ownership contract: fixed in PR #1954.
- Joint heteroskedastic likelihood variance-floor consistency: fixed in PR #1955.
- Variational cache lifecycle and state restoration: addressed in PR #1956.
- Selected optimizer infeasibility handling: tested with independent impossible
  q=2 constraints in PRs #1968 and #1969.

The above are concrete scoped findings, not a statement that no other defects exist.

## Empirical findings and correct interpretation

1. The archived BO experiments cover Branin and Hartmann6 with q=1/q=3,
   four strategies and ten seeds per cell. Direct archived-trajectory checks
   confirmed 80/80 cross-q strategy/seed initial-design pairs and budget
   invariants. This does not prove global algorithm superiority.
2. The ten-seed GP active-learning study compared Random, Sobol and
   PredictiveEntropy with the same GP predictor and acquisition budget.
   At eight fit epochs, PredictiveEntropy improved mean endpoint Accuracy
   relative to Random but worsened mean Brier score.
3. The completed eight-versus-32-epoch sensitivity run
   [#44982094508](https://github.com/tanakakao/robotorchan/actions/runs/44982094508)
   improved mean Brier in **all three** policies at 32 epochs. At 32 epochs,
   PredictiveEntropy still has the highest mean Accuracy (0.892676) and
   highest/worst mean Brier (0.092995); Random Brier is 0.082234 and Sobol
   Brier is 0.079394. The fit budget matters, but this does not establish
   convergence, statistical significance, a general policy ranking, or a defect.
4. These studies are empirical evidence about the tested protocols; they do
   not establish mathematical validity for every other library feature.

## Completion criteria and explicit boundaries

- **Completed:** Consolidate the existing mathematical reference checks,
  corrected defects, representative optimizer feasibility contracts, archived
  empirical evidence, paired-seed checks and training-budget sensitivity.
- **Completed:** Distinguish Verified-in-targeted-cases, Defect-fixed,
  Justified/contractual limitation and Unverified; avoid treating absence of
  evidence as evidence of failure.
- **Completed:** Make multiclass and ordinal coverage explicit, rather than
  allowing the binary GP benchmark to stand in for all classification models.
- **Not claimed:** Exhaustive feature-matrix certification, universal numerical
  convergence, calibrated posterior uncertainty for all models, global
  optimizer optimality, superiority of acquisition policies, or all-platform
  reproducibility.
- **Not performed in this closure:** New experiments for every advanced
  extension or every possible q/mixed/constraint combination. These are
  separately scoped future research or maintenance tasks, **not an
  indefinitely expanding condition for this audit to close**.

## Release-oriented assessment

**Decision: the scoped scientific-validity audit can be closed.** The available
evidence supports targeted mathematical and numerical correctness of the
documented representative behaviors, with concrete defects corrected and
remaining limitations recorded. It does **not** independently certify
unexamined features or substitute for packaging, security, compatibility and
release-process checks. A newly reproducible correctness defect should be
handled as a focused issue/PR, not as automatic reopening of the entire audit.

This is an evidence-based scope decision, not a proof of total correctness.

# Scientific validity audit — Phase 3: acquisition and active-learning mathematics

Status: **scoped audit complete; limitations retained** (2026-11-21).

This report documents targeted mathematical reference checks for acquisition functions. It does **not** certify every acquisition, optimizer, constraint, mixed-variable configuration, or statistical performance regime. Phase 2 GP-model validity is documented separately in [scientific-validity-phase2.md](scientific-validity-phase2.md).

## Evidence and interpretation

| Area | Evidence | Interpretation and boundary |
| --- | --- | --- |
| Analytic Gaussian BO (EI/PI/UCB) | `tests/acquisition/test_phase8_analytic_bo_gaussian_reference.py` | Fixed Gaussian references; does not prove optimality of candidate search |
| MC qUCB / qSimpleRegret | `test_phase8_mc_qucb_reference.py`, `test_phase8_mc_simple_regret_reference.py` | Sample aggregation checked; finite-sample MC error remains |
| Regression AL marginal uncertainty | `test_phase8_regression_al_formula_reference.py`, `test_batch_marginal_variance.py` | qPosteriorVariance averages marginal variances; qPosteriorStd takes the square root **after** averaging variances, not the mean of standard deviations |
| Binary AL scores and BALD | `test_phase8_classification_al_reference.py`, `test_phase8_bald_probability_variance_reference.py` | Selected predictive probability and information-score references; depends on classifier posterior approximation |
| Multiclass qBALD | PR #1961; `test_phase3_multiclass_qbald_reference.py` | Independent explicit enumeration of 3² joint labels for q=2; conditional label independence **given each posterior draw** is an assumption of the estimator; finite posterior samples are not an exact posterior expectation |
| Gaussian joint/conditional EPIG | PR #1960; `test_phase3_epig_logdet_reference.py` | Independent Gaussian log-determinant and conditional-MI references in targeted covariance settings |
| Randomized straddle | PR #1962; `test_phase3_randomized_straddle_reference.py` | Independently seeded inverse-CDF coefficient, q=3 aggregation, reuse and resampling |
| Gaussian joint entropy | PR #1963; `test_phase3_joint_entropy_closed_form.py` | Independent bivariate closed-form determinant; positive, negative and zero covariance |
| Boundary-variance heuristic | `test_phase8_regression_al_formula_reference.py` | Formula reference, **not** a proof that the heuristic is an optimal information-gain criterion |
| EPIG marginal q aggregation | `src/robotorchan/acquisition/active_learning/epig.py` | `qExpectedPredictiveInformationGain` averages per-point scores; use `qJointExpectedPredictiveInformationGain` for joint Gaussian batch information |
| Classification marginal q aggregation | `src/robotorchan/acquisition/active_learning/classification.py` | `qPredictiveEntropy`, `qProbabilityVariance`, `qMarginUncertainty` average marginal scores, not joint batch information; `qBALD` computes joint label MI by enumeration |
| EPIG observation model | `src/robotorchan/acquisition/active_learning/epig.py` | Gaussian noisy-observation MI; not an automatic guarantee for non-Gaussian labels, heteroskedastic observation noise, or arbitrary observation models |

## Mathematical and statistical caveats

1. **Joint versus marginal q acquisition:** An average of marginal scores does not account for redundant or correlated batch observations. A q>1 API does not by itself imply joint information gain.
2. **Classification qBALD:** For C classes and q locations, exact enumeration of C^q label vectors is used (subject to a 256-vector limit), but expectation over latent posterior draws is Monte Carlo. Class-probability calibration and the posterior approximation remain separate validation questions.
3. **Gaussian EPIG:** Mutual-information log-determinant references are valid under the stated joint Gaussian observation model. Results must not be generalized to classification or arbitrary likelihoods without further derivation.
4. **Randomized straddle:** A deterministic seeded numerical reference validates the implemented sampling transform and score, not the theoretical superiority of the selection policy.
5. **Numerical stability:** Cholesky, log-determinant and near-singular covariance behavior depend on floating-point precision and jitter; the targeted references are not exhaustive stress tests.
6. **Optimization is separate:** Agreement between a score and its reference does not establish that a downstream optimizer locates its global maximizer or respects every constraint.

## Remaining scientific-validation work

- Independent calibration and coverage experiments for posterior uncertainty, including classification probabilities.
- Multi-seed end-to-end regret, information gain, and sample-efficiency benchmarks against meaningful baselines.
- Targeted stress tests for near-singular covariances, large q, mixed-variable spaces, and constrained optimizers, prioritized by demonstrated risk.
- Review of scalar-versus-tensor observation-noise contracts across EPIG variants before claiming uniform behavior.
- Explicitly record any unsupported acquisition/model combinations rather than interpreting an untested path as verified.

## Phase 3 exit decision

**Close the scoped acquisition-formula audit with limitations.** Independent mathematical references now cover representative analytic BO, MC BO, regression AL, classification AL, Gaussian EPIG, randomized straddle and joint entropy. No universal correctness, numerical calibration, or optimization-convergence claim is made. Proceed to the next audit phase focused on optimizer/constraint integration and end-to-end reliability, without adding redundant unit tests solely to increase test count.

## PR references

- https://github.com/tanakakao/robotorchan/pull/1960
- https://github.com/tanakakao/robotorchan/pull/1961
- https://github.com/tanakakao/robotorchan/pull/1962
- https://github.com/tanakakao/robotorchan/pull/1963

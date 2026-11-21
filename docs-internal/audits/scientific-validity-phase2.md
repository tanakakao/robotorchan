# Scientific validity audit — Phase 2: GP models and posterior semantics

Status: **completed with documented limitations** (2026-11-20)

Scope: Gaussian-process surrogate models, likelihood and variational objectives, posterior assembly, and focused numerical reference checks. This is **not** a proof of correctness for every model variant, every optimizer, or every data regime. Phase 3 acquisition functions and later phases remain separate.

## Findings and dispositions

| Area | Disposition | Evidence and boundary |
| --- | --- | --- |
| Exact GP posterior / covariance | Verified in targeted cases | `tests/models/standard/test_phase7_exact_gp_reference.py`, `test_phase7_multitask_covariance_reference.py`, `test_phase7_kronecker_covariance_reference.py`; does not cover all transforms and kernels |
| Mixed and multifidelity kernel behavior | Verified in targeted cases | `test_phase7_mixed_categorical_reference.py`, `test_phase7_multifidelity_kernel_reference.py`; broad posterior calibration remains unverified |
| Binary probit classification | Verified in targeted cases | `tests/models/classification/test_phase7_binary_probit_reference.py`; optimization convergence is not established for every binary variant |
| Multiclass class-wise posterior | Verified in targeted cases | PR #1957; independent class mean/covariance blocks and zero cross-class blocks for standard and Mixed inputs |
| Multiclass variational learning | Verified in a synthetic case | PR #1958; posterior-mean cross-entropy improves on a separated three-class dataset, standard and Mixed; this is not a general ELBO convergence guarantee |
| Multiclass MLL ownership | Defect fixed | PR #1954; unsupported inherited `make_mll()` now explicitly rejected rather than exposing an unrelated latent model |
| Joint heteroskedastic Gaussian log likelihood | Defect fixed | PR #1955; variance floor now consistently affects Gaussian normalization and quadratic terms; independent seeded MC reference with both KL contributions |
| Joint heteroskedastic cache lifecycle | Defect / maintenance risk addressed | PR #1956; direct private `_clear_cache()` calls replaced, with Mixed state round-trip regression |
| Ordinal classification and probability moments | Limited / targeted coverage | `tests/models/classification/test_ordinal_gp.py`, `test_probability_posterior_ordinal_multiclass.py`; full likelihood quadrature and calibration across settings not established |
| Variational ELBO minibatch scaling | Supported by implementation contract and targeted tests | `tests/models/contracts/test_variational_fit.py`, `tests/models/standard/test_variational_gp.py`; no universal convergence claim |
| End-to-end uncertainty calibration and coverage | Unverified across the model inventory | Requires independent multi-seed calibration experiments and coverage metrics |

## Important interpretation

- **Verified in targeted cases** means an explicit reference or regression assertion exists for the specified behavior. It does not imply all supported shapes, hyperparameters, mixed dimensions, q-batches, or hardware are covered.
- **Defect fixed** means a concrete implementation or public contract error was corrected and regression coverage added. The evidence should be rechecked against future BoTorch/GPyTorch upgrades.
- **Limited** or **Unverified** means the audit did not establish a broad correctness claim; it does not by itself imply the implementation is defective.
- A decrease in posterior-mean classification cross-entropy after fitting is a useful deterministic learning signal, but it is **not** a direct test of monotonic ELBO improvement or convergence to a global optimum. Monte Carlo ELBO training is stochastic.
- A posterior constructed from independent latent class GPs should have zero **latent cross-class covariance**. Softmax class probabilities remain dependent because they are normalized jointly.

## Phase 2 exit criteria

1. Major identified defects in MLL ownership and heteroskedastic likelihood: **fixed** (PRs #1954, #1955).
2. Private variational cache clearing and state restoration: **addressed** (PR #1956).
3. Independent numerical check of multiclass latent posterior: **added** (PR #1957).
4. Reproducible multiclass learning-signal regression: **added** (PR #1958).
5. Exhaustive scientific validation of every GP variant: **not claimed**; record limitations for later benchmark and release audits.

**Decision:** Phase 2 can be closed for the scoped scientific audit, with the explicit limitations above. Proceed to Phase 3 (acquisition and active-learning mathematics). Do not silently promote unverified variants to a universal \"verified\" status.

## PR references

- https://github.com/tanakakao/robotorchan/pull/1954
- https://github.com/tanakakao/robotorchan/pull/1955
- https://github.com/tanakakao/robotorchan/pull/1956
- https://github.com/tanakakao/robotorchan/pull/1957
- https://github.com/tanakakao/robotorchan/pull/1958

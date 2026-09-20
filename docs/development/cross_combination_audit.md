# Cross-combination model audit

Phase 14 audits public model coverage across structural axes instead of creating every Cartesian-product class.

| Family | Single | Mixed | MultiTask | Mixed × MultiTask | Kronecker | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Standard exact GP | yes | yes | yes | yes | yes | complete |
| Reduced GP | yes | yes | yes | common base | yes | complete via shared reducer bases |
| DKL / JointEncoder | yes | yes | yes | yes | yes | complete |
| SAAS fully Bayesian | yes | yes | yes | yes | no dedicated variant | no automatic Kronecker expansion |
| Robust relevance pursuit | yes | yes | yes | yes | no | keep long-format robust contract |
| Heteroskedastic | yes | yes | yes | yes | no | keep long-format noise model |
| Student-t / contaminated | yes | yes | yes | yes | no | variational likelihood; no automatic Kronecker expansion |
| Nonstationary | yes | yes | yes | yes | no | long-format task covariance is canonical |
| DeepGP | yes | yes | yes | no | no | Mixed × MultiTask remains a meaningful future combination |
| Infinite-width BNN | yes | yes | yes | no | no | Mixed × MultiTask remains a meaningful future combination |
| Spectral mixture | yes | yes | yes | no | no | Mixed × MultiTask remains a meaningful future combination |

## Audit rules

A missing cell is not automatically a missing implementation. A cross-product model is added only when its statistical semantics and public API are distinct and useful. Kronecker variants require aligned multi-output observations and are not aliases for long-format MultiTask models.

The remaining expressive gap is Mixed × MultiTask for DeepGP, infinite-width BNN, and spectral-mixture families. These combine two independently useful axes and should be evaluated as explicit implementations rather than generated wrappers. Phase 14 records the gap; it does not introduce compatibility aliases or speculative Cartesian-product classes.

## Public-contract checks

Every new public model must be reflected in `robotorchan.models.__all__`, the explicit public-model inventory, the Mixed inventory when applicable, and `docs/model_coverage.json`. Runtime support claims require executable posterior tests; acquisition compatibility claims require an executable acquisition path rather than construction-only tests.

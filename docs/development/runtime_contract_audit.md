# Runtime contract audit

Phase 17 verifies executable runtime contracts for previously implemented model families.

## Scope

- Robust relevance-pursuit MultiTask and Mixed MultiTask posterior / acquisition semantics.
- Nonstationary MultiTask and Mixed MultiTask posterior / acquisition execution.
- Heteroskedastic MultiTask fitting, predicted-noise, posterior, and acquisition paths.
- Heavy-tail MultiTask task-label and public posterior assumptions.
- Public exports, inventories, model coverage, and documentation consistency.

## Audit rule

A public runtime capability is supported only when an executable test exercises that path. Tests must not be weakened to hide implementation defects. If a model cannot satisfy a claimed posterior or acquisition contract, fix the implementation or document the operation as unsupported.

## Initial findings

- Robust relevance-pursuit MultiTask tests currently verify construction and conversion to a standard MultiTaskGP, but do not execute the robust model posterior or an acquisition function.
- Nonstationary MultiTask tests currently verify kernel structure and local lengthscales, but do not execute posterior or acquisition paths.
- Heteroskedastic MultiTask tests verify the pre-fit contract; fit/noise/posterior integration requires explicit audit coverage.

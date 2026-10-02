# Notebook Public API Audit

## Scope

This audit covers the user-facing notebooks under `examples/notebooks/` after the executable
documentation expansion through Phase 23.

The audit checks that notebooks:

- import robotorchan functionality from documented public package surfaces;
- use BoTorch APIs directly when robotorchan intentionally delegates that responsibility;
- do not depend on private or backend implementation modules;
- do not add wrappers solely for cosmetic API uniformity;
- keep model, acquisition, optimization, sampling, and workflow responsibilities explicit.

## Public surfaces used by notebooks

The canonical robotorchan notebook imports are package-level public surfaces such as:

- `robotorchan.models`;
- `robotorchan.acquisition`;
- `robotorchan.objectives`;
- `robotorchan.optim`.

Direct BoTorch imports remain intentional for upstream-native contracts including acquisition
functions, posterior transforms, samplers, cost models, input transforms, and initial-condition
generators when robotorchan does not own a distinct abstraction.

## Finding: internal optimizer backend import

`03_multi_fidelity_gp.ipynb` imported
`robotorchan.optim.backends.optimize_acqf_botorch` directly.

That backend module is an implementation detail. The notebook now uses the public dispatcher:

```python
from robotorchan.optim import optimize_acqf as optimize_robotorchan_acqf

candidate, value = optimize_robotorchan_acqf(
    acq_function=mf_acqf,
    bounds=mf_bounds,
    q=1,
    optimizer="botorch",
    num_restarts=3,
    raw_samples=32,
)
```

This preserves the one-shot-compatible BoTorch backend while keeping the executable example on the
supported public API.

## BoTorch-first conclusion

No additional robotorchan wrapper is required for the standard BoTorch APIs demonstrated in the
notebooks. In particular, direct use of BoTorch acquisition classes, samplers,
`PosteriorTransform`, MC objectives, input perturbations, and acquisition initializers is
intentional where robotorchan delegates those contracts upstream.

Public API consistency means stable ownership boundaries, not replacing every BoTorch import with a
robotorchan alias.

## Review rule

Future notebook changes should fail review if they require imports from implementation-oriented
paths such as optimizer backends or other non-public modules when an equivalent package-level
public API exists.

The authoring standard in `notebook-authoring-standard.md` remains the normative rule; this file
records the Phase 24 audit result.

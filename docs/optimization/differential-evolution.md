# Differential Evolution optimizer backend

Phase 6 adds Differential Evolution (DE) as a derivative-free acquisition
optimizer. It uses SciPy `differential_evolution` and keeps the BoTorch
`AcquisitionFunction`, tensor bounds, and q-batch concepts.

A joint `q x d` candidate is flattened to a `q*d` DE decision vector. Each
objective evaluation reconstructs the tensor on the acquisition function device
and dtype, evaluates the acquisition, and minimizes its negative value.

The backend is intentionally CPU-orchestrated because SciPy owns the evolutionary
loop. Acquisition evaluation can still execute on the model tensor device, but
Phase 6 does not advertise GPU capability or vectorized population evaluation.

Candidate constraints are rejected explicitly in Phase 6. Constraint translation
is handled in the dedicated cross-optimizer constraint phase. SciPy polishing is
disabled by default so the DE backend remains derivative-free and its semantics
do not silently change to a local optimizer.

```python
from robotorchan.optim.backends import optimize_acqf_de

candidates, value = optimize_acqf_de(
    acq_function=acq,
    bounds=bounds,
    q=3,
    seed=123,
    options={"maxiter": 200, "popsize": 15},
)
```


## Integer and mixed continuous/integer variables

Differential Evolution now supports continuous plus integer search spaces. Integer
coordinates remain part of the DE decision vector, but every acquisition and
constraint evaluation repairs those coordinates to the nearest legal integer and
clips them to the integer domain defined by the bounds. The returned candidate is
repaired with the same rule.

This is intentionally limited to continuous and integer variables in this phase.
Categorical variables are rejected rather than treating category labels as a
numeric metric. Categorical-aware DE mutation is evaluated separately before it
can be advertised as supported.

Both the direct `integer_dims` backend argument and a `MixedVariableSpace`
containing only continuous/integer variables are supported. Supplying both is
rejected to keep a single source of search-space semantics.


## Categorical and full mixed-variable scope

Phase 10 extends the structured repair rule to categorical coordinates. SciPy DE
continues to evolve a numeric proposal vector, but categorical coordinates are
never passed to the acquisition function or constraints as interpolated category
values. Before every evaluation, each categorical proposal is mapped to the
nearest legal value from the explicit category set. Integer coordinates are
repaired independently by rounding and clipping.

This makes the evaluated search domain discrete and legal for continuous,
integer, categorical, and combined mixed spaces. The category labels are used
only to partition the proposal coordinate into deterministic attraction regions;
the acquisition function observes only legal category values. This is not a
claim that category labels possess a meaningful metric or that SciPy's
differential mutation itself is categorical.

The same repair is applied to the returned candidate. Mixed-variable-dependent
`CandidateConstraints` are evaluated only after repair, so their callable sees
the same legal raw coordinates as the acquisition function.


## Constraint handling

Differential Evolution no longer combines acquisition value and constraint
violation with a fixed penalty coefficient. Candidate constraints are exposed to
SciPy DE through a nonlinear feasibility constraint built from robotorchan's
common violation evaluator. The DE population therefore follows feasibility
before objective quality rather than relying on an acquisition-scale-dependent
penalty.

Structured-domain repair and fixed-feature application happen before both
acquisition and feasibility evaluation, so continuous/integer/categorical mixed
candidates use identical raw coordinates in both paths. The public constraint
contract remains `CandidateConstraints`; no DE-specific constraint DSL is
introduced.

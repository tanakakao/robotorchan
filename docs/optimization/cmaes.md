# CMA-ES optimizer backend

CMA-ES is an optional derivative-free acquisition optimizer. It consumes a
BoTorch `AcquisitionFunction`, tensor bounds, and a joint candidate batch
size `q`; the optional `cmaes` package implements the evolutionary loop.

The joint `q x d` candidate batch is flattened to one CMA-ES vector and
restored before each acquisition evaluation. The initial mean is the box
center. `sigma` is relative to the largest box width.

Install the optional dependency with `pip install "robotorchan[cmaes]"`.
The package is imported only when this backend is used.

## Candidate constraints

The backend accepts the shared `CandidateConstraints` contract via
`constraints`. It evaluates candidate violation and minimizes a
penalized objective:

```text
penalized_objective = -acquisition_value + constraint_penalty * violation
```

`constraint_penalty` must be positive (default `1e6`).
`equality_tolerance` defaults to `1e-6`. These settings apply to
the joint q-batch constraint evaluation, including inter-point
constraints. A finite penalty does **not** guarantee feasibility.
Validate returned candidates against the original constraints when
strict feasibility matters.

The evolutionary loop is CPU-side; acquisition and constraint
evaluation use the supplied tensor device and dtype. The returned
`value` is the **unpenalized acquisition value** at the selected
candidate, not the penalized search objective.

## Usage

```python
from robotorchan.optim.backends import optimize_acqf_cmaes

candidates, value = optimize_acqf_cmaes(
    acq_function=acq,
    bounds=bounds,
    q=3,
    sigma=0.25,
    max_generations=200,
    seed=123,
    constraints=candidate_constraints,
    constraint_penalty=1e6,
    equality_tolerance=1e-6,
)
```

The example assumes that `acq`, `bounds`, and
`candidate_constraints` have been constructed by the caller.
For differences across backends, see
[Candidate constraints](candidate-constraints.md).

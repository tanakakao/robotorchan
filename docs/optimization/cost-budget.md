# Evaluation cost budget management

Use CostBudget from robotorchan.workflow to track actual evaluation cost
separately from predicted cost. This ledger is independent of the BoTorch
acquisition function: cost-aware acquisition ranks candidates, while the
ledger determines whether an evaluation may be submitted.

~~~python
from robotorchan.workflow import CostBudget

budget = CostBudget(total=100.0)
candidate_cost = 12.0  # Predicted cost from a cost model
if budget.can_afford(candidate_cost):
    budget.reserve("experiment-1", candidate_cost)
    # Submit the candidate to the evaluator.
    budget.settle("experiment-1", actual_cost=15.0)
assert budget.spent == 15.0
assert budget.remaining == 85.0
~~~

Reserve each submitted evaluation before launching it, and settle on
completion. Cancel pending reservations when an evaluation is not run.
A batch can be reserved one evaluation at a time, using unique IDs.
For initial observations, record their actual cost if they count toward
the total budget. The budget is measured in user-defined cost units,
not the number of evaluations.

Predicted costs are admission estimates, not hard spending guarantees:
an actual evaluation may cost more than predicted, and spending may
exceed the configured total. Further submissions are then blocked.
The ledger is in-memory and does not coordinate concurrent processes.
External experiment schedulers must serialize ledger updates or provide
a transactional store. Persisting and restoring pending reservations
is not provided in this phase.

See multifidelity-cost-aware.md for the existing BoTorch MF-KG and
InverseCostWeightedUtility integration. Connecting this ledger to a
complete MF-KG experiment loop is covered in Phase 4.

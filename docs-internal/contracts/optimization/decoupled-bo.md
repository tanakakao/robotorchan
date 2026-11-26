# Decoupled BO contract

Baseline: 2b00b7b433a333461fe55042b8ba3189bdfeab3a.
Supported dependency range: BoTorch >=0.18.1,<0.19.

Phase 2 uses native ModelListGP with one independent model per output.
Partial observation utilities preserve output order, dtype, device and optional
known observation variance. Explicit boolean masks distinguish unobserved
entries from invalid observed values. Empty output datasets are returned but
must not be fitted without observations. Model construction and fitting remain
caller-owned. Mixed and fidelity columns are preserved unchanged.

Future optimizer contract: candidates (q,d), selected masks (q,m), allowed
masks (k,m). Pending masks are separate. One-shot auxiliary rows are not
physical evaluations. Ordinary fantasize support does not imply masked fantasy
support. HeterogeneousModel semantic outputs are not categorical latent logits.
Native HVKG compatibility is to be established by runtime tests; no private API
or new surrogate class is required for the independent GP baseline.

## Phase 3 native acquisition discovery

The acquisition factory resolves native HVKG and MFHVKG constructors by exact
name. Constructor discovery is not a workflow compatibility guarantee. The
initial numerical contract tests HVKG with independent partially observed GPs,
boolean evaluation masks, augmented one-shot inputs and finite gradients.
MFHVKG numerical validation, constrained value of information and correlated
output models remain separate gates. Do not mask EI or UCB as a substitute for
information value. Registry entries now record native one-shot and fantasy requirements;
workflow selection remains conditional on model and optimizer requirements.

### Phase 3 scope boundary

MFHVKG has a forward and gradient test using native fidelity projection and
partial observations. Cost-aware incremental value is a Phase 5 gate.
Single-objective masked KG and constrained or entropy-based information value
are not provided by the native HVKG constructor contract; they are not claimed
as implemented. Workflow selection is conditional, not a recommendation of
arbitrary model and optimizer combinations. Classification information value
remains a Phase 8 topic.


## Phase 4: candidate and mask optimization

`optimize_acqf_decoupled` enumerates allowed boolean masks of shape (k, m)
for q=1 and delegates each continuous optimization to the existing BoTorch
dispatcher. It returns (candidate, selected_mask, value), with shapes (1, d)
and (1, m) for the first two items. The original mask is restored in a finally
block. Ties retain the first mask. Nonfinite results fail explicitly.
One-shot auxiliary rows are handled by the underlying BoTorch optimizer.
Mixed domains, joint q>1, candidate constraints and pending evaluations are
separate integration gates. Raw acquisition values are compared without cost
normalization.


## Phase 5: fixed evaluation costs

The optimizer accepts either additive positive output_costs (m,) or explicit
positive mask_costs (k,), including nonadditive joint measurement costs.
For fixed positive costs, maximizing gain per cost for each mask leaves the
within-mask maximizer unchanged. Negative gains are multiplied by cost
rather than divided, following the sign-aware inverse-cost convention.
The third return value is the adjusted ranking score when costs are supplied.
CostBudget filters unaffordable masks using remaining funds including
reservations; candidate proposal does not mutate the ledger. No affordable
mask raises ValueError. Native cost-aware utilities must not be combined with
external cost scaling. Input-dependent and learned costs require a separate
within-mask optimization integration.


### Input-dependent predicted costs

For ordinary acquisitions, cost_function(X, mask) returns one strictly positive
finite differentiable cost per t-batch, matching the acquisition value shape.
The cost-adjusted objective is optimized directly, rather than adjusting only
the final candidate. Fidelity columns can be used as ordinary inputs to the
cost function. A learned model's posterior mean can be used as a deterministic
predicted cost if positive and differentiable; this is not expected inverse cost
under a random cost posterior. One-shot acquisitions require their native
cost-aware utility. Input-dependent budgets require feasible initial conditions.


### Native one-shot cost utility

HVKG and MFHVKG use the native InverseCostWeightedUtility with a positive
multi-output cost model and a current hypervolume baseline. The native
acquisition handles evaluation masks and applies cost to actual candidates,
not fantasy decision rows. The decoupled optimizer compares the resulting
native scores without additional scaling. The integration test uses a
finite target-set hypervolume baseline as an approximation, not an exact
continuous Pareto optimum.


### Predicted-cost budget constraint

For ordinary acquisitions with input-dependent costs, supply explicit
batch_initial_conditions. Each mask filters those starts by remaining budget
and delegates a differentiable nonlinear inequality to the existing BoTorch
backend. Masks without feasible supplied starts are skipped; this does not
prove global infeasibility. Returned candidates are checked against the
ledger again, without spending or reserving funds. This constrains predicted
cost, not uncertain realized cost. Native one-shot budgets remain unsupported.

If a local optimizer returns a point outside the strict predicted budget,
the best affordable supplied restart is retained instead of accepting an
overspend or discarding the entire mask.


### Phase 5 scope boundary

The continuous q=1 path supports fixed additive and nonadditive costs,
differentiable predicted costs, and predicted-cost budgets for ordinary
acquisitions with supplied feasible starts. Native HVKG and MFHVKG support
their own cost utility; hard input-dependent budgets for native one-shot
acquisitions are outside this phase's supported contract. Mixed and batch
candidate optimization are subsequent extensions. The learned-cost EPIG
integration fits a log-cost GP before candidate generation.


## Phase 6: joint batch enumeration

The continuous optimizer enumerates ordered assignments of allowed mask rows
to q actual candidates, with an explicit max_mask_assignments cap (default
256). Fixed per-mask costs are summed across candidates; cross-candidate
discounts are not implied. Candidate constraints and fixed features are
delegated to the existing dispatch. Predicted hard budgets combined with
other constraints or q > 1 remain unsupported until feasible-start handling
is extended. Mixed and sequential strategies require separate validation.


### Mixed q=1

MixedVariableSpace supplies integer and categorical choices. Ordinary masked
acquisitions delegate to the existing mixed BoTorch dispatch; one-shot
acquisitions delegate to the existing row-specific mixed one-shot optimizer.
The discrete enumeration cap is separate from the mask cap. Mixed predicted hard budgets and custom joint or one-shot constraints,
fixed features, starts and options are not yet supported.

### Mixed joint batches

The existing row-specific one-shot enumeration also handles ordinary joint
acquisitions, using q rows without fantasy augmentation. Each row has its own
discrete assignment. One-shot acquisitions retain auxiliary rows and extract
only actual candidates. The discrete cap applies per evaluation-mask assignment.

### Greedy sequential batches

`strategy="sequential"` fixes previously selected candidate rows and masks,
then optimizes the full prefix acquisition over the next row. Fixed costs
rank total prefix utility against total prefix cost; the budget reserves
minimum possible cost for remaining rows without mutating the ledger.
This is a greedy approximation, not joint optimality or pending fantasies.
One-shot, custom starts, candidate constraints and input-dependent costs
are not supported in this mode yet.

### Bounded heuristic and named backends

`strategy="heuristic"` evaluates at most `max_mask_assignments` distinct
joint masks, including the cheapest homogeneous mask when costs are known,
other homogeneous masks, then seeded random assignments. It is exhaustive
when the full set fits the cap. It does not approximate the acquisition itself.
Ordinary acquisitions delegate named optimizer selection and backend options
to existing dispatch, including mixed-capable backends. One-shot and predicted
hard budgets require BoTorch. Backend capability restrictions remain in force.

### Joint mask constraints and predicted batch budgets

`mask_constraint(mask)` filters complete boolean `[q, m]` assignments before
optimization in joint and heuristic modes. Sequential rejects arbitrary
full-batch mask predicates because prefix feasibility cannot be inferred.
Ordinary continuous BoTorch acquisitions support predicted hard batch budgets
with feasible explicit starts and an inter-point nonlinear cost constraint.
Other candidate constraints and fixed features cannot yet be combined with
this predicted budget path. One-shot candidate constraints are rejected until
actual-row mapping is available. Heuristic failure does not prove infeasibility.

### Phase 6 capability boundary

Factory selection remains conditional on masked acquisition semantics, backend,
strategy, discrete assignment limits, and runtime constraints. Ordinary joint
and heuristic acquisitions delegate candidate constraints to existing backends.
Sequential supports fixed features and fixed costs, but not arbitrary joint
predicates, one-shot acquisitions or input-dependent costs. Native one-shot
candidate constraints need actual-row mapping and remain unsupported.
Mixed BoTorch joint enumeration does not accept custom constraints or starts;
other mixed backends follow their existing capability contracts. Predicted hard
budgets are continuous BoTorch only and cannot combine other candidate constraints.

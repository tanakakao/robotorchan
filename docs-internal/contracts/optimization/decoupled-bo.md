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

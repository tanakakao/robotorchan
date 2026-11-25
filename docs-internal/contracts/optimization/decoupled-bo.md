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
information value. Registry recommendation is deferred until model and optimizer
requirements can be represented without overstating support.

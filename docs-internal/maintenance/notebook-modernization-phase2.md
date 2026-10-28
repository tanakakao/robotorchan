# Notebook modernization: Phase 2 evidence and Phase 3 worklist

This is a development inventory, not user-facing Guide or Theory content.

## Evidence policy

- A constructor or function call in a code cell is **static evidence only**.
- A text mention, including a Markdown explanation, is not executable evidence.
- A notebook listed in the default CI manifest has execution coverage, but this
  does not prove that every model variant or candidate path in it was exercised.
- The audit CSV marks execution as `unverified` until runtime evidence is linked.
- Do not infer `model.fit()` support from `fit_gpytorch_mll()` usage.
- Do not add fit methods to models as part of notebook modernization.

## Phase 3: inspected notebooks

The following findings come from inspecting the code cells in the Phase 3
notebooks on the Phase 2 branch. They are not independent runtime validations.

| Notebook | Current training | Posterior | Candidate optimizer | Action |
| --- | --- | --- | --- | --- |
| 01_single_task_gp | `fit_gpytorch_mll` | Present | `optimize_acqf` | Check `supports_fit`; prefer supported `model.fit()`. |
| 02_mixed_single_task_gp | `fit_gpytorch_mll` | Present | `optimize_acqf_mixed` | Check `supports_fit`; preserve categorical constraints. |
| 03_multi_fidelity_gp | `fit_gpytorch_mll` | Present | `optimize_acqf` and robotorchan optimizer | Check fit contract; verify fidelity projection and candidate feasibility. |
| 04_multitask_gp | `fit_gpytorch_mll` | Present | `optimize_acqf` | Check each multitask variant's fit contract and output-task handling. |
| 05_model_list_gp | `fit_gpytorch_mll` | Present | **Absent** | Add multiobjective acquisition, optimize, evaluate proposed candidate. |

The `05_model_list_gp.ipynb` gap is the first confirmed Phase 3 candidate-flow
omission. Its existing two objectives are a natural multiobjective example.
Keep the two independent child models and add an acquisition compatible with
`ModelListGP`, then a bounded candidate search and objective evaluation.
Choose the acquisition based on supported BoTorch APIs, and verify it by
executing the notebook rather than relying on token detection.

## Phase 2 completion requirements

1. Generate the public-model inventory and notebook-pipeline CSVs.
2. Cross-reference model `supports_fit` and actual fit methods; mark
   unsupported and unverified cases separately.
3. Record missing acquisition/optimizer/candidate paths for all notebooks,
   with the Phase 3–13 worklist derived from evidence.
4. Distinguish default, slow, and excluded notebook CI manifest groups.
5. Validate the scripts and the inventory; do not equate static detection with
   successful candidate generation.

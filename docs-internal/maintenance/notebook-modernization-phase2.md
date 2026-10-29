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

## Additional code-cell inspection (06–20)

The following notebooks have no syntactic `optimize_acqf*` call and no
`candidate` token in inspected code cells: 06, 07, 08, 09, 11, 12, 13,
14, 15, 16, 17, 18, and 20. Notebook 10 and 19 have both signals.
This is a **static triage result**: custom candidate APIs and runtime
behavior require separate inspection. Each missing candidate flow is a
modernization target in its assigned phase, not an invitation to add new
model-level fit methods.

The previous CI commit passed both CI and Documentation links. Notebook
execution itself remains unverified by this inspection.

## Remaining notebook inspection (21–41 and heterogeneous benchmark)

All 44 notebook files listed in `examples/run_notebooks.py` have now been
inspected for code-cell training, posterior, optimizer and candidate tokens.
This is a **static inspection**, not notebook execution.

Among the remaining 24 files (21–41 plus the heterogeneous benchmark):

- Both `optimize_acqf*` calls and candidate tokens appear in 26, 27,
  28_active_learning, 28_regression_active_learning, 29, 30, 31, 32, 33,
  35, 36_end_to_end, 36_sequential, 37, 38, and 39.
- Neither signal appears in 21, 22, 24, 25, 40, 41, and
  heterogeneous_bo_benchmark_phase19.
- 23 and 34 mention candidates without a matching `optimize_acqf*` call.
  These require manual review for custom or strategy-based optimization.
- The absence of a literal `.posterior()` call in a workflow notebook
  does not mean it cannot propose a candidate.

### Phase handoff priorities

1. Phase 3: implement candidate optimization for 05; review `fit()`
   compatibility in 01–04.
2. Phase 4–7: prioritize missing candidate flows in 06–09, 11–18,
   20–22; preserve model-specific training and original-space mapping.
3. Phase 8–10: add candidate flows in 24, 25, 40, 41; review 39's
   classifier probability and acquisition semantics.
4. Phase 11–13: preserve existing optimizer examples, inspect the
   strategy-specific candidate path in 34, and expand heterogeneous
   constrained BO beyond its benchmark-only demonstration.

No model-level `fit()` implementation is authorized by this plan.

## Verified fit contract for Phase 3 models

Source inspection of `models/base.py` and the standard model wrappers:

| Model | Wrapper base | Existing `model.fit()` contract |
| --- | --- | --- |
| `SingleTaskGP` | `FittableExactGPModelMixin` | Supported (`supports_fit=True`) |
| `MixedSingleTaskGP` | `FittableExactGPModelMixin` | Supported (`supports_fit=True`) |
| `SingleTaskMultiFidelityGP` | `ExactGPModelMixin` | Not advertised by wrapper |
| `MultiTaskGP` | `ExactGPModelMixin` | Not advertised by wrapper |
| `KroneckerMultiTaskGP` | `ExactGPModelMixin` | Not advertised by wrapper |
| `ModelListGP` | `ModelTrainingMixin` | Not advertised by wrapper |

The first two notebooks can use their existing wrapper `fit()` method.
For the remaining models, retain supported `make_mll()` +
`fit_gpytorch_mll()` unless further runtime checks establish an
explicitly supported fit path. Do not implement new fit methods here.

**Phase 2 status:** all notebook code cells were inspected for static
pipeline evidence, and Phase 3 model fit contracts were checked in source.
The full public-export fit contract and notebook execution results are not
yet verified; Phase 2 is not certified complete.

## Phase 2 decision and bounded handoff

**Phase 2 static-audit deliverables are complete.** All 44 manifest notebooks
were inspected, pipeline evidence scripts and tests are in place, CI and
documentation checks passed at commit `e8a775c`, and Phase 3's six model
fit contracts were checked against source.

This is **not** a claim that all public model classes have runtime-verified
`fit()` support or that any notebook was executed during this audit.
Full public-model fit coverage remains `unverified` in the model inventory;
do not replace it with inferred support. The notebook execution column also
remains `unverified`. These are tracked verification gaps, not blockers
for beginning model-by-model notebook modernization.

**Phase 3 acceptance criteria:**

1. Update 01 and 02 to use the already-supported `model.fit()` path,
   including their sequential BO loops.
2. Keep `fit_gpytorch_mll(model.make_mll())` for 03, 04, and 05 unless
   a supported fit contract is established.
3. Add a compatible multiobjective acquisition, bounded optimization,
   candidate proposal, and objective evaluation to 05.
4. Preserve valid mixed categories, fidelity semantics, and multitask
   output handling in 02–04.
5. Execute all five updated notebooks and record concrete CI or local
   execution evidence; do not treat static token matches as success.

# Empirical execution fairness and oracle-isolation contract

This contract applies to publishable measured optimization comparisons.
The existing runner supports the necessary switches, but defaults do
**not** satisfy this contract automatically.

## Required runner configuration

For each strategy and paired seed, call `run_benchmark` with:

- `restrict_candidate_problem=True` so callbacks receive
  `CandidateProblemView`, never the truth-bearing `BenchmarkProblem`.
- `independent_rng_streams=True` so initial Sobol designs and candidate
  generation have separate deterministic streams.
- `isolate_strategy_rng=True` to isolate PyTorch global RNG by seed and
  batch index; this does not isolate NumPy, Python random, or all GPU
  nondeterminism.
- An explicit observation-noise policy for noisy problems. Prefer
  `SeededObservationNoise(policy="paired")` for paired initial
  observations; record the policy and noise implementation.
- A validated `ComparativeExperimentCell` with matching problem,
  seeds, initial count, budget, q, float64 and device.

Do not pass hidden truth evaluators, known optima, reference fronts,
or future outcomes to candidate generation. `CandidateProblemView`
contains only name, bounds, directions, variable types and constraint
count. The bounds are copied at view construction.

## Paired comparison assertions

1. Run each strategy against the same registered problem and seed set.
2. Verify `validate_initial_pairing` across strategies. Require
   observation pairing when a paired-noise policy is declared.
3. Verify each trajectory has exactly `initial_points +
   evaluation_budget` evaluated points.
4. Verify `completed_batches[-1] == evaluation_budget` and, for
   empirical_standard q=3, a final increment of two points.
5. Candidate callbacks receive observed Y, not Y_truth. Truth,
   constraints and costs are computed by the runner for metrics only.
6. Keep q=1 and q=3 results in separate comparison strata.
7. Record exceptions, failed seeds and omitted cells; never drop
   failures from a paired comparison silently.

## Noise and reproducibility limitations

Paired noise shares a seed and a sequential generator; equal initial
batch shapes yield paired initial noise, but diverging candidate batch
schedules or consumption patterns can break pointwise noise pairing.
The runner does not automatically enforce deterministic CUDA kernels,
NumPy or Python random seeds, or a complete restart checkpoint.
The initial-design and candidate streams use
`derive_stream_seed` and the named-stream hash contract.

## Required experiment manifest

For each cell record the following fields before claiming a result:

| Group | Fields |
| --- | --- |
| Identity | problem, strategy, tier, q, seed, run ID |
| Source | git commit SHA, dirty state, environment provenance |
| Design | initial count, additional budget, paired design policy |
| RNG | stream policy, strategy RNG isolation, noise policy |
| Model | model class, training configuration, fit settings |
| Acquisition | class, MC sampler, samples, objective/constraint policy |
| Optimizer | backend, restarts, raw samples, constraints, failure policy |
| Runtime | dtype, device, hardware, timing boundaries |
| Artifacts | raw trajectory, truth metrics, curves, CSV/JSON, errors |
| Certification | config-valid, oracle-isolated, runtime-validated, reviewed |

The `BenchmarkExperimentConfig` serialization alone does **not**
contain all these fields. Verify the persisted execution manifest and
associated trajectory artifacts before claiming complete provenance;
do not infer that a later reporting stage automatically captured missing
metadata. See the [Frozen empirical protocol](../../../docs/benchmarks/empirical-evaluation-protocol.md)
for the required pre-execution manifest and acceptance gates.

## Phase 23 acceptance and Phase 24 handoff

Phase 23 freezes the contract and audits existing primitives:
`runner.py`, `candidate_view.py`, `random_streams.py`,
`strategy_rng.py`, `observation_noise.py`, and
`initial_pairing.py`. Static inspection alone does not certify any
benchmark result. Phase 24 must execute smoke tests under these
settings, assert oracle isolation and budget correctness, and retain
failure evidence before promoting any experiment cell.

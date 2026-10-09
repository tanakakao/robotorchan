# Benchmark result storage

Phase 26 adds versioned, non-executable JSON storage for full benchmark
trajectories, including observations, ground truth, constraint values,
evaluation costs, candidate locations, and experiment configuration.

```python
from robotorchan.benchmarks import load_benchmark_results, save_benchmark_results

save_benchmark_results("results/run.json", config, trajectories)
restored_config, restored_runs = load_benchmark_results("results/run.json")
```

The JSON schema contains `schema_version=1`, a serialized
`BenchmarkExperimentConfig`, and seed-ordered trajectory records.
Each tensor records its dtype, shape and nested values. Both float32
and float64 are supported. Loaded tensors are on CPU; the configured
device remains in the configuration metadata.

Validation rejects mismatched seeds, evaluation counts, non-finite
values, unsupported dtypes, malformed tensor shapes and unknown schema
versions. Unlike pickle, JSON loading does not execute Python objects.

This API is intended for trusted benchmark result files of manageable
size. It is not an atomic/concurrent writer, does not store model or
optimizer state, and does not yet support resuming an interrupted run.
Phase 27 addresses resumable execution.

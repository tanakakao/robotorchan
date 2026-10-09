# Resumable benchmark execution

Phase 27 introduces single-seed checkpointing and deterministic continuation.

```python
from robotorchan.benchmarks import (
    load_benchmark_checkpoint,
    run_resumable_benchmark,
    save_benchmark_checkpoint,
)

partial, checkpoint = run_resumable_benchmark(
    config, random_candidates, registry=registry, max_batches=2
)
save_benchmark_checkpoint("checkpoint.json", checkpoint)

restored = load_benchmark_checkpoint("checkpoint.json")
finished, _ = run_resumable_benchmark(
    config, random_candidates, registry=registry, checkpoint=restored
)
```

The checkpoint stores the full trajectory prefix, experiment configuration,
and the run-local PyTorch generator state. The schema is versioned JSON;
loading does not execute Python objects. Only one configured seed is
supported per checkpoint. A completed checkpoint is safe to pass again
and produces no new evaluations.

**Determinism:** Exact continuation is supported when the candidate
generator is a pure function of the supplied history and run-local RNG.
Generators with internal state, global random generators, optimizer
state, pending asynchronous evaluations, or nondeterministic evaluation
noise cannot be resumed exactly by this interface. The checkpoint
is written after a completed batch, not mid-batch. The caller must
explicitly persist the returned checkpoint; there is no automatic
periodic saving or atomic writer.

Phase 26 result files and Phase 27 checkpoint files have distinct schemas.

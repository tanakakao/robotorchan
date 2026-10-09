# Benchmark result persistence (Phase 13)

Benchmark trajectories can be exported to and loaded from versioned JSON
without Python pickle execution:

```python
from robotorchan.benchmarks import load_trajectory, save_trajectory

save_trajectory(trajectory, "run-42.json")
restored = load_trajectory("run-42.json")
```

The format preserves seed, initial design count, input points, observed
objectives, ground-truth objectives, constraint residuals, evaluation costs,
and tensor dtype (`float32` or `float64`). Tensors are loaded on CPU.
A `schema_version` field supports explicit compatibility checks.

Malformed shapes, unsupported dtypes, missing tensor fields, nonfinite
values and incompatible versions are rejected. The JSON is intended for
**reproducible result inspection**, not as a model checkpoint. To reproduce
the full experiment, retain the problem definition, algorithm configuration,
library versions and seed alongside the trajectory; these metadata are not
yet embedded in this minimal schema.

JSON decimal encoding does not guarantee bit-for-bit preservation of all
floating-point states or RNG states. No legacy aliases or pickle fallback
are provided.

# Transfer Bayesian Optimization benchmark

Run from the repository root after installing the project:

```bash
python examples/benchmarks/transfer_bo.py --steps 2 --seed 0
```

The script reports the **best observed target objective** at initialization
(iteration 0) and after each new evaluation, in CSV format. Higher is better.

| Method | Definition |
| --- | --- |
| `target` | Target-only EI, using TAF with source weight zero |
| `taf` | Fixed 0.5 source / 0.5 target acquisition weighting |
| `rgpe` | Ranking-weighted source/target GP ensemble with EI |

Two scenarios use the same target objective (optimum at x=0.75):
an **aligned source** (optimum x=0.75) and a **misaligned source**
(optimum x=0.05). The latter deliberately probes negative transfer.

The source GP is constructed from six observations without fitting its
hyperparameters; this keeps the comparison inexpensive and reproducible
but limits its external validity. All methods start from identical target
observations and share optimizer settings. The methods do not necessarily
propose the same candidates, so the observations diverge after iteration 0.

This is a lightweight functional comparison, **not** a claim that any
method is superior. For statistical conclusions, run multiple seeds and
report distributions of best-observed values and target evaluation counts.
RGPE also incurs LOO fitting cost, which is not measured by this script.

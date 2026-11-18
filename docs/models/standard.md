# Standard / Mixed / Multi-Fidelity / Variational

## Single-task baseline

`SingleTaskGP` は連続入力の標準的な exact GP baseline です。特殊なデータ構造がない場合、まずこのモデルとの比較を残すと、複雑な surrogate を導入する効果を判断しやすくなります。

`MixedSingleTaskGP` は連続変数とカテゴリ変数が共存する入力空間向けです。カテゴリコードを連続量として距離計算する代替ではなく、categorical covariance を使うことが重要です。

Notebook: [SingleTaskGP](../../examples/notebooks/01_single_task_gp.ipynb)、[MixedSingleTaskGP](../../examples/notebooks/02_mixed_single_task_gp.ipynb)  
Theory: [Gaussian Process](../theory/02_gaussian_process.md)、[Mixed Variables](../theory/05_mixed_variables.md)

## Multi-Fidelity

`SingleTaskMultiFidelityGP` と `MixedSingleTaskMultiFidelityGP` は、評価コストや精度の異なる fidelity 間で情報共有します。シミュレーション解像度、簡易試験と本試験など、fidelity に順序・意味があり相関が期待できる場合に使います。

Mixed版では「カテゴリ入力」と「fidelity」を別の構造として扱います。

Notebook: [Multi-Fidelity](../../examples/notebooks/03_multi_fidelity_gp.ipynb)  
Theory: [Multi-Fidelity](../theory/06_multi_fidelity.md)

## Variational

`SingleTaskVariationalGP` と `MixedSingleTaskVariationalGP` は inducing points を使う近似 GP です。Exact GP の計算量がボトルネックになるデータ量で候補になります。

Exact GP と同じ fitting contract ではありません。`make_mll()` は Variational ELBO 系の学習を前提とするため、データ件数と minibatch の扱いを確認してください。

Notebook: [Variational GP](../../examples/notebooks/06_variational_gp.ipynb)  
Theory: [Variational GP](../theory/09_variational_gp.md)

## 選択基準

カテゴリがあるだけなら Mixed、fidelity 構造があるなら Multi-Fidelity、データ量による exact inference の計算負荷が主問題なら Variational を検討します。これらは排他的なラベルではなく、実装が対応する範囲では複数の問題軸を同時に扱えます。

## Known-DAG / Intervention GP (Causal BO Phase 3)

`InterventionGPSurrogate` is a thin adapter over the existing
`SingleTaskGP` for an **explicitly performed hard-intervention set**. It does
not discover a causal graph or identify effects from observational data.
Observational records and intervention records for other target sets are
excluded from GP training; each selected intervention set has its own GP.

```python
import torch
from robotorchan.causal import (
    CausalGraph,
    CausalObservation,
    HardIntervention,
    InterventionGPSurrogate,
)

graph = CausalGraph({"X": (), "Y": ("X",)})
records = [
    CausalObservation({"X": x, "Y": x * x}, HardIntervention({"X": x}))
    for x in (0.0, 0.3, 0.6, 1.0)
]
surrogate = InterventionGPSurrogate(graph, "Y", ("X",), records)
surrogate.fit()
posterior = surrogate.posterior(torch.tensor([[0.4]], dtype=torch.double))
print(posterior.mean, posterior.variance)
```

The feature order of posterior inputs follows the DAG topological order
restricted to the intervention nodes; query shape is `[..., q, d]`. The
`surrogate.model` attribute exposes the existing native BoTorch-compatible
`SingleTaskGP` for future acquisition integration.

At least two records with **exactly the specified intervention variables**
are needed. The GP uses the existing inferred-noise defaults. Its predicted
response is interpretable as an empirical response under the performed
intervention regime only if background experimental conditions are comparable.
In particular, `E[Y | X=x]` from observational data is not automatically
`E[Y | do(X=x)]`. Cross-set information sharing, candidate generation,
intervention cost, and a complete CBO loop are not implemented in Phase 3.

## Causal BO Phase 5: Sequential Ask-and-Tell

`CausalBOStudy` is a dedicated in-memory intervention workflow, separate
from the ordinary `Study` parameter search space. It reuses the shared
`TrialState` lifecycle and the Phase 3 and Phase 4 GP/LogEI implementation.
It does **not** learn a causal graph or infer interventional effects from
observational-only data.

```python
from robotorchan.causal import CausalBOStudy, CausalGraph
from robotorchan.study import TrialState

graph = CausalGraph({"X": (), "Z": (), "Y": ("X", "Z")})
study = CausalBOStudy(
    graph, "Y", (("X",), ("Z",)),
    {"X": (0.0, 1.0), "Z": (0.0, 1.0)},
    n_initial_trials=2,
    optimizer="sobol",
    seed=3,
)
for _ in range(6):
    trial = study.ask()
    # Replace this with the actual hard-intervention experiment:
    y = sum(value * value for value in trial.intervention.values.values())
    study.tell(trial, y)

print(len(study.observations), study.trials[-1].state)
```

**Cold start:** each configured intervention set receives
`n_initial_trials` completed Sobol interventions (at least 2 per set),
using the existing `sample_candidate_batches` helper. The next
`ask()` fits one existing `InterventionGPSurrogate` per set and delegates
the best intervention selection to `propose_causal_intervention`.
A successful `tell()` invalidates cached models so the next model-based
`ask()` refits with the latest interventional observations.

`tell(trial, outcome)` records a scalar outcome; an optional mapping of
measured graph nodes is also supported, with required intervention-value
consistency. `tell(trial, state=TrialState.FAIL)` records failure without
adding training data. `trials` includes failed and running experiments;
`observations` only includes completed hard interventions. Trials receive
monotonically increasing IDs; a second `ask()` while a trial is running,
or repeated `tell()`, is rejected.

This Phase 5 flow is sequential `q=1` only. It does not implement
parallel or persistent trials, mixed variables, categorical interventions,
causal discovery, observational-data adjustment, or stochastic causal graph
inference. A workflow cannot be used to claim that associations learned
from purely observational records are causal effects.

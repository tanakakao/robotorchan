# BoTorch GP の benchmark 戦略

`make_botorch_gp_strategy` は BoTorch `SingleTaskGP`、`fit_gpytorch_mll`、`qExpectedImprovement` または `qNoisyExpectedImprovement`、`optimize_acqf` を `run_benchmark` の候補 callback に接続します。

対象は制約なし・連続・単一目的です。観測値だけで学習し、最小化は内部で最大化へ変換します。候補選択時に真値や既知の最適値へアクセスしません。

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    BenchmarkProblemRegistry,
    make_botorch_gp_strategy,
    run_benchmark,
)
from robotorchan.benchmarks.standard_problems import register_standard_problems

registry = BenchmarkProblemRegistry()
register_standard_problems(registry)
config = BenchmarkExperimentConfig(
    problem="branin",
    strategy="botorch-qNEI",
    seeds=(0,),
    initial_points=8,
    evaluation_budget=4,
)
strategy = make_botorch_gp_strategy(acquisition="qNEI", num_restarts=2, raw_samples=32)
trajectory = run_benchmark(config, strategy, registry=registry)[0]
print(trajectory.X.shape)
```


初期 Sobol 設計は問題・seed・dtype・device のみに依存し、Random と GP の初期条件を揃えられます。これは動作確認の基準法で、性能の主張ではありません。公平な多 seed 比較やノイズ実験は別途プロトコルに従って行ってください。

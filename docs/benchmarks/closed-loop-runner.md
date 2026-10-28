# Closed-loop benchmark runner

問題契約、registry、実験設定を接続します。初期 Sobol 設計を評価した後、**追加評価**予算が尽きるまで候補を要求します。最後の batch は q より小さくなる場合があります。

```python
from robotorchan.benchmarks import (
    BenchmarkExperimentConfig,
    random_candidates,
    run_benchmark,
)

config = BenchmarkExperimentConfig(
    problem="quadratic",
    strategy="random",
    seeds=(0, 1, 2),
    initial_points=8,
    evaluation_budget=20,
    q=3,
)
# Register "quadratic" with register_problem(...) before running.
trajectories = run_benchmark(config, random_candidates)
```


callback は `(problem, X, Y_observed, q, generator) -> candidates` です。任意の BoTorch 獲得関数・optimizer を接続でき、runner の変更は不要です。`strategy` 文字列は意図を記録するだけで、対応 callback は呼び出し側が渡します。

`BenchmarkTrajectory` は X、観測 Y、決定的な真値 Y、制約残差、費用、初期設計の境界を保存します。`evaluation_count` は初期設計を除き、`cumulative_cost` は含みます。候補 callback には真値を渡さず観測値だけを渡します。

戦略間で問題・seed ごとの初期 Sobol 設計を揃えます。Random は seed ごとの torch generator を使い、global RNG を変更しません。[結果保存](storage.md)と[指標](metrics.md)は別モジュールです。

観測 callback の global 乱数は呼び出し側の責任です。ノイズ実験では明示的なノイズ方針と seed の意味を[公平性プロトコル](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)に従って記録してください。候補 callback に真値を含む problem フィールドを見せない場合は `restrict_candidate_problem=True` を使い、比較の認定前に制限 view の契約を確認します。

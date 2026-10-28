# 実行 profile

runner や device を暗黙に変更せず、再利用可能な実験規模を指定します。

| Profile | Seed 数 | 初期点数 | Seed ごとの新規評価数 | q |
| --- | ---: | ---: | ---: | ---: |
| smoke | 1 | 4 | 4 | 1 |
| standard | 3 | 8 | 40 | 1 |
| extended | 5 | 16 | 100 | 4 |

既定は CPU・float64 です。予算は初期設計を含まず、最後は q 未満でも構いません。

```python
from robotorchan.benchmarks import get_execution_profile, run_benchmark

profile = get_execution_profile("smoke")
config = profile.make_config(problem="branin", strategy="random")
# Supply a registry containing the named problem and a candidate callback.
trajectories = run_benchmark(config, random_candidates, registry=registry)
```


`get_execution_profile("standard", dtype="float32", device="cuda")` のように明示的に変更できます。CUDA の存在は要求時に検査しますが、問題・戦略・任意依存の CUDA・float32 対応は実行で別途検証します。

profile は資源予算であり性能基準ではありません。[実測プロトコル](empirical-evaluation-protocol.md)の大きな cohort とは別です。戦略間で profile と registry を揃え、設定を結果に保存してください。対応付けと評価軸は[公平性契約](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)を参照してください。

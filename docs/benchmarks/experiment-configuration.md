# 実験設定

`BenchmarkExperimentConfig` は GP・獲得関数・optimizer・保存実装から独立した、変更不能で JSON 互換のプロトコル記録です。

```python
from robotorchan.benchmarks import BenchmarkExperimentConfig

config = BenchmarkExperimentConfig(
    problem="quadratic",
    strategy="sobol",
    seeds=(0, 1, 2, 3, 4),
    initial_points=12,
    evaluation_budget=60,
    q=3,
    dtype="float64",
    device="cpu",
)
record = config.to_dict()
restored = BenchmarkExperimentConfig.from_dict(record)
assert restored == config
```


- `problem` は `resolve_problem(registry)` で解決します。
- `strategy` は識別子で、対応する処理は runner 側で接続します。
- `seeds` は重複しない非負整数で、順序を保持します。
- `initial_points` は最適化予算より前に評価します。
- `evaluation_budget` は BO 反復数でなく追加評価数です。
- `q` は要求 batch 数で、最後の部分 batch を許可します。
- `dtype` は float32・float64、`device` は cpu・cuda です。
- `to_dict`・`from_dict` は未知フィールドを拒否して JSON 往復に対応します。

設定を作るだけでは global RNG の変更、CUDA Tensor の確保、学習、実験実行は行いません。[runner](closed-loop-runner.md)が seed ごとの実行と初期設計の対応を担います。ノイズ方針と環境は別途記録してください。

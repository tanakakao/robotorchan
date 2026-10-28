# 実験結果の保存

観測、真値、制約、費用、候補位置、実験設定を version 付きの非実行 JSON として保存します。

```python
from robotorchan.benchmarks import load_benchmark_results, save_benchmark_results

save_benchmark_results("results/run.json", config, trajectories)
restored_config, restored_runs = load_benchmark_results("results/run.json")
```


schema は `schema_version=1`、直列化した `BenchmarkExperimentConfig`、seed 順の軌跡を含みます。Tensor は dtype・shape・入れ子の値を記録し、float32・float64 に対応します。読み込み先は CPU で、設定 metadata の device は維持します。

seed・評価数の不整合、非有限値、未対応 dtype、不正形状、未知 version を拒否します。pickle と異なり Python オブジェクトを実行しません。

信頼できる適切なサイズの結果が対象です。atomic・並行 writer ではなく、モデル・optimizer 状態も保存しません。完了軌跡の保存と中断からの継続は別で、後者は [checkpoint](checkpoint.md)・[全体手順](end_to_end.md)を使います。

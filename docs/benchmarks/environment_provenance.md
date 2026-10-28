# 実行環境と再現性

既存 runner を使い、各結果の隣に環境 manifest を保存します。Git の SHA、Python・package version、OS、CPU、thread 設定、device、精度、実験設定を記録します。取得できない SHA・version は `null` とし、固定済み環境の証拠とは扱いません。

```python
from pathlib import Path

from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.empirical_environment import save_empirical_environment

config = BenchmarkExperimentConfig(
    problem="branin",
    strategy="random",
    seeds=(0, 1),
    initial_points=10,
    evaluation_budget=10,
    q=1,
    dtype="float64",
    device="cpu",
)
save_empirical_environment(Path("benchmark_outputs") / "environment.json", config)
```


`verify_random_baseline_reproducibility` は CPU Random の反復軌跡と JSON 往復を確認します。基準法を検証済みとする前に、次の回帰・manifest テストを実行してください。

```bash
pytest -q tests/benchmarks/test_empirical_phase2_reproducibility.py tests/benchmarks/test_empirical_environment.py
ruff check src/robotorchan/benchmarks/empirical_environment.py tests/benchmarks/test_empirical_environment.py
ruff format --check src/robotorchan/benchmarks/empirical_environment.py tests/benchmarks/test_empirical_environment.py
```


## 限界

- manifest は PR 準備端末ではなく、実験を実行した環境で取得します。
- 数値再現性は実時間を含みません。
- GP 学習、獲得関数 MC、最適化、観測ノイズ、CUDA 決定性は別途検証が必要です。
- ノイズには `SeededObservationNoise(policy="paired")` などを明示し、方針・seed 対応を保存します。異なる適応候補位置間で点ごとの共通乱数になる保証はありません。[公平性契約](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)を確認してください。
- peak memory、全環境での確実な CPU 型番、解決済み依存 lockfile は記録しません。必要なら別途保全します。
- CI の smoke test 成功を多 seed 性能結果とみなしてはいけません。

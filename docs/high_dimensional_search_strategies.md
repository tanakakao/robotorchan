# 高次元ベイズ最適化の探索戦略

robotorchan では surrogate model と acquisition function と search strategy を分離する。
高次元モデルを変更せず、獲得関数の最適化方法だけを切り替えられる。

## 戦略

| Strategy | 探索空間 | 状態 | 主な用途 |
| --- | --- | --- | --- |
| `OriginalSpaceStrategy` | 元の d 次元 | なし | 標準 `optimize_acqf`、低〜中次元の基準 |
| `RandomSearchStrategy` | 元の d 次元 | なし | 高次元での単純・堅牢なベースライン |
| `LatentSpaceStrategy` | PCA / RandomProjection 潜在空間 | reducer に依存 | 既知の低次元表現を使う探索 |
| `REMBOStrategy` | 固定ランダム埋め込み | 固定 | intrinsic dimension が低いと期待できる場合 |
| `TuRBOStrategy` | 元空間の局所 trust region | あり | 高次元で局所探索を適応的に集中 |
| `BAxUSStrategy` | 適応的ランダム部分空間 | あり | intrinsic dimension が不明で、探索中に部分空間を拡張したい場合 |

## BAxUS の使い方

```python
strategy = BAxUSStrategy(bounds, initial_target_dim=2, new_dimensions=3, seed=0)
result = strategy.optimize(acq_function)
y_new = objective(result.candidates)
state = strategy.update_state(y_new)

if state.restart_triggered:
    strategy.expand_subspace()
```

`BAxUSStrategy` は目的関数を評価しない。目的関数評価後に `update_state()` を明示的に呼ぶ。
trust region が最小長を下回ると `restart_triggered=True` となり、`expand_subspace()` で target dimension を増やす。
元の埋め込み方向は保持し、新しい方向だけを追加する。

## 設計上の注意

- search strategy は surrogate を置き換えない。
- acquisition function には最終的に元入力空間の候補を渡す。
- `SearchResult` は候補点・獲得関数値・探索診断だけを返し、wall-clock 時間は保持しない。
- 探索時間は benchmark / 呼び出し側で `strategy.optimize(...)` の前後を計測する。
- `LatentSpaceStrategy` はデータから学習した reducer、REMBO / BAxUS は探索用の埋め込みであり役割が異なる。
- TuRBO / BAxUS は stateful なので、逐次 BO ループ側が観測値を state に戻す必要がある。
- BAxUS の target-space は `[-1, 1]^k` を基準とし、元の box bounds へアフィン変換するため、入力変数の物理スケールに依存しにくい。

## ベンチマーク

`benchmarks/high_dimensional_acqf_optimization.py` は連続獲得関数最適化そのものを比較する。
`benchmarks/high_dimensional_sequential_bo.py` は逐次 BO の simple regret と探索時間を比較する。
モデル性能と探索戦略性能を混同しないため、探索戦略比較では同じ surrogate / acquisition を使うことを原則とする。
探索時間はライブラリの戻り値ではなく、benchmark 側で外部計測する。

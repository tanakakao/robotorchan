# 実測曲線・時間 artifact

完全性を検証済みの実測報告から、再現可能な CSV を出力します。未実行の benchmark 結果を生成する機能ではありません。

```python
from robotorchan.benchmarks.empirical_curves import export_empirical_curves

export_empirical_curves("results/branin_q1", "results/branin_q1_artifacts")
```


- `curves.csv`：初期 20 点を含む各評価後の真値 simple regret または固定 reference hypervolume。`total_points` は実履歴長、`evaluations` は追加評価数で初期設計中は 0 です。
- `timing.csv`：seed ごとの候補生成、評価、初期評価、総経過秒数。
- `summary.json`：対応 seed の要約と bootstrap 区間。

完全に検証した 10 seed cohort だけを出力でき、問題・q・ノイズ条件ごとに分離します。CSV は描画用に使えますが、この層は描画、費用の大きな実験、性能の主張を行いません。

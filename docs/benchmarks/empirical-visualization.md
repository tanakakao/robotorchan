# 実測根拠に基づく可視化

完全な既存 artifact ディレクトリから描画します。

```python
from robotorchan.benchmarks.empirical_visualization import render_empirical_charts

render_empirical_charts("results/branin_q1_artifacts", "results/branin_q1_charts")
```


任意依存の `matplotlib` が必要で、`performance.svg`、`timing.svg`、`provenance.json` を生成します。性能図は追加予算 0–50 に対する 10 seed の真値 regret または hypervolume 平均、時間図は平均実時間を示します。

入力の欠落・不完全性と既存の出力先を拒否し、問題・q・ノイズ条件を混ぜません。図は記述的なもので統計的優越性を示しません。このモジュールは benchmark 実験を実行しません。

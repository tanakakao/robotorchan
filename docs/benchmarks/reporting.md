# 結果の報告

`make_metric_report(strategy, metric, curves, interval_method="normal")` は独立 run seed から有限な 1 次元 PyTorch 曲線への辞書を受け取ります。曲線の長さ・dtype・device を揃え、ソートした seed、点ごとの平均と信頼区間を返します。

既定の `normal` は正規近似、`bootstrap` は seed 付き percentile bootstrap です。

```python
report = make_metric_report(
    "random",
    "simple_regret",
    {0: regret_seed_0, 1: regret_seed_1},
    interval_method="bootstrap",
    n_resamples=2000,
    seed=0,
)
json_text = report.to_json()
csv_text = report.to_csv()
```


JSON・CSV の出力は決定的です。JSON は `schema_version=1`、strategy、metric、seed、mean、lower・upper、confidence、interval method を含みます。CSV の各行は 0 始まりの曲線 index で、追加 BO 評価数へ自動変換しません。

この API は実験や描画を実行せず、公平性も保証しません。比較前に[公平性契約](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)を検証し、差には[対応統計](multiseed_statistics.md)を使います。描画時は mean と lower・upper 間の帯を示してください。

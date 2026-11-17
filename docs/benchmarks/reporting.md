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

## Phase 4：実測4条件の統合レポート

Phase 3で保存した `report.json` を使い、最適化を再実行せずに結果表とシード別の散布図を生成できます。

```bash
python scripts/render_empirical_four_cell_report.py empirical-artifacts phase4-report.md --figure phase4-regret.svg
```

`empirical-artifacts/` の下に `branin-q1/`、`branin-q3/`、`hartmann6-q1/`、`hartmann6-q3/` を置き、各フォルダに元の `report.json` を配置してください。元の GitHub Actions artifact から展開します。未実測条件を自動補完せず、4条件のいずれかが欠落・不完全なら失敗します。

出力する Markdown は各手法の最終 simple regret 平均、実行時間平均、Random/Sobol との対応シード bootstrap 95%区間、実験SHAを記録します。オプションの SVG は **全10シードの最終 regret と平均** を表示し、シード間のばらつきを隠しません。SVG の生成のみ matplotlib が必要です。

ここでの実測はノイズなしの単目的合成問題に限定されます。q=1 と q=3 はモデル再学習回数が異なるため、実行時間の単純比較を並列効率の証拠とはしません。多目的・制約付き・分類 Active Learning の性能は未評価です。各セルの初期点・実験SHA・実行条件・元の軌跡を保持し、q 間の有意差を断定しないでください。

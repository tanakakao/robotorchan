# Benchmark Quickstart — 実行から結果の解釈まで

このページは、既存の benchmark API と保存済み実測結果の利用入口です。**smoke 実行と実測性能の認定は別**です。

## 1. 最初の実行（軽量 smoke）

リポジトリルートで次を実行します。

```bash
python examples/benchmark_end_to_end.py
```

Branin をローカル registry に登録し、Random の closed-loop 実行、軌跡の JSON 保存・再読込、simple regret の集計を行います。出力は `benchmark_outputs/branin_random_results.json`、`branin_random_report.json`、`branin_random_report.csv` です。これは **1 seed の動作例**であり、最適化性能の優劣は判断できません。

- 問題の追加・registry の登録方法は [Problem registry](problem-registry.md) と [Problem contract](../../docs-internal/contracts/benchmarks/problem-contract.md) を参照してください。
- 実行設定は [Experiment configuration](experiment-configuration.md)、予算・q・seed は [全体手順](end_to_end.md) を参照してください。
- 軌跡の保存・再読込は [Storage](storage.md)、途中再開は [Checkpoint](checkpoint.md) を参照してください。完了軌跡の再読込と途中再開は異なります。

## 2. 保存済みの実測4条件を可視化する

GitHub Actions の成果物から、以下の各フォルダに **元の `report.json`** を配置します。これらは10 seed、初期20点＋追加50評価、CPU/float64 のノイズなし単目的実験です。

```text
empirical-artifacts/
  branin-q1/report.json
  branin-q3/report.json
  hartmann6-q1/report.json
  hartmann6-q3/report.json
```

```bash
python scripts/render_empirical_four_cell_report.py empirical-artifacts phase4-report.md
python scripts/render_empirical_four_cell_report.py empirical-artifacts phase4-report-with-figure.md --figure phase4-regret.svg
```

SVG を出力する場合のみ `matplotlib` が必要です。スクリプトは既存の出力を上書きしません。別名で実行するか、不要な出力を明示的に整理してください。4条件が揃わない場合はレポートを生成できません。

出力される Markdown は最終 simple regret の平均、平均経過時間、Random/Sobol との対応 seed bootstrap 区間、実験 SHA を含みます。SVG は10 seed すべての最終 regret と平均を表示します。**低い simple regret が良い**ことを示しますが、q=1 と q=3 の壁時計時間はモデル更新回数が異なるため、並列計算効率の直接比較には使えません。

## 3. 新しい比較問題を追加する場合

1. [Problem contract](../../docs-internal/contracts/benchmarks/problem-contract.md) に従い、入力境界、目的方向、真値、制約・reference point の有無を定義します。
2. [Problem registry](problem-registry.md) の登録方法を使用します。import だけでは登録されません。
3. [比較行列](comparative-experiment-matrix.md) と [適格性・実測プロトコル](empirical-evaluation-protocol.md) を確認し、未対応の戦略を成功例として扱わないでください。
4. 同一問題・seed・初期点・評価予算・q・dtype/device で比較し、元の trajectory、manifest、エラー記録を保全します。
5. [Reporting](reporting.md) と [多 seed 統計](multiseed_statistics.md) に従って集計します。smoke で性能の優位性を宣言しません。

## 4. 既知の制限と再現性

- 上記4条件の実測は **Branin/Hartmann6 × q=1/3 × Random/Sobol/qEI/qNEI** のみです。多目的・制約付き・分類 Active Learning や Mixed の性能を代表しません。
- q=1 と q=3 は探索の更新回数が異なります。両者の seed 番号が一致しても、初期観測が一致することを別途検証しない限り確認的な対応比較にはできません。
- Branin q=1 はほかの3条件と実験 SHA が異なります。差分確認なしに同一実装と断定しないでください。
- Actions artifact は保存期限があります。再現性のため、manifest・report・trajectory・commit SHA と依存バージョンを別途保全してください。
- 実測の統計的な解釈は [評価契約](../../docs-internal/contracts/benchmarks/empirical-evaluation-protocol.md) に従います。

## 5. Notebookとの使い分け

[Notebook一覧](../../examples/README.md) はモデルの学習・posterior・獲得関数・候補提案を理解する入口です。Benchmark Suite の実験条件・複数seedの統計比較を再現する場合は、このページの Python 例と保存済み実測レポートを使用してください。Notebook の単発実行結果を、上記4条件の実測証拠と混同しないでください。

## 6. 関連資料

[全体手順](end_to_end.md) / [結果の報告](reporting.md) / [実測評価プロトコル](empirical-evaluation-protocol.md) / [保存](storage.md) / [Checkpoint](checkpoint.md)

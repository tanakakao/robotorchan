# Benchmark の全体手順

問題 registry、profile、closed-loop runner、真値指標、多 seed 報告、結果保存、checkpoint を接続します。

## 最初の実行

リポジトリルートで自己完結した例を実行します。

```bash
python examples/benchmark_end_to_end.py
```


`benchmark_outputs/branin_random_results.json`、`branin_random_report.json`、`branin_random_report.csv` を生成します。ローカル registry へ Branin を登録し、smoke profile で Random を実行、軌跡を保存・再読込し、真値 simple regret と報告を出力します。

## 実験の条件

- `evaluation_budget` は新規評価数のみで、初期設計は別です。
- q は要求 batch 数で、最後は小さくても構いません。
- registry に問題を明示登録します。import だけでは登録しません。
- 戦略には `Y_observed` のみを見せ、`Y_truth` は指標に使います。
- 実行可能性は `g(X) >= 0`。simple regret は既知の単一目的最適値、hypervolume は 2 目的と reference point が必要です。
- 報告曲線は初期設計を含みます。CSV の `evaluation_index=0` は最初の初期観測で、最初の追加候補ではありません。

## 戦略比較

問題、seed、初期設計、予算、q、dtype、device を揃え、全 seed で同じ指標を計算します。`{seed: curve}` を `make_metric_report` に渡し、seed 再抽出には `interval_method="bootstrap"` を使います。対応差は `compare_paired_curves`・`compare_paired_bootstrap` で求めます。smoke は 1 seed なので、不確実性分析には standard・extended を使ってください。

## 保存と再開

`save_benchmark_results`・`load_benchmark_results` は完了軌跡の JSON 保存用です。中断継続には `run_resumable_benchmark` と専用 checkpoint API を使います。1 呼び出し 1 seed で run-local generator 状態を保存します。厳密な継続には保存履歴と当該 generator のみに依存する戦略が必要で、外部 optimizer 状態、global RNG、確率的観測ノイズは復元しません。

## 再現性と限界

正確な設定・環境・package version を報告と併せて保存し、対応比較では初期設計と seed 集合を一致させます。profile は資源の規則で品質保証ではありません。CUDA profile は存在確認だけを行い、問題・戦略の対応は別です。異なるハードウェアの実時間差をアルゴリズム差だけで解釈しないでください。

## 関連資料

- [過去の release 判定](../../.github/project-records/benchmarks/release_readiness.md)
- [Profile](execution_profiles.md)
- [結果保存](storage.md)
- [Checkpoint](checkpoint.md)
- [多 seed 報告](reporting.md)

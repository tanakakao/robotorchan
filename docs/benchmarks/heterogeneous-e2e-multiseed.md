# 対応 seed による統計評価

`robotorchan.benchmarks.heterogeneous_baselines.run_strategy` を基に、`compare_strategies` で qEI・一様 Random・scrambled Sobol を比較します。seed 内では初期設計とノイズ付き初期観測を揃え、各戦略のノイズなし best-so-far Strength を `(n_seeds, steps + 1)` で返します。

`summarize_trajectories` は評価ごとの平均、不偏標本標準偏差（`ddof=1`）、標準誤差 `std / sqrt(n_seeds)` を計算します。smoke test は 2 seed・逐次 2 評価で、性能比較には十分な seed と事前指定予算が必要です。目的評価数は揃えますが計算 overhead は正規化しません。

標準誤差は seed 間の推定平均の不確実性であり、信頼区間やモデルの予測不確実性ではありません。テストは集約・再現性の契約を検証し、優越性や有意差は示しません。主張には seed 別軌跡と対応差を報告してください。

`pytest -q tests/benchmarks/test_heterogeneous_multiseed_e2e.py tests/benchmarks/test_heterogeneous_baseline_comparison_e2e.py` で実行します。

## 固定予算の基準法

単一 seed の確認では、3 戦略が同じ初期 12 点とノイズ観測から始まり、追加評価を各 3 回行います。qEI は毎候補前に再学習します。その Sobol 獲得関数 optimizer と、独立した Sobol 基準戦略は別です。

指標は初期設計を含む全評価点のノイズなし best-so-far Strength です。候補選択後に真値で計算し、qEI の選択には使いません。形状、bounds、有限性、単調性、予算一致、最終指標の再構成を検査し、qEI の勝利は要求しません。

`pytest -q tests/benchmarks/test_heterogeneous_baseline_comparison_e2e.py` で確認できます。2 seed・2 評価は検出力を備えた性能比較ではありません。[過去の報告記録](../../.github/project-records/benchmarks/heterogeneous-e2e-benchmark-report.md)も、生成器だけで実測結果を示すものではありません。

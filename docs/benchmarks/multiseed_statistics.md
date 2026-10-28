# 多 seed の統計評価

正規近似に加え、seed 単位の percentile bootstrap 信頼区間を提供します。

`bootstrap_curves({seed: metric_curve, ...})` は点ごとの平均と区間を返します。評価時点をばらして再抽出せず、独立 seed の軌跡全体を再抽出して軌跡内の依存を保ちます。同じ `seed`・`n_resamples` で同じ区間を再現します。

`compare_paired_bootstrap(first, second)` は対応 seed の first-minus-second 曲線を計算して差を再抽出します。`probability_positive` は bootstrap 平均差が正となった経験的割合で、Bayesian 事後確率や正式な p 値ではありません。

比較前に `FairComparisonProtocol` で設定と初期観測の一致を確認してください。[公平性契約](../../docs-internal/contracts/benchmarks/comparative-fairness-protocol.md)に従い、独立 seed 数、信頼水準、再抽出数、指標方向、予算を報告します。

- 区間は点ごとで、同時信頼帯ではありません。
- 独立 seed が少ない場合は不安定です。
- 1 seed の区間は点に縮退し、不確実性を推定できません。
- 多重比較補正は行いません。
- 欠損・非有限曲線は共通 validator が拒否します。

# 実測評価の実行手順

凍結した比較規則であり、実測結果を認定済みとする文書ではありません。適格な問題・戦略の組合せを選び、`ComparativeExperimentCell(tier="empirical_standard", ...)` で設定を検証してください。

基本条件は対応する seed 0..9、共通 Sobol 初期 20 点、追加 50 評価、q=1・q=3 を別々に集計、CPU・float64 です。q=3 は 17 batch で最後が 2 評価になります。CUDA・ノイズ条件を混ぜません。多目的 reference point は結果を見ずに事前指定します。

実行前に SHA・dirty 状態・環境と version・全 seed・予算・dtype/device・目的方向・ノイズ・モデルと学習・MC sampler・optimizer 予算・失敗方針を manifest に保存します。必要事項が欠けた場合は、結果から都合よく復元せず失敗として扱います。

候補生成は `CandidateProblemView` と観測済みデータに制限し、真値 evaluator・最適値・未来観測を見せません。seed ごとの初期 X を揃え、ノイズ方針が保証する場合に初期観測 Y も一致させます。成功 run は合計 70 点、追加履歴末尾 50 を満たします。

真値の best-so-far・既知最適値の regret・固定 reference hypervolume を追加評価数に対して報告します。時間は候補生成・評価・総時間を分け、失敗 run と例外・成功 seed 数も明示します。Random・Sobol との差は対応 seed の 95% bootstrap 区間で評価し、失敗があれば対応比較を不完全とします。周辺区間の重なりだけで優越性を宣言しません。

失敗 seed・設定・traceback・途中軌跡・環境を保全し、別 seed に差し替えません。再実行は同じ commit と凍結 manifest で試行数・結果を記録し、最良試行だけを選びません。hyperparameter・予算・ノイズ・reference point・seed を変えるなら version 付き改訂と新 cohort にします。否定的・不確定・未対応の結果も報告してください。

適格性の全行列・厳密な受入条件は[評価契約の正本](../../docs-internal/contracts/benchmarks/empirical-evaluation-protocol.md)を参照してください。日本語の[全体手順](end_to_end.md)と併せて使用します。

# Active Learning の境界問題

決定的な二値分類の真値を持つ 2 問題を提供します。

- `circle_boundary`：中心 `(0.5, 0.5)` の閉じた境界。
- `wave_boundary`：正弦波による非線形境界。

どちらも `[0, 1]^2` 上で定義し、符号付き margin は `circle_margin(X)`・`wave_margin(X)` で取得します。境界を含む非負 margin は class 1、負は class 0 です。正準の 0/1 label は `active_learning_labels(X, boundary="circle"|"wave")` で生成します。

`BenchmarkProblem` が目的を要求するため、定数 0 の dummy objective と、境界真値を表す出力制約 1 本を使います。最適化 regret は Active Learning 性能の指標になりません。

独立に生成して固定した reference 集合を使い、獲得予算ごとに次を記録してください。

- `classification_accuracy(labels, predictive_probabilities)`：閾値 0.5 の正解率。
- `brier_score(labels, predictive_probabilities)`：確率の較正・識別を評価する Brier score。
- `boundary_mae(true_margin, predicted_margin)`：符号付き margin を予測できる場合の平均絶対誤差。幾何的な Hausdorff 距離ではありません。

ランダム候補と seed 付き runner は再現可能な smoke test 用です。この問題定義自体はモデル学習、予測 entropy、BALD、boundary-straddle、学習曲線の実行制御を提供しません。それらには Active Learning 実験用 runner が必要です。

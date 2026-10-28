# 比較実験の計画と適格性

比較 tier は実行 profile とは別です。Smoke は 1 seed・初期 4 点・追加 4 評価、従来 Standard は 5 seed・20 点・40 評価、Empirical Standard は 10 seed・20 点・50 評価です。Full は 20 seed・30 点・100 評価で q=1/3/5 を分けます。通常は CPU・float64 とし、CUDA は別 cohort にします。

実測の既定は `empirical_standard`、q=1 と q=3 は独立に集計し、後者の最後は 2 点です。seed は各 tier で 0..N-1、初期設計・評価予算を戦略間で揃え、従来 tier と混ぜません。

実行前に registration、目的方向、q 対応、初期設計、oracle 分離、ノイズ系列、真値指標、reference point、設定と環境の保存を確認してください。Smoke の成功や static eligibility は性能比較の結果ではありません。分類制約、3 目的 hypervolume、未対応の観測履歴には追加の適格性条件があります。

各 cell の問題・手法・tier・q・seed・設定・適格性と理由・commit・実行状態・artifact を機械可読に保存します。`ComparativeExperimentCell` の構成検証だけでは callback の安全性や実行済みを保証しません。

適格性の全行列・厳密な受入条件は[評価契約の正本](../../docs-internal/contracts/benchmarks/comparative-experiment-matrix.md)を参照してください。日本語の[全体手順](end_to_end.md)と併せて使用します。

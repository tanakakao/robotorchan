# LogEI ベンチマーク設計メモ

このベンチマークでは候補点を1点ずつ追加するため、analytic `LogExpectedImprovement` を使用します。`best_f` は各 iteration の学習データから再計算し、探索戦略間で同じ acquisition definition を共有します。

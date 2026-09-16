# Phase 11 完了記録

高次元入力モデルのPhase 11では、モデル実装後の比較・選択・利用方法を整備しました。

完了した項目:

- predictive benchmark: RMSE / NLL / training / posterior timing
- acquisition evaluation timing
- MAP-SAAS / opt-in Fully Bayesian SAAS benchmark
- finite candidate pool上のSequential BO benchmark
- neural / supervised / joint / MAP-SAASへの比較対象拡張
- repeated-seed実行とsimple regret統計
- 日本語モデル選択ガイド
- 日本語benchmark Notebook
- Joint neuralモデルの共通 `training_loss()` 契約
- 高次元入力ドキュメントの実装済み状態への更新

Phase 11のpool-based BO benchmarkは、surrogateとacquisition評価の比較を目的とし、連続D次元空間でのacquisition optimization難易度を意図的に分離しています。

今後はPhase 11の延長ではなく、探索戦略、連続acquisition optimization、Joint model lifecycle、VAE安定化などを独立した拡張テーマとして扱います。

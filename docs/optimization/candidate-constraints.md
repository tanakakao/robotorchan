# 最適化 backend ごとの候補制約

`CandidateConstraints` は提案可能な入力候補を制限する共通契約です。獲得関数を構成する出力制約とは区別します。

## Backend ごとの適用方法

| Backend | 候補制約の扱い |
| --- | --- |
| BoTorch / BoTorch-mixed | 対応する線形・非線形制約を BoTorch に渡す |
| Differential Evolution | 共通違反量を SciPy `NonlinearConstraint` に変換する |
| CMA-ES | 探索目的に違反の罰則を加える |
| Continuous GA | 実行可能な候補を優先する |
| Mixed-variable GA | 混合領域の演算子と実行可能性優先の順位付けを使う |
| PSO | 粒子のスコアに罰則を加え、返り値には元の獲得値を使う |
| Hybrid | 大域・局所の両段階に渡す。大域段階の扱いは backend に従う |
| Random / Sobol | 未対応として明示的に拒否する |
| PyTorch candidate generator | 未対応として明示的に拒否する |

微分不要の backend は制約評価を共有しますが、同一の罰則法は使いません。CMA-ES と PSO の正の `constraint_penalty` が DE や GA の処理を制御するわけではありません。Hybrid は DE・CMA-ES・Mixed GA による大域探索と BoTorch の局所改良を組み合わせ、段階ごとの処理を保ちます。

`equality_tolerance` は既定で `1e-6` です。点内制約と q-batch の点間制約に対応します。微分不要の backend でも制約 callable は有効な Tensor を返す必要があります。BoTorch の勾配法では微分可能性も必要です。

## 実行可能性の確認

制約対応は毎回実行可能な候補を返す保証ではありません。CMA-ES・PSO の罰則法は違反候補を返す場合があり、実行可能性優先の順位付けも評価した集団に依存します。厳密な実行可能性が必要なら返り値を元の制約で確認してください。

未対応の候補制約を出力の実行可能性重みや近似罰則へ暗黙に置き換えません。dispatch 前に backend の capability を確認してください。

# Safe Bayesian Optimization 実装ガイド

[Safe BOの理論](../theory/25_safe_bayesian_optimization.md)で導入した安全集合と信頼区間が、robotorchanでどのように実装されるかを説明する。実行可能なコードは[日本語利用Notebook](../../examples/notebooks/safe_bo_usage_guide.ipynb)に集約する。

## 理論と実装の対応

| 理論上の役割 | 実装 | 留意点 |
|---|---|---|
| 安全な初期観測と安全制約 | `semantics.safety.SafetySpecification` | 実測安全制約値を検証する |
| 安全出力の制約方向・閾値 | `semantics.constraints.ContinuousConstraint` | 未知の出力制約。入力空間の既知制約とは異なる |
| GP事後信頼区間と安全集合判定 | `semantics.safety_bounds.safety_confidence_bounds` | `beta` は固定可能。理論上の安全保証ではない |
| 安全候補内の獲得関数最大化 | `optim.safe.optimize_safe_acqf` | Sobol候補集合とBoTorchの離散獲得関数最適化を使用 |
| 安全集合内の不確実性探索 | `optim.safe_expansion.propose_safe_expansion` | SafeOptの厳密なexpander集合は計算しない |
| 逐次候補提案・実測記録 | `optim.safe_workflow.SafeBOWorkflow` | `ask()` / `tell()`。モデル再学習は利用者が担当 |

## APIの責務分離

`SafetySpecification` は安全制約、初期安全観測、信頼係数を保持する。`ContinuousConstraint` は安全出力のインデックスまたは名前、閾値、方向を表す。これらは既知の入力制約を表す `CandidateConstraints` とは異なる。

`safety_confidence_bounds` は、GP事後の平均と分散から制約ごとの安全余裕と、すべての制約を満たす真偽マスクを返す。上限制約の安全余裕は $h-\mu-\sqrt{\beta}\sigma$、下限制約では $\mu-\sqrt{\beta}\sigma-l$ である。

`optimize_safe_acqf` は有限のSobol候補集合を生成し、安全候補のみを `botorch.optim.optimize_acqf_discrete` に渡す。安全候補が見つからない場合はエラーにする。連続空間全域での大域最適性を主張しない。

`propose_safe_expansion` は安全候補内で制約出力の事後標準偏差の和が最大の候補を選ぶ。到達可能性や拡張可能性を証明するアルゴリズムではない。

`SafeBOWorkflow` は、目的関数最適化モード `objective` と安全集合内の探索モード `expansion` を切り替える。候補を `ask()` で受け取り、実測安全出力を `tell()` で記録する。目的関数の観測履歴、GP学習、獲得関数の生成、実験の実行は呼び出し側の責務である。

## 対応範囲

| 項目 | 対応 |
|---|---|
| 連続入力・逐次候補 `q=1` | 対応 |
| 連続値の安全制約 | 対応 |
| 複数の安全制約 | 対応 |
| 安全な初期実測値の検証 | 対応 |
| 上下限方向の信頼区間判定 | 対応 |
| 安全候補の獲得関数最適化 | 対応（有限候補） |
| 安全集合内の不確実性探索 | 対応（近似的） |
| 逐次 `ask/tell` | 対応 |
| GPの自動再学習 | 未対応 |
| SafeOptの厳密な拡張点・到達可能性 | 未対応 |
| 時間全体にわたる確率的安全保証 | 未対応 |
| Mixed、`q>1`、非同期、MF、複数目的 | 今回の範囲外 |

## Notebook

- [Safe BO日本語利用ガイド](../../examples/notebooks/safe_bo_usage_guide.ipynb)：設定、学習、候補提案、観測登録、再学習、探索モード切替
- [Safe BO比較実験](../../examples/notebooks/safe_bo_benchmark.ipynb)：予測平均制約、保守的信頼区間、安全集合内探索の比較

両Notebookは手動実行対象であり、登録や構文検査だけでは実行成功の証拠にならない。

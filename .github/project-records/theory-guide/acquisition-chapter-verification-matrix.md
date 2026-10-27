# 獲得関数12章：理論・実装・テストの検証対応表

本表は固定48章中の獲得関数12章について、確認可能な証拠と未検証の範囲を区別する。
モデル24章、最適化12章、旧Mappingの移行検収とは独立の判定である。

## 証拠の区分

- **掲載例実行**：Markdownのコードブロックを直接実行するテストへの参照。
  ファイルが存在することと、各CIジョブでpassしたことは別に確認する。
- **独立数値参照**：実装の獲得値を、別経路で計算した数式・値と照合する。
- **—**：この表では直接の証拠を登録していない。機能未実装を意味しない。
- ガイドの理論リンクが有効でも、API・学習収束・統計推定誤差を保証しない。

## 12章の証拠

| 章 | 理論 / 実装ガイド | 掲載例実行 | 独立数値参照 |
| --- | --- | --- | --- |
| 01 | Foundations | — | — |
| 02 | Improvement / LogEI / PI | analytic_examples（LogEI） | Gaussian EI reference |
| 03 | Confidence Bound / UCB | analytic_examples（UCB） | Gaussian UCB reference |
| 04 | Batch / Noisy | batch_noisy_example（qLogNEI） | — |
| 05 | Information Theoretic | information_cost_examples（MES/GIBBON） | — |
| 06 | Lookahead | lookahead_example（qKG augmented q） | — |
| 07 | Multiobjective | multiobjective_example（qLogEHVI） | — |
| 08 | Constraints | constraints_example（continuous qEI） | — |
| 09 | Active Learning | active_learning_example（regression） | posterior-moment reference |
| 10 | Level Set | level_set_example（Straddle） | posterior-moment reference |
| 11 | Multifidelity / Cost Aware | information_cost_examples（fidelity cost） | affine cost formula |
| 12 | Selection Guide | —（選択・案内章） | — |

## 判定と未検証事項

**構造・学習導線：達成。** 全12章があり、理論・実装の説明へ到達できる。
第01章Foundationsと第12章Selection Guideは説明・選択の導線が中心のため、
個別数値獲得関数のコード実行とは別基準で扱う。

**掲載例テスト：第02～11章の10章に登録。**
第02・03章はGaussian EI/UCBの式、第09・10章はposterior moments、
第11章はaffine fidelity costの値を比較する。
第04～08章のテストでは主にAPI・shape・有限性・候補構造を検査する。
登録数は全Pythonブロック実行数や全獲得関数の実証件数ではない。

**獲得関数12章の完全技術検収：保留。**
情報利得の推定精度、q-batchのjoint MC、分類PoFの校正、
noisy/constrained hypervolume、KGのone-shot最適化の結果、
全API・全制約・全組合せの数値的正当性は未網羅。

## 直接参照したテスト

| 対象 | ファイル | 確認の限界 |
| --- | --- | --- |
| LogEI / UCB | `tests/docs/test_theory_guide_acquisition_analytic_examples.py` | 固定Gaussian式との照合 |
| qLogNEI | `tests/docs/test_theory_guide_acquisition_batch_noisy_example.py` | q=2のshapeと有限性 |
| MES / GIBBON | `tests/docs/test_theory_guide_acquisition_information_cost_examples.py` | スコアの有限性 |
| qKG | `tests/docs/test_theory_guide_acquisition_lookahead_example.py` | augmented qと実験候補の抽出 |
| qLogEHVI | `tests/docs/test_theory_guide_acquisition_multiobjective_example.py` | 2目的・ref point・shape |
| constrained qEI | `tests/docs/test_theory_guide_acquisition_constraints_example.py` | 連続outcome制約のshape |
| 回帰AL | `tests/docs/test_theory_guide_acquisition_active_learning_example.py` | posterior variance/std |
| Straddle | `tests/docs/test_theory_guide_acquisition_level_set_example.py` | posterior momentと閾値 |
| Fidelity cost | `tests/docs/test_theory_guide_acquisition_information_cost_examples.py` | affine costの計算値 |

## 次の監査ゲート

最適化12章の公開optimizer API、候補制約、mixed/one-shot、最終候補shapeを
同様に照合する。獲得関数12章の未実証事項と、旧Mappingの174見出しの
原文区間照合はPhase18の未完了ゲートとして維持する。
旧Mapping5本の内容や固定18 Phase・48章の構成を変更しない。

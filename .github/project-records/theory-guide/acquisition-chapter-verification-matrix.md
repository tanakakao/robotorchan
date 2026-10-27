# 獲得関数12章：理論・実装・テストの照合

モデル24章の条件付き監査に続き、獲得関数12章の検証を開始する。
理論・実装・掲載例実行・独立数値参照を別の根拠として扱う。

| 章 | 理論 / 実装ガイド | 掲載例実行 | 独立数値参照 |
| --- | --- | --- | --- |
| 01 | Foundations | — | — |
| 02 | Improvement / LogEI / PI | analytic_examples（LogEI） | Gaussian EI reference |
| 03 | Confidence Bound / UCB | analytic_examples（UCB） | Gaussian UCB reference |
| 04 | Batch / Noisy | batch_noisy_example（qLogNEI） | — |
| 05 | Information Theoretic | — | — |
| 06 | Lookahead | — | — |
| 07 | Multiobjective | — | — |
| 08 | Constraints | — | — |
| 09 | Active Learning | — | — |
| 10 | Level Set | — | — |
| 11 | Multifidelity / Cost Aware | — | — |
| 12 | Selection Guide | —（選択・案内章） | — |

`batch_noisy_example`は`tests/docs/test_theory_guide_acquisition_batch_noisy_example.py`を表し、q=2のbaseline付きMC獲得値を実行・shape検査する。独立数値参照ではない。

表中の`analytic_examples`は
`tests/docs/test_theory_guide_acquisition_analytic_examples.py`を表す。
Markdownの掲載コードを直接実行し、別経路の正規分布公式と比較する。
既存の獲得関数単体テストやMappingの全件調査が完了した意味ではない。
CI未実行・失敗・skipの際は検証成功と認定しない。

続いてBatch/Noisy、Constraints、Active Learningなど、
独自posterior形状・制約意味論を持つ章を優先する。
元Mappingと174見出しのverified状態を推測で変更しない。

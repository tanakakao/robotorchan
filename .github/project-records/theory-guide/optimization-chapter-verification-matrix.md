# 最適化12章：理論・実装・テストの検証対応表

獲得関数12章の条件付き監査に続き、候補最適化12章の直接検証証拠を整理する。
構造・掲載例実行・独立数値照合・最適化性能は別々の判定とする。

| 章 | 主題 | 掲載例直接実行 | 独立数値参照 |
| --- | --- | --- | --- |
| 01 | Foundations | —（概念・導線章） | — |
| 02 | Gradient Based | gradient_example（BoTorch backend） | — |
| 03 | Initialization | — | — |
| 04 | Mixed / Discrete | — | — |
| 05 | Constraints | — | — |
| 06 | Derivative Free | — | — |
| 07 | Evolutionary | — | — |
| 08 | Batch / One-shot | — | — |
| 09 | High Dimensional | — | — |
| 10 | Trust Region | — | — |
| 11 | Hybrid | — | — |
| 12 | Selection Guide | —（比較・選択章） | — |

`gradient_example`は
`tests/docs/test_theory_guide_optimization_gradient_example.py`を示す。
`SingleTaskGP`とBoTorch analytic UCBに対し、robotorchan公開dispatcherから
`optimizer="botorch"`で候補を生成し、shape・有限性・boundsを確認する。
このテストは大域最適性・全制約・全backend互換性を証明しない。

次はInitialization、Mixed / Discrete、Constraints、Batch / One-shotなどの
異なる最適化契約を検証する。旧Mapping5本と174見出しの状態は変更しない。

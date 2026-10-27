# 最適化12章：理論・実装・テストの検証対応表

獲得関数12章の条件付き監査に続き、候補最適化12章の直接検証証拠を整理する。
構造・掲載例実行・独立数値照合・最適化性能は別々の判定とする。

| 章 | 主題 | 掲載例直接実行 | 独立数値参照 |
| --- | --- | --- | --- |
| 01 | Foundations | —（概念・導線章） | — |
| 02 | Gradient Based | gradient_example（BoTorch backend） | — |
| 03 | Initialization | initialization_example（explicit starts） | — |
| 04 | Mixed / Discrete | mixed_discrete_example（fixed-feature列挙） | — |
| 05 | Constraints | constraints_example（linear candidate） | feasibility tolerance |
| 06 | Derivative Free | sampling_evolutionary_examples（Sobol） | pool内argmax照合 |
| 07 | Evolutionary | sampling_evolutionary_examples（NSGA-II） | 返却集合内の非支配性 |
| 08 | Batch / One-shot | batch_one_shot_example（qKG） | augmented q抽出形状 |
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

第03章は`tests/docs/test_theory_guide_optimization_initialization_example.py`で明示的な初期点を指定した公開dispatcherの実行を検査する。Sobol poolの品質、constraint-aware initialization、one-shot初期化の全面検収ではない。

第04章は`tests/docs/test_theory_guide_optimization_mixed_discrete_example.py`でnative mixed optimizerのカテゴリ固定候補を直接実行する。ここでのGPは数値入力を連続扱いする簡易モデルであり、categorical kernelの学習やMixedVariableSpace dispatcherの全面検証ではない。

第05章は`tests/docs/test_theory_guide_optimization_constraints_example.py`で公開dispatcherによる線形candidate制約を直接実行し、許容誤差1e-6内の候補可行性を確認する。outcome制約・非線形・inter-point・全backendの保証ではない。

第06・07章の掲載例を`tests/docs/test_theory_guide_optimization_sampling_evolutionary_examples.py`で直接実行する。Sobolは有限サンプル内の最大スコア、NSGA-IIは返却された近似Pareto集合の内部非支配性を検査する。真の大域最適解・Pareto frontへの収束や制約対応の保証ではない。

第08章は`tests/docs/test_theory_guide_optimization_batch_one_shot_example.py`でqKGの拡張q=6から実験候補q=2を抽出する掲載コードを直接実行する。獲得関数の最適化成功やMixed one-shot q>1対応の証拠ではない。

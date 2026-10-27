# 獲得関数12章：理論・実装・テストの照合

モデル24章の条件付き監査に続き、獲得関数12章の検証を開始する。
理論・実装・掲載例実行・独立数値参照を別の根拠として扱う。

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

`batch_noisy_example`は`tests/docs/test_theory_guide_acquisition_batch_noisy_example.py`を表し、q=2のbaseline付きMC獲得値を実行・shape検査する。独立数値参照ではない。

`constraints_example`は`tests/docs/test_theory_guide_acquisition_constraints_example.py`を表し、
第08章の既存`~~~python`コードを直接実行して返り値の形状・有限性を確認する。
候補点制約や分類PoFを含む全builderの検証と同義ではない。

表中の`analytic_examples`は
`tests/docs/test_theory_guide_acquisition_analytic_examples.py`を表す。
Markdownの掲載コードを直接実行し、別経路の正規分布公式と比較する。
既存の獲得関数単体テストやMappingの全件調査が完了した意味ではない。
CI未実行・失敗・skipの際は検証成功と認定しない。

続いてBatch/Noisy、Constraints、Active Learningなど、
独自posterior形状・制約意味論を持つ章を優先する。
元Mappingと174見出しのverified状態を推測で変更しない。

第09章の掲載例と独立moment参照は`tests/docs/test_theory_guide_acquisition_active_learning_example.py`、既存のより広い回帰AL数式検証は`tests/acquisition/test_phase8_regression_al_formula_reference.py`で確認する。分類entropyとBALDの完全な統合実行を認定するものではない。

第10章の掲載例は`tests/docs/test_theory_guide_acquisition_level_set_example.py`で直接実行し、Straddleの閾値距離とposterior標準偏差の式を別経路で確認する。境界分類の較正やbatch最適化の実証ではない。

第05章は既存のMES/GIBBON掲載コードを直接実行するが、情報利得の独立数値照合は行わない。第11章のコスト関数は`tests/docs/test_theory_guide_acquisition_information_cost_examples.py`で固定fidelityごとの線形コストを確認する。これはmulti-fidelity KGの最適化性能やone-shot動作の検証ではない。

第06章は`tests/docs/test_theory_guide_acquisition_lookahead_example.py`でone-shot候補の拡張q次元と`extract_candidates`を検証する。KG獲得値や最適化性能の独立数値検証ではない。第07章Multiobjectiveの数値検証は次の監査対象とする。

第07章の掲載例は`tests/docs/test_theory_guide_acquisition_multiobjective_example.py`で2目的のreference point、partitioning、qLogEHVIのAPI実行を確認する。hypervolume改善量の独立数値参照ではない。

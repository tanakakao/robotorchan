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
| 09 | High Dimensional | high_dimensional_example（HeSBO） | latent→original照合 |
| 10 | Trust Region | trust_region_example（TuRBO） | expand/shrink state values |
| 11 | Hybrid | hybrid_example（DE→BoTorch local） | shape / bounds / finite |
| 12 | Selection Guide | —（比較・選択章） | — |

`gradient_example`は
`tests/docs/test_theory_guide_optimization_gradient_example.py`を示す。
`SingleTaskGP`とBoTorch analytic UCBに対し、robotorchan公開dispatcherから
`optimizer="botorch"`で候補を生成し、shape・有限性・boundsを確認する。
このテストは大域最適性・全制約・全backend互換性を証明しない。

第03章は`tests/docs/test_theory_guide_optimization_initialization_example.py`で明示的な初期点を指定した公開dispatcherの実行を検査する。Sobol poolの品質、constraint-aware initialization、one-shot初期化の全面検収ではない。

第04章は`tests/docs/test_theory_guide_optimization_mixed_discrete_example.py`でnative mixed optimizerのカテゴリ固定候補を直接実行する。ここでのGPは数値入力を連続扱いする簡易モデルであり、categorical kernelの学習やMixedVariableSpace dispatcherの全面検証ではない。

第05章は`tests/docs/test_theory_guide_optimization_constraints_example.py`で公開dispatcherによる線形candidate制約を直接実行し、許容誤差1e-6内の候補可行性を確認する。outcome制約・非線形・inter-point・全backendの保証ではない。

第06・07章の掲載例を`tests/docs/test_theory_guide_optimization_sampling_evolutionary_examples.py`で直接実行する。Sobolは有限サンプル内の最大スコア、NSGA-IIは返却された近似Pareto集合の内部非支配性を検査する。真の大域最適解・Pareto frontへの収束や制約対応の保証ではない。

第08章は`tests/docs/test_theory_guide_optimization_batch_one_shot_example.py`でqKGの拡張q=6から実験候補q=2を抽出する掲載コードを直接実行する。獲得関数の最適化成功やMixed one-shot q>1対応の証拠ではない。

第09章は`tests/docs/test_theory_guide_optimization_high_dimensional_example.py`でHeSBOの元6次元／潜在2次元の候補と`project`による復元を掲載コードから直接検査する。REMBO・ALEBO・BAxUS全体の精度や低次元探索のregret保証ではない。

第10章は`tests/docs/test_theory_guide_optimization_trust_region_example.py`で掲載コードを直接実行し、観測後の2回成功で幅拡大、2回失敗で幅縮小する状態遷移を検査する。restart・最適化候補・regret改善の実証ではない。

第11章は`tests/docs/test_theory_guide_optimization_hybrid_example.py`で公開dispatcherのhybrid経路を通るDE→BoTorch局所探索の掲載例を直接実行し、候補shape・bounds・finiteを検査する。global stageの最適性や制約付きmixedの保証ではない。

## 最適化12章の条件付き完了判定

**学習導線・代表例整備：一区切り。技術的な全面検収：未完了。**

第01章Foundationsと第12章Selection Guideは、理論と実装の選択・導線を
扱う章である。第02〜11章の10章には、Markdown掲載コードを直接実行する
テストファイルを対応付けた。PR #1525はマージ済みで、当該HEADのCIと
Documentation linksは成功している。ただし、全環境・全掲載コードの
個別成功や、全アルゴリズムの数値同値性を意味しない。

| 監査項目 | 判定 | 保留する範囲 |
| --- | --- | --- |
| 12章の構成・理論への導線 | 整備済み | 数式・APIの全件照合 |
| 第02〜11章の掲載例実行テスト | 10章に登録 | 各章のすべての例・オプション |
| 候補のshape・bounds・有限性 | 代表例にassertionあり | 全backend・q・mixedの組合せ |
| 入力制約の可行性 | 線形の限定例で検査 | 非線形・inter-point・penalty backend |
| one-shot候補抽出 | qKGの拡張行と実候補を照合 | KGの獲得値最適化・mixed q>1 |
| HeSBO座標変換・TuRBO更新 | 代表例を検査 | その他embedding・restart・regret |
| 最適化性能・大域最適性 | **未検収** | 収束、計算効率、seed間ばらつき |
| 旧Mapping174見出し | **保留** | 原文区間の意味・negative contract移行 |

**次のゲート：** モデル24章、獲得関数12章、最適化12章を横断して
理論参照・実装所有権・掲載コード・検証根拠・未検証境界を照合する。
旧Mapping5本と固定18 Phase・48章の構成を維持する。

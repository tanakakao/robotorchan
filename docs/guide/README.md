# 理論と実装をつなぐ日本語技術ガイド — 章構成

## このガイドの目的

[Theory](../theory/README.md)の数式・アルゴリズムが、robotorchanのどのクラス・
メソッドによって、どのような処理として実現されるかを解説します。
内部構造、処理の流れ、設計判断を説明し、BoTorch・GPyTorchへの委譲と独自の拡張を区別します。
使用手順や実行結果を再掲せず、実装を理解するために必要なソースへ導きます。

| 文書 | 担当する内容 |
| --- | --- |
| Theory | 数学的理論、導出、仮定、アルゴリズム |
| Guide | 理論と実装の対応、内部構造、処理の仕組み、設計判断 |
| Notebooks | 使用方法、実行例、実験、結果の可視化 |
| API Reference | 公開API、引数、戻り値、使用上の契約 |
| Tests | 正当性、回帰、境界条件の検証 |
| Development | 開発記録、監査履歴、進行管理、CI関連情報 |

## 最初に読む：モデルから実験候補まで

本ガイドは数学の正本である理論章と並行して読むための技術書です。
最初から全48章を読む必要はありません。まず次の順に進み、
各章で、数式に対応する部品と内部の処理経路を確認してください。

1. **観測から予測へ：**
   [ベイズ最適化](01_bayesian_optimization.md) →
   [GP](02_gaussian_process.md)で`train_X[n,d]`、
   `train_Y[n,1]`からposteriorを求める実装を追います。
2. **予測から探索優先度へ：**
   [獲得関数の基礎](04_acquisition_function.md) →
   [Improvement / LogEI](acquisition/02_improvement.md)で、
   高価な目的関数の観測値と計算上のacquisition scoreを区別します。
3. **探索優先度から候補点へ：**
   [最適化の基礎](optimization/01_foundations.md) →
   [Gradient-based](optimization/02_gradient_based.md)で
   `q × d`の候補を生成する数値探索の内部を読みます。
4. **観測更新との接続を読む：**
   候補を選ぶ部品と、外部実験の観測をモデルへ戻す処理の境界を確認します。
   実験を含む反復の実行例はNotebookを参照してください。

| 実行上の境界 | 主な入力と出力 | 関連章 |
| --- | --- | --- |
| Model | 観測`[n,d]` → posterior mean/variance | [GP](02_gaussian_process.md) |
| Acquisition | posteriorと候補`[...,q,d]` → utility | [Improvement](acquisition/02_improvement.md) |
| Candidate optimizer | acquisitionとbounds`[2,d]` → 候補`[q,d]` | [Gradient-based][link01] |
| 外部実験 | 候補`[q,d]` → 新しい観測Y | [BOの総論](01_bayesian_optimization.md) |

### 条件に応じて分岐する

- **混合変数：** [Mixedモデル](05_mixed_variables.md)と
  [Mixed候補最適化](optimization/04_mixed_discrete.md)。
  カテゴリ列の符号化とoptimizerのfixed-feature列挙は別の責務です。
- **制約：** [Constrained acquisition](acquisition/08_constraints.md)と
  [Candidate constraints](optimization/05_constraints.md)。
  未知の出力Yへの制約と既知の候補Xへの制約を混ぜないでください。
- **バッチ／Lookahead：** [KG](acquisition/06_lookahead.md)と
  [Batch / one-shot](optimization/08_batch_one_shot.md)。
  Fantasy補助変数を実験候補として扱わないでください。
- **多目的：** [EHVI](acquisition/07_multiobjective.md)と
  [進化的最適化](optimization/07_evolutionary.md)。
  スカラー獲得関数の最適化とNSGA-IIによるベクトル目的探索は異なります。
- **高次元：** [高次元探索](20_high_dimensional_search.md)と
  [低次元座標での最適化](optimization/09_high_dimensional.md)。
  元の実験座標Xと内部探索座標Zを区別してください。

これらは実装を読むための経路です。モデルと獲得関数、探索手法の処理を分けて追います。
横断的なデータと責務の境界は、
[モデル・獲得関数・最適化の契約](../../docs-internal/contracts/theory-guide/cross-layer-api-contracts.md)
も参照できます。

## 章構成：理論の24章と1対1対応

TheoryとGuideは章番号とファイル名を対応させています。
以下の表から、数学的な説明と、その実装を読む章へ進めます。

| 理論の正本 | 実装の解説 |
| --- | --- |
| [01_bayesian_optimization.md][link02] | [Guide][link03] |
| [02_gaussian_process.md][link04] | [Guide][link05] |
| [03_kernel.md][link06] | [Guide][link07] |
| [04_acquisition_function.md][link08] | [Guide][link09] |
| [05_mixed_variables.md][link10] | [Guide][link11] |
| [06_multi_fidelity.md][link12] | [Guide][link13] |
| [07_multitask_multioutput.md][link14] | [Guide][link15] |
| [08_high_dimensional_gp.md][link16] | [Guide][link17] |
| [09_variational_gp.md][link18] | [Guide][link19] |
| [10_preference_learning.md][link20] | [Guide][link21] |
| [11_structured_output.md][link22] | [Guide][link23] |
| [12_hierarchical_contextual_gp.md][link24] | [Guide][link25] |
| [13_model_selection.md][link26] | [Guide][link27] |
| [14_robust_gaussian_process.md][link28] | [Guide][link29] |
| [15_heteroskedastic_noise.md][link30] | [Guide][link31] |
| [16_uncertain_input_gp.md][link32] | [Guide][link33] |
| [17_nonstationary_gp.md][link34] | [Guide][link35] |
| [18_dimensionality_reduction_gp.md][link36] | [Guide][link37] |
| [19_neural_reduction_gp.md][link38] | [Guide][link39] |
| [20_high_dimensional_search.md][link40] | [Guide][link41] |
| [21_advanced_high_dimensional_models.md][link42] | [Guide][link43] |
| [22_expressive_gp.md][link44] | [Guide][link45] |
| [23_non_gp_surrogates.md][link46] | [Guide][link47] |
| [24_classification.md][link48] | [Guide][link49] |

## 獲得関数：理論12サブ章と1対1対応

| 理論の正本 | 実装の解説 |
| --- | --- |
| [acquisition/01_foundations.md][link50] | [Guide][link51] |
| [acquisition/02_improvement.md][link52] | [Guide][link53] |
| [acquisition/03_confidence_bound.md][link54] | [Guide][link55] |
| [acquisition/04_batch_noisy.md][link56] | [Guide][link57] |
| [acquisition/05_information_theoretic.md][link58] | [Guide][link59] |
| [acquisition/06_lookahead.md][link60] | [Guide][link61] |
| [acquisition/07_multiobjective.md][link62] | [Guide][link63] |
| [acquisition/08_constraints.md][link64] | [Guide][link65] |
| [acquisition/09_active_learning.md][link66] | [Guide][link67] |
| [acquisition/10_level_set.md][link68] | [Guide][link69] |
| [acquisition/11_multifidelity_cost_aware.md][link70] | [Guide][link71] |
| [acquisition/12_selection_guide.md][link72] | [Guide][link73] |

## 獲得関数の最適化：理論12サブ章と1対1対応

| 理論の正本 | 実装の解説 |
| --- | --- |
| [optimization/01_foundations.md][link74] | [Guide][link75] |
| [optimization/02_gradient_based.md][link76] | [Guide][link01] |
| [optimization/03_initialization.md][link77] | [Guide][link78] |
| [optimization/04_mixed_discrete.md][link79] | [Guide][link80] |
| [optimization/05_constraints.md][link81] | [Guide][link82] |
| [optimization/06_derivative_free.md][link83] | [Guide][link84] |
| [optimization/07_evolutionary.md][link85] | [Guide][link86] |
| [optimization/08_batch_one_shot.md][link87] | [Guide][link88] |
| [optimization/09_high_dimensional.md][link89] | [Guide][link90] |
| [optimization/10_trust_region.md][link91] | [Guide][link92] |
| [optimization/11_hybrid.md][link93] | [Guide][link94] |
| [optimization/12_selection_guide.md][link95] | [Guide][link96] |

## 各章の執筆契約

1. **Theoryとの対応**：該当する章・節を示し、数式の構成要素を実際のクラス・メソッドへ対応付けます。
   数式の導出そのものは再掲しません。
2. **内部処理**：クラス構造と継承、主要メソッド、処理順序、データ変換を説明します。
   対応表だけで終わらず、なぜその構成が必要かを文章でつなぎます。
3. **実装の照合**：現行ソースを読み、実在する識別子とファイルへの参照を示します。
   公開入口と内部計算を区別し、一般的な手法の知識だけから実装を推測しません。
4. **担当範囲**：上流へのDelegated、既存部品のExtended、独自のImplementedを区別します。
   近似、制約、非対応の機能は、その設計上の理由とともに必要な範囲で説明します。
5. **コードの選択**：実装の理解に必要な実際のクラス・メソッドや計算処理の抜粋に限ります。
   コード掲載は必須ではありません。使用例・データ生成・描画・検証コードは本文へ複製しません。
6. **重複の整理**：使用例はNotebook、公開契約はAPIの定義、検証はTestsへ参照を設けます。
   対応する資料がなければ、必要性を確認せずに新たな資料を複製・追加しません。
7. **説明の整合**：日本語の論理、章間リンク、識別子、数式、行長、空行を確認します。
   開発記録や監査・CIの履歴は学習本文へ含めません。

## 既存文書との責務境界

使用方法とモデル選択は[モデル文書](../models/README.md)・
[最適化文書](../optimization/README.md)、
実行と可視化は[Notebook一覧](../../examples/README.md)へ進めます。
APIの詳細は、各章から参照する公開クラス・関数の定義とdocstringを正本として読みます。

内部構造を横断して読む資料は[Architecture](../../docs-internal/architecture/README.md)、
契約は[Contracts](../../docs-internal/contracts/README.md)にあります。
検証方法は[Verification](../../docs-internal/verification/README.md)、
文書の保守手順は[Maintenance](../../docs-internal/maintenance/README.md)を参照してください。
開発履歴は[Project records](../../.github/project-records/README.md)へ分けます。

## 学習ガイドの完成基準

読者が「Theoryで学んだ数式・アルゴリズムは、どのクラス・メソッドによって、
どのような処理として実現されるのか」に答えられることを完成基準とします。
使用方法や実行例だけが理解できる状態では、この目的を満たしません。

各章は、既存の有用な実装説明を再利用しながら、対応付け、処理経路、設計判断を補います。
独自実装と委譲の区別、近似や適用上の制約も、実際のソースに即して説明します。

[link01]: optimization/02_gradient_based.md
[link02]: ../theory/01_bayesian_optimization.md
[link03]: 01_bayesian_optimization.md
[link04]: ../theory/02_gaussian_process.md
[link05]: 02_gaussian_process.md
[link06]: ../theory/03_kernel.md
[link07]: 03_kernel.md
[link08]: ../theory/04_acquisition_function.md
[link09]: 04_acquisition_function.md
[link10]: ../theory/05_mixed_variables.md
[link11]: 05_mixed_variables.md
[link12]: ../theory/06_multi_fidelity.md
[link13]: 06_multi_fidelity.md
[link14]: ../theory/07_multitask_multioutput.md
[link15]: 07_multitask_multioutput.md
[link16]: ../theory/08_high_dimensional_gp.md
[link17]: 08_high_dimensional_gp.md
[link18]: ../theory/09_variational_gp.md
[link19]: 09_variational_gp.md
[link20]: ../theory/10_preference_learning.md
[link21]: 10_preference_learning.md
[link22]: ../theory/11_structured_output.md
[link23]: 11_structured_output.md
[link24]: ../theory/12_hierarchical_contextual_gp.md
[link25]: 12_hierarchical_contextual_gp.md
[link26]: ../theory/13_model_selection.md
[link27]: 13_model_selection.md
[link28]: ../theory/14_robust_gaussian_process.md
[link29]: 14_robust_gaussian_process.md
[link30]: ../theory/15_heteroskedastic_noise.md
[link31]: 15_heteroskedastic_noise.md
[link32]: ../theory/16_uncertain_input_gp.md
[link33]: 16_uncertain_input_gp.md
[link34]: ../theory/17_nonstationary_gp.md
[link35]: 17_nonstationary_gp.md
[link36]: ../theory/18_dimensionality_reduction_gp.md
[link37]: 18_dimensionality_reduction_gp.md
[link38]: ../theory/19_neural_reduction_gp.md
[link39]: 19_neural_reduction_gp.md
[link40]: ../theory/20_high_dimensional_search.md
[link41]: 20_high_dimensional_search.md
[link42]: ../theory/21_advanced_high_dimensional_models.md
[link43]: 21_advanced_high_dimensional_models.md
[link44]: ../theory/22_expressive_gp.md
[link45]: 22_expressive_gp.md
[link46]: ../theory/23_non_gp_surrogates.md
[link47]: 23_non_gp_surrogates.md
[link48]: ../theory/24_classification.md
[link49]: 24_classification.md
[link50]: ../theory/acquisition/01_foundations.md
[link51]: acquisition/01_foundations.md
[link52]: ../theory/acquisition/02_improvement.md
[link53]: acquisition/02_improvement.md
[link54]: ../theory/acquisition/03_confidence_bound.md
[link55]: acquisition/03_confidence_bound.md
[link56]: ../theory/acquisition/04_batch_noisy.md
[link57]: acquisition/04_batch_noisy.md
[link58]: ../theory/acquisition/05_information_theoretic.md
[link59]: acquisition/05_information_theoretic.md
[link60]: ../theory/acquisition/06_lookahead.md
[link61]: acquisition/06_lookahead.md
[link62]: ../theory/acquisition/07_multiobjective.md
[link63]: acquisition/07_multiobjective.md
[link64]: ../theory/acquisition/08_constraints.md
[link65]: acquisition/08_constraints.md
[link66]: ../theory/acquisition/09_active_learning.md
[link67]: acquisition/09_active_learning.md
[link68]: ../theory/acquisition/10_level_set.md
[link69]: acquisition/10_level_set.md
[link70]: ../theory/acquisition/11_multifidelity_cost_aware.md
[link71]: acquisition/11_multifidelity_cost_aware.md
[link72]: ../theory/acquisition/12_selection_guide.md
[link73]: acquisition/12_selection_guide.md
[link74]: ../theory/optimization/01_foundations.md
[link75]: optimization/01_foundations.md
[link76]: ../theory/optimization/02_gradient_based.md
[link77]: ../theory/optimization/03_initialization.md
[link78]: optimization/03_initialization.md
[link79]: ../theory/optimization/04_mixed_discrete.md
[link80]: optimization/04_mixed_discrete.md
[link81]: ../theory/optimization/05_constraints.md
[link82]: optimization/05_constraints.md
[link83]: ../theory/optimization/06_derivative_free.md
[link84]: optimization/06_derivative_free.md
[link85]: ../theory/optimization/07_evolutionary.md
[link86]: optimization/07_evolutionary.md
[link87]: ../theory/optimization/08_batch_one_shot.md
[link88]: optimization/08_batch_one_shot.md
[link89]: ../theory/optimization/09_high_dimensional.md
[link90]: optimization/09_high_dimensional.md
[link91]: ../theory/optimization/10_trust_region.md
[link92]: optimization/10_trust_region.md
[link93]: ../theory/optimization/11_hybrid.md
[link94]: optimization/11_hybrid.md
[link95]: ../theory/optimization/12_selection_guide.md
[link96]: optimization/12_selection_guide.md

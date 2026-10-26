# 理論と実装をつなぐ日本語技術ガイド — 章構成

> 現在は**章構成と執筆契約の確定段階**です。以下の `guide/` 配下の各章は配置予定であり、未作成の章へのリンクは置いていません。実装済み・検証済みと推定しないでください。

## このガイドの目的

[理論ガイド](../../../theory/README.md) の数式・仮定を出発点に、robotorchanの実装箇所、BoTorch等への委譲、公開API、利用上の制約、検証の根拠までを日本語で追えるようにします。既存の [Mapping原文](../README.md) と [モデル](../models.md)・[獲得関数](../acquisition.md)・[最適化](../optimization.md)・[検証](../verification.md) は移行完了まで保持します。

## 章構成：理論の24章と1対1対応

以下の `guide/<filename>` は**予定パス**です。章番号とbasenameを理論側に合わせます。内容は単純な翻訳ではなく、理論から実装へ進むための解説です。

| 理論の正本 | 日本語実装ガイドの予定パス |
| --- | --- |
| [01_bayesian_optimization.md](../../../theory/01_bayesian_optimization.md) | `guide/01_bayesian_optimization.md` |
| [02_gaussian_process.md](../../../theory/02_gaussian_process.md) | `guide/02_gaussian_process.md` |
| [03_kernel.md](../../../theory/03_kernel.md) | `guide/03_kernel.md` |
| [04_acquisition_function.md](../../../theory/04_acquisition_function.md) | `guide/04_acquisition_function.md` |
| [05_mixed_variables.md](../../../theory/05_mixed_variables.md) | `guide/05_mixed_variables.md` |
| [06_multi_fidelity.md](../../../theory/06_multi_fidelity.md) | `guide/06_multi_fidelity.md` |
| [07_multitask_multioutput.md](../../../theory/07_multitask_multioutput.md) | `guide/07_multitask_multioutput.md` |
| [08_high_dimensional_gp.md](../../../theory/08_high_dimensional_gp.md) | `guide/08_high_dimensional_gp.md` |
| [09_variational_gp.md](../../../theory/09_variational_gp.md) | `guide/09_variational_gp.md` |
| [10_preference_learning.md](../../../theory/10_preference_learning.md) | `guide/10_preference_learning.md` |
| [11_structured_output.md](../../../theory/11_structured_output.md) | `guide/11_structured_output.md` |
| [12_hierarchical_contextual_gp.md](../../../theory/12_hierarchical_contextual_gp.md) | `guide/12_hierarchical_contextual_gp.md` |
| [13_model_selection.md](../../../theory/13_model_selection.md) | `guide/13_model_selection.md` |
| [14_robust_gaussian_process.md](../../../theory/14_robust_gaussian_process.md) | `guide/14_robust_gaussian_process.md` |
| [15_heteroskedastic_noise.md](../../../theory/15_heteroskedastic_noise.md) | `guide/15_heteroskedastic_noise.md` |
| [16_uncertain_input_gp.md](../../../theory/16_uncertain_input_gp.md) | `guide/16_uncertain_input_gp.md` |
| [17_nonstationary_gp.md](../../../theory/17_nonstationary_gp.md) | `guide/17_nonstationary_gp.md` |
| [18_dimensionality_reduction_gp.md](../../../theory/18_dimensionality_reduction_gp.md) | `guide/18_dimensionality_reduction_gp.md` |
| [19_neural_reduction_gp.md](../../../theory/19_neural_reduction_gp.md) | `guide/19_neural_reduction_gp.md` |
| [20_high_dimensional_search.md](../../../theory/20_high_dimensional_search.md) | `guide/20_high_dimensional_search.md` |
| [21_advanced_high_dimensional_models.md](../../../theory/21_advanced_high_dimensional_models.md) | `guide/21_advanced_high_dimensional_models.md` |
| [22_expressive_gp.md](../../../theory/22_expressive_gp.md) | `guide/22_expressive_gp.md` |
| [23_non_gp_surrogates.md](../../../theory/23_non_gp_surrogates.md) | `guide/23_non_gp_surrogates.md` |
| [24_classification.md](../../../theory/24_classification.md) | `guide/24_classification.md` |

## 獲得関数：理論12サブ章と1対1対応

| 理論の正本 | ガイドの予定パス |
| --- | --- |
| [acquisition/01_foundations.md](../../../theory/acquisition/01_foundations.md) | `guide/acquisition/01_foundations.md` |
| [acquisition/02_improvement.md](../../../theory/acquisition/02_improvement.md) | `guide/acquisition/02_improvement.md` |
| [acquisition/03_confidence_bound.md](../../../theory/acquisition/03_confidence_bound.md) | `guide/acquisition/03_confidence_bound.md` |
| [acquisition/04_batch_noisy.md](../../../theory/acquisition/04_batch_noisy.md) | `guide/acquisition/04_batch_noisy.md` |
| [acquisition/05_information_theoretic.md](../../../theory/acquisition/05_information_theoretic.md) | `guide/acquisition/05_information_theoretic.md` |
| [acquisition/06_lookahead.md](../../../theory/acquisition/06_lookahead.md) | `guide/acquisition/06_lookahead.md` |
| [acquisition/07_multiobjective.md](../../../theory/acquisition/07_multiobjective.md) | `guide/acquisition/07_multiobjective.md` |
| [acquisition/08_constraints.md](../../../theory/acquisition/08_constraints.md) | `guide/acquisition/08_constraints.md` |
| [acquisition/09_active_learning.md](../../../theory/acquisition/09_active_learning.md) | `guide/acquisition/09_active_learning.md` |
| [acquisition/10_level_set.md](../../../theory/acquisition/10_level_set.md) | `guide/acquisition/10_level_set.md` |
| [acquisition/11_multifidelity_cost_aware.md](../../../theory/acquisition/11_multifidelity_cost_aware.md) | `guide/acquisition/11_multifidelity_cost_aware.md` |
| [acquisition/12_selection_guide.md](../../../theory/acquisition/12_selection_guide.md) | `guide/acquisition/12_selection_guide.md` |

## 獲得関数の最適化：理論12サブ章と1対1対応

| 理論の正本 | ガイドの予定パス |
| --- | --- |
| [optimization/01_foundations.md](../../../theory/optimization/01_foundations.md) | `guide/optimization/01_foundations.md` |
| [optimization/02_gradient_based.md](../../../theory/optimization/02_gradient_based.md) | `guide/optimization/02_gradient_based.md` |
| [optimization/03_initialization.md](../../../theory/optimization/03_initialization.md) | `guide/optimization/03_initialization.md` |
| [optimization/04_mixed_discrete.md](../../../theory/optimization/04_mixed_discrete.md) | `guide/optimization/04_mixed_discrete.md` |
| [optimization/05_constraints.md](../../../theory/optimization/05_constraints.md) | `guide/optimization/05_constraints.md` |
| [optimization/06_derivative_free.md](../../../theory/optimization/06_derivative_free.md) | `guide/optimization/06_derivative_free.md` |
| [optimization/07_evolutionary.md](../../../theory/optimization/07_evolutionary.md) | `guide/optimization/07_evolutionary.md` |
| [optimization/08_batch_one_shot.md](../../../theory/optimization/08_batch_one_shot.md) | `guide/optimization/08_batch_one_shot.md` |
| [optimization/09_high_dimensional.md](../../../theory/optimization/09_high_dimensional.md) | `guide/optimization/09_high_dimensional.md` |
| [optimization/10_trust_region.md](../../../theory/optimization/10_trust_region.md) | `guide/optimization/10_trust_region.md` |
| [optimization/11_hybrid.md](../../../theory/optimization/11_hybrid.md) | `guide/optimization/11_hybrid.md` |
| [optimization/12_selection_guide.md](../../../theory/optimization/12_selection_guide.md) | `guide/optimization/12_selection_guide.md` |

## 各章の執筆契約

1. **この章で解決する問題**：理論上の対象・仮定・入出力を簡潔に示す。理論の数式を無批判に複製せず、理論章の該当節へ導く。
2. **アルゴリズムからコードまで**：数学的対象→アルゴリズム→公開API→実装ファイル/シンボル→BoTorch等への委譲を、実際にソースを確認して記す。
3. **APIとデータ契約**：fit、posterior、predict_proba、q/batch、dtype/device、制約、探索座標などを、当該章に必要な範囲で説明する。
4. **実装関係のラベル**：`Implemented` / `Delegated` / `Extended` / `Approximated` / `Unsupported` / `Unverified` を使い分ける。理論章の存在は実装済みの証拠ではない。
5. **検証と再現**：対象のproperty、テストのassertion、許容誤差、Notebook、未検証事項を明記する。存在確認と数値検証を混同しない。
6. **学習の流れ**：直感→理論との対応→処理の流れ→最小利用例→適用限界→参照先の順に読めるようにする。
7. **横断事項の分離**：registry全件表、API差分、共通制約、委譲・互換性、検証限界はAppendixで保持し、各章から参照する。PR/CI/Phaseの履歴は本文に混ぜず `.github/project-records/` に保管する。

## 横断的な補助文書（予定）

- `guide/appendix/README.md`：Appendix索引。
- `guide/appendix/model-registry-and-public-api.md`：モデル登録・公開API・分類registry差分。
- `guide/appendix/model-composition-and-delegation.md`：複合モデルと上流委譲。
- `guide/appendix/acquisition-contracts-and-compatibility.md`：獲得関数の互換性と合成契約。
- `guide/appendix/optimizer-dispatch-and-constraints.md`：optimizer dispatch、candidate/outcome制約の責務。
- `guide/appendix/cross-cutting-implementation-contracts.md`：dtype、batch、fit、posterior等。
- `guide/appendix/limitations-and-unverified.md`：未対応・未検証の横断一覧。
- `guide/verification/README.md`：数値参照とproperty別検証の入口。

## 既存文書との責務境界

- `docs/theory/`：数学的定義・仮定・導出の正本。
- `docs/models/`、`docs/optimization/`：実際の使い方と選択の正本。
- `docs/development/theory-implementation/` の旧5 Mapping：保全対象の移行元。既存のリンク・anchor・CI契約を壊さない。
- `guide/`：**理論から実装を理解するための日本語の説明**。原文の技術的主張は根拠を確認して移す。
- `.github/project-records/theory-guide/`：作業の監査証拠と履歴。

## 移行・公開の判定

24本の主章と獲得関数・最適化各12本の計48章は**構成の予定**であり、実装や執筆の完了件数ではありません。移行台帳の各原文区間についてprimary/secondary移行先、数式・API・制約・negative contract・テスト証拠を照合し、`verified`になるまで旧Mappingから削除しません。新章は内容が整った単位で公開し、未作成の章への壊れたリンクを作りません。

**Phase 2の対象**はこの章体系、配置、責務境界の確定です。本文の網羅的な移行・意味論検証は後続の固定18 Phase内で実施します。

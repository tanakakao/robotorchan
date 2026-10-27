# モデル24章：掲載例と理論式の検証対応表

本表は日本語の理論・実装ガイド第1～24章に対し、**この開発サイクルで
直接確認した証拠のみ**を整理する。テストが空欄でも、他のテストが存在しないことを意味しない。
各章の理解と利用は[ガイド目次](../../../docs/development/theory-implementation/guide/README.md)
から始め、過去の変更履歴は本監査記録を参照する。

## 判定ルール

- **掲載例実行欄**：Markdownからコードを抽出するテストへの参照。各ジョブの実行結果・skipとは別。
- **独立数値照合**：理論式から別経路で計算した参照値と実装を比較するテストがある。
- **—**：この表では対応する証拠をまだ確認していない（未実装・未検証の断定ではない）。
- 形状確認、有限値確認、独立数値照合、学習収束、最適化性能は異なる証拠である。

## モデル24章の現時点の証拠

| 章 | 主題 | 掲載例実行 | 独立数値照合 |
| --- | --- | --- | --- |
| 01 | Bayesian Optimization | bo_acquisition_examples | — |
| 02 | Gaussian Process | core_model_examples | — |
| 03 | Kernel | kernel_preference_examples | — |
| 04 | Acquisition Function | bo_acquisition_examples | — |
| 05 | Mixed Variables | core_model_examples | — |
| 06 | Multi-Fidelity | structured_model_examples | — |
| 07 | MultiTask / MultiOutput | structured_model_examples, heterogeneous_example | — |
| 08 | High-Dimensional GP | saas_lightweight_example（optional依存なしではskip） | — |
| 09 | Variational GP | core_model_examples | — |
| 10 | Preference Learning | kernel_preference_examples | — |
| 11 | Structured Output | structured_model_examples | — |
| 12 | Hierarchical / Contextual | hierarchical_example, contextual_example | — |
| 13 | Model Selection | —（案内・比較章） | — |
| 14 | Robust GP | robust_model_example, contaminated_example | — |
| 15 | Heteroskedastic Noise | heteroskedastic_example | — |
| 16 | Uncertain Input GP | kernel_model_examples | gaussian_input_kernel_reference |
| 17 | Nonstationary GP | kernel_model_examples | gibbs_reference |
| 18 | Dimensionality Reduction GP | surrogate_examples | — |
| 19 | Neural Reduction GP | joint_encoder_examples | — |
| 20 | High-Dimensional Search | high_dimensional_examples | — |
| 21 | Advanced High-Dimensional | high_dimensional_examples | — |
| 22 | Expressive GP | joint_encoder_examples | — |
| 23 | Non-GP Surrogates | surrogate_examples | — |
| 24 | Classification | surrogate_examples | — |

## 第13章の別種の証拠

第13章は個別モデルの掲載コードではなく、モデル選択の比較表と関連章への案内を担う。
モデル別MLL・posteriorの学習経路を第7節に追記し、SAASへの参照先を第08章に修正した。
リンク検査のCI結果は別途確認し、数値テストの代わりに「実行成功」とは記録しない。

## 証拠の場所

掲載例実行の各識別子は、すべて `tests/docs/test_theory_guide_*.py` のファイル名
末尾（`.py`を除く）を示す。独立数値照合の識別子は次の実ファイルに対応する。

- `gibbs_reference`：
  `tests/models/robust/test_theory_guide_gibbs_reference.py`
- `gaussian_input_kernel_reference`：
  `tests/models/uncertain/test_theory_guide_gaussian_input_kernel_reference.py`

この表はテストファイルの存在と対象の対応を示すものであり、
すべての掲載コードブロックを実行したという意味ではない。
CI全体が成功していても、skipされた個別テストの実行成功を意味しない。
特に第08章のSAAS軽量例はNumPyroのない通常CI環境ではskipされる。
そのため、optional依存を揃えたテストジョブの成功証拠を別途確認する必要がある。
テスト実行成功も数値計算の普遍的な正しさの証明ではない。

## モデル24章のゲート判定

- **構造・索引：達成。** 固定24章と対応する理論章へのリンクを既存の
  `scripts/check_theory_guide_quality.py`・リンクCIで検査する。
- **掲載例のテスト参照：23章に登録。** 第13章は比較・ナビゲーション中心のため別基準。
  登録数は全掲載ブロック数や実際にpassした数を表さない。
- **条件付き：第08章。** NumPyro等のoptional依存がなければSAASテストはskipされる。
  通常CIのsuccessだけを根拠に実行成功と認定しない。
- **独立数値照合：第16・17章の2章のみ、この表に直接証拠を登録。**
  他章の数式まで検証済みとはしない。
- **最終完了：保留。** 未照合の旧Mapping見出し・数式・表・negative contract、
  全APIと学習収束・数値property・性能試験を含む完全検収は終了していない。

## 引き継ぐ未完了作業

第12章のSACGP・LCEAGPなど、contextual familyの全面実行検証は対象外。
第08章は依存関係を満たしたCIジョブの個別passを確認する。
次は獲得関数12章・最適化12章について、掲載コードの直接実行と
数式/実装の対応を監査する。旧Mappingからの174見出し区間移行は
独立ゲートとして残し、`verified`を推測で更新しない。

旧Mapping5本と174見出しの未完了監査は、この表をもってverifiedに変更しない。
固定18 Phase・48章の構成は維持する。

# モデル24章：掲載例と理論式の検証対応表

本表は日本語の理論・実装ガイド第1～24章に対し、**この開発サイクルで
直接確認した証拠のみ**を整理する。テストが空欄でも、他のテストが存在しないことを意味しない。
各章の理解と利用は[ガイド目次](../../../docs/development/theory-implementation/guide/README.md)
から始め、過去の変更履歴は本監査記録を参照する。

## 判定ルール

- **掲載例実行**：MarkdownからPythonコードを直接抽出してCIで実行するテストがある。
- **独立数値照合**：理論式から別経路で計算した参照値と実装を比較するテストがある。
- **—**：この表では対応する証拠をまだ確認していない（未実装・未検証の断定ではない）。
- 形状確認、有限値確認、独立数値照合、学習収束、最適化性能は異なる証拠である。

## モデル24章の現時点の証拠

| 章 | 主題 | 掲載例実行 | 独立数値照合 |
| --- | --- | --- | --- |
| 01 | Bayesian Optimization | — | — |
| 02 | Gaussian Process | core_model_examples | — |
| 03 | Kernel | kernel_preference_examples | — |
| 04 | Acquisition Function | — | — |
| 05 | Mixed Variables | core_model_examples | — |
| 06 | Multi-Fidelity | structured_model_examples | — |
| 07 | MultiTask / MultiOutput | structured_model_examples, heterogeneous_example | — |
| 08 | High-Dimensional GP | — | — |
| 09 | Variational GP | core_model_examples | — |
| 10 | Preference Learning | kernel_preference_examples | — |
| 11 | Structured Output | structured_model_examples | — |
| 12 | Hierarchical / Contextual | — | — |
| 13 | Model Selection | — | — |
| 14 | Robust GP | robust_model_example, contaminated_example | — |
| 15 | Heteroskedastic Noise | — | — |
| 16 | Uncertain Input GP | kernel_model_examples | gaussian_input_kernel_reference |
| 17 | Nonstationary GP | kernel_model_examples | gibbs_reference |
| 18 | Dimensionality Reduction GP | — | — |
| 19 | Neural Reduction GP | — | — |
| 20 | High-Dimensional Search | — | — |
| 21 | Advanced High-Dimensional | — | — |
| 22 | Expressive GP | — | — |
| 23 | Non-GP Surrogates | — | — |
| 24 | Classification | — | — |

## 証拠の場所

掲載例実行の各識別子は、すべて `tests/docs/test_theory_guide_*.py` のファイル名
末尾（`.py`を除く）を示す。独立数値照合の識別子は次の実ファイルに対応する。

- `gibbs_reference`：
  `tests/models/robust/test_theory_guide_gibbs_reference.py`
- `gaussian_input_kernel_reference`：
  `tests/models/uncertain/test_theory_guide_gaussian_input_kernel_reference.py`

この表はテストファイルの存在と対象の対応を示すものであり、
すべての掲載コードブロックを実行したという意味ではない。
CI成功は記録したコミットのテスト実行成功を示すが、数値計算の普遍的な正しさの証明ではない。

## 以降の優先順位

次は掲載例実行の証拠が未登録の第12章Hierarchical / Contextualと
第15章Heteroskedastic Noiseを確認する。
次に第08・18～24章の高次元・非GP・分類関連を確認する。
学習済みposteriorの独立参照や獲得関数・最適化の統合検証は、
単純なconstructor実行テストと切り分けて整備する。

旧Mapping5本と174見出しの未完了監査は、この表をもってverifiedに変更しない。
固定18 Phase・48章の構成は維持する。

# Model registryと公開API：回帰・分類の境界

このAppendixは[旧Model Mapping](../../../models.md)の「Registry boundaries」「Regression model inventory by existing guide」「Binary classification registry inventory」（旧原文の行18–91、および分類ID表）を読者向けに対応付ける。**静的inventoryの出典を保持するための補助説明**であり、元文書の全レコードや検証証拠の完全移行を意味しない。

## 1. Registry IDとpublic classは異なる集合

- 旧Mappingに記録された監査時点の回帰向けモデルregistryは**142 entries**。全公開classの集合とは限らない。
- 二値分類registryは**29 model IDs**（`BINARY_CLASSIFICATION_MODEL_SPECS`）。回帰クラス名と命名規約が異なり、その包含関係を仮定しない。
- `HeterogeneousModel`は回帰向けcoverage generatorのpublic modelスコープ外にあり、heterogeneous adapter層で扱う。
- 分類packageの公開exportにはcalibration、conformal、likelihood、posterior、reliability helperも含まれ、すべてをモデルregistry entryとして数えない。

この数字は**旧監査時点の集計**。現在のmainに登録されているモデル数の新規独立集計はこの補助文書では行っていない。現行のmachine-readable coverageは[model_coverage.json](../../../../model_coverage.json)、生成規則は`scripts/generate_model_coverage.py`を正本とする。

## 2. 回帰registryを利用ガイドのfamilyへ振り分ける

以下は旧Mappingのファミリー別**監査時点の集計**を、公開利用ガイドへの導線に整理したもの。各registry名・モデル固有のtheory/notebookリンクは旧Mappingおよびgenerated coverageを参照する。

| 利用ガイド | 旧監査時点のregistry件数 | 選択する際の分岐 |
| --- | ---: | --- |
| [Expressive](../../../../models/expressive.md) | 16 | DeepGP、NNGP、spectral-mixture |
| [Hierarchical / Contextual](../../../../models/hierarchical_contextual.md) | 6 | 階層・条件付き・context decomposition |
| [High-dimensional](../../../../models/high_dimensional.md) | 12 | SAAS・additive・ALEBO surrogate |
| [Multi-task / Multi-output](../../../../models/multitask_multioutput.md) | 10 | 情報共有とtask設計 |
| [Non-GP](../../../../models/non_gp.md) | 5 | Tree ensemble・boosting |
| [Preference](../../../../models/preference.md) | 1 | 対比較の観測 |
| [Reduced](../../../../models/reduced.md) | 52 | PCA/PLS/random projection/neural reducer |
| [Robust / Noise](../../../../models/robust_noise.md) | 27 | 外れ値・位置依存noise・非定常 |
| [Standard](../../../../models/standard.md) | 6 | Exact / Mixed / Fidelity / Variational |
| [Structured Output](../../../../models/structured_output.md) | 4 | tensor、output-coordinate |
| [Uncertain Input](../../../../models/uncertain_input.md) | 3 | 不確かな連続・カテゴリ入力 |

**注意：** この一覧は旧原文の11行を保全して解釈するもので、個々のモデルの数学的検証を済ませたという意味ではない。また原文のリスト全件は本Appendixへ転記していないため、元Mappingの行32–91は`verified migration`ではなく**partial / review required**とする。

## 3. 分類registryの境界

分類モデルのIDは `binary.standard`、`binary.mixed`、`binary.multitask`、`binary.kronecker_multitask`、`binary.deep_gp`、`binary.ensemble.gp`など、役割を含む文字列である。これは`BinarySingleTaskGPClassifier`などの公開Pythonクラス名と一致させるためのaliasではない。特に分類のKronecker表記をExact Kronecker GPと誤解してはならない。

各分類IDの静的ソース照合やassertion対象は旧MappingのID別表で確認する。[分類利用ガイド](../../../../models/classification.md)と[分類理論→実装](../24_classification.md)は別に参照する。

## 4. Source-of-truthと検証水準

| 対象 | 正本 | このAppendixでの扱い |
| --- | --- | --- |
| regression model registry | `src/robotorchan/models/registry.py` | 旧監査件数とfamilyを説明 |
| binary classification registry | `src/robotorchan/models/classification/registry.py` | IDとpublic exportを混同しない |
| 生成coverage | `scripts/generate_model_coverage.py`と`docs/model_coverage.json` | 手動編集・再集計しない |
| 全件静的監査 | `scripts/audit_model_api.py` | ソース存在と参照の検査、数学的同値性ではない |
| 数学的検証 | [Verification Mapping](../../../verification.md) | 一つのpropertyごとの数値参照のみをVerifiedとする |

関連：[Model Selection](../13_model_selection.md)、[検証索引](../verification/models.md)、[原文](../../../models.md)。

この文書は、旧原文の`Registry boundaries`と11 familyの導線を再配置した**部分的レビュー**。29分類IDの全行、142回帰名、H4・数式・negative contracts、最新mainとの全件再照合が終わるまでは元文書を削除しない。

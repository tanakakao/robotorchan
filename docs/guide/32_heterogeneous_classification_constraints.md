# 異種回帰・分類モデルによる制約付き候補提案

本ガイドは[理論第32章](../theory/32_heterogeneous_classification_constraints.md)の単目的qEI・多目的qEHVIと分類PoFの組み合わせを、既存の公開実装へ対応付けます。実際のモデル生成・学習・候補提案は[Notebook 61](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)に集約しています。

## モデル構造と役割

連続値の回帰GPと、離散ラベルの分類GPは別々の予測意味論を持ちます。`HeterogeneousModel` は各モデルを登録し、グローバル出力名から所有モデルを解決します。分類モデルをBoTorchの回帰事後分布へ無理に連結するのではありません。

| 役割 | 公開API |
| --- | --- |
| 単目的／多目的回帰GP | `SingleTaskGP` |
| 3クラス以上の分類モデル | `MulticlassSingleTaskGPClassifier` |
| 順序付き3クラス以上の分類モデル | `OrdinalSingleTaskGPClassifier` |
| 回帰・分類モデルの登録 | `HeterogeneousModel` |
| 回帰目的の方向と出力名 | `RegressionObjective` |
| 許容クラス集合 | `ClassificationConstraint(feasible_classes=(...))` |
| 目的と制約の宣言 | `ProblemSemantics` |
| 単目的の獲得関数 | `make_classification_constrained_qei_acquisition` |
| 多目的の獲得関数 | `make_constrained_qehvi_acquisition` |
| 多目的の可行Pareto参照集合 | `NondominatedPartitioning`（BoTorch） |
| 候補提案 | `robotorchan.optim.optimize_acqf` |

**クラス集合は出力ラベルの条件**です。分類器の入力カテゴリ（Mixed変数）の指定とは異なり、例えば `feasible_classes=(1, 2)` はクラス1と2の確率の和を意味します。

## 単目的回帰 × 多クラス／順序分類

1. 回帰GPをモデル化して `fit()` で学習する。分類GPも `fit()` で学習する。
2. `HeterogeneousModel` に回帰出力名と分類出力名を与える。
3. `ProblemSemantics` に1個の `RegressionObjective` と少なくとも1個の `ClassificationConstraint` を登録する。
4. `make_classification_constrained_qei_acquisition` が回帰GPの `qExpectedImprovement` と分類GPのPoFを組み合わせる。
5. `optimize_acqf` で獲得関数を最適化し、提案入力と獲得値を受け取る。

`best_f` は、品質ラベルが既知であるなら**可行と判定した既存観測の最良目的値**を指定します。不適合な既存観測の最高値を含めると改善量の参照基準を誤る可能性があります。

## 多目的回帰 × 多クラス／順序分類

1. **同じ回帰モデルが2個以上の目的出力を持つ**ように構成し、学習する。分類GPは別エントリでよい。
2. 複数個の `RegressionObjective` を宣言し、対象の目的出力を指定する。
3. 最大化／最小化の向きを統一した目的値で参照点 `ref_point` と `NondominatedPartitioning` を構築する。
4. 観測された分類ラベルで可行と分かる点だけをPareto参照集合に含める。
5. `make_constrained_qehvi_acquisition` がBoTorchの `qExpectedHypervolumeImprovement` と分類GPのPoF重み付けを組み合わせる。
6. `optimize_acqf` がqEHVIベースの獲得関数を最適化し、候補を提案する。

`qEHVI` の参照点は目的空間で定義されるため、入力探索範囲 `bounds` とは別物です。既知の可行観測点がない場合に同じ例の参照集合をそのまま作れるとは限りません。参照点と初期化方針の設計が必要です。

**複数目的の回帰出力を複数の異種モデルエントリに分ける方法**は、上記の多目的ブリッジの対象外です。異種モデルに分類器が加わっていることと、複数の回帰モデル間の結合事後を持つことは別です。

## 分類確率を重み付けする仕組み

`ProblemSemantics` から分類出力を特定し、`ClassificationProbabilityOfFeasibility` が `predict_proba(X)` のクラス確率のうち許容集合の列を合計します。`FeasibilityWeightedAcquisition` が既存の回帰獲得関数値にPoFを掛け合わせます。

- `q_reduction="product"` は候補ごとの周辺PoFの積を使います。q>1での厳密な同時実行可能確率ではありません。
- `q_reduction="minimum"` は候補間の最小PoFをスコアに使います。積より一般に大きくなり、真の同時PoFとは異なります。
- 複数の分類制約間でも独立性の近似が入ります。
- qEI・qEHVI側の候補間のGP事後相関と、PoF重み付け側の独立性近似は別に考えます。

## マルチクラス分類のMonte Carlo評価と再現性

`MulticlassSingleTaskGPClassifier.predict_proba(X)` は潜在GPの事後サンプルをSoftmax変換したMonte Carlo平均です。そのため、**同じ候補を再評価しても**有限サンプルによる確率の揺れがあり、重み付け後の獲得関数値も変わり得ます。これは回帰側qEI/qEHVIの `SobolQMCNormalSampler` だけにseedを設定しても解消されません。

一方、`OrdinalSingleTaskGPClassifier.predict_proba(X)` はGaussian潜在事後とOrdered-probitの下で周辺予測確率を解析的に計算します。両者の計算上のばらつきを区別してください。

この違いは、特に獲得関数を反復評価する最適化器で注意が必要です。候補の安定性を評価するときは乱数シード・Monte Carloサンプル数と複数回評価を確認し、**単一回の候補提案が再現性や確率校正の保証にはならない**ことを明記します。これは制約付きBOの理論上の独立性近似とは別の、有限サンプルによる数値誤差です。

## 注意点と適用範囲

- `probability_threshold` を指定した確率下限は、ここで紹介するPoF重み付けにより強制されません。確率重み付けと閾値制約は別物です。
- `qEHVI` は既知のPareto参照集合を使います。観測ノイズ下でのqNEHVIと同じものではありません。
- `MixedVariableSpace` による入力の整数・カテゴリなどの変数制約は、分類の許容クラス集合とは区別します。対応Optimizerとモデルを確認してください。
- `X_pending` と異種モデルのjoint fantasizationは本合成では対象外です。
- `predict_proba(X)` の予測確率の校正はモデル・データに依存します。Notebookの少数epoch学習は実装例であり、統計性能の保証ではありません。

## ソースの対応

- `src/robotorchan/models/heterogeneous.py`：異種モデルの出力名と所有モデルの管理
- `src/robotorchan/semantics/constraints.py`：許容クラス集合の指定
- `src/robotorchan/semantics/probability.py`：クラス集合のPoF計算
- `src/robotorchan/acquisition/composition.py`：回帰qEI／多目的qEHVIと分類重み付けの合成
- `src/robotorchan/acquisition/classification_constraints.py`：FeasibilityWeightedAcquisition

## 関連資料

- [マルチクラスGP分類のTheory／Guide](../theory/30_multiclass_classification.md) / [Guide](30_multiclass_classification.md)
- [順序GP分類のTheory／Guide](../theory/31_ordinal_classification.md) / [Guide](31_ordinal_classification.md)
- [Notebook 61：単目的・多目的回帰×分類制約](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)
- [候補制約のガイド](optimization/05_constraints.md)
- [Mixed候補最適化](optimization/04_mixed_discrete.md)

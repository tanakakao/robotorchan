# 30. マルチクラスGP分類：理論と実装の対応

数学的な背景は[理論30章](../theory/30_multiclass_classification.md)を参照してください。モデルの構築・学習・不確実性評価・候補提案は[Notebook 30](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)で実行できます。

## クラス確率と潜在GPの事後分布

`posterior(X)` は潜在GPの事後分布、`predict_proba(X)` は観測ラベルの事後予測確率、`probability_posterior(X)` はクラス確率の不確実性を表します。これらは別の量です。

`probability_posterior(X)` の `mean` / `variance` / `credible_interval` は**クラスごとの周辺的な量**であり、全クラスが同時に指定された信用区間に入るという意味ではありません。候補間の依存関係を捨てない評価では、同時潜在サンプルの扱いを確認します。

## Softmaxに基づく分類確率Posterior

マルチクラスGPはSoftmax変換後の確率を、BoTorchの `SobolQMCNormalSampler` を用いた潜在GPの同時事後サンプルから推定します。平均・分散とも同一サンプルを使い、信用区間はクラスごとの周辺分位点です。

`predict_proba(X)` のモンテカルロ評価と `probability_posterior(X)` のQMC推定は、同じ確率を目標にしていても有限サンプルでは一致しないことがあります。

## マルチクラスGPの公開APIと獲得関数

`MulticlassSingleTaskGPClassifier` は3クラス以上の分類に対応します。
カテゴリ入力を含む場合は `MixedMulticlassSingleTaskGPClassifier` に
`cat_dims` を指定します。ラベルは `0, ..., C-1` の整数です。

`posterior(X)` はクラスごとの潜在GP分布であり、クラス確率ではありません。
`predict_proba(X)` は形状 `[..., q, C]` の事後予測確率を返します。
分類確率の最後の軸は合計1になります。

Active Learningでは既存の `PredictiveEntropy`、`MarginUncertainty`、
`ProbabilityVariance`、`BALD` と各q版を共通利用できます。
`qBALD` はラベルの組合せ数 `C**q <= 256` に制限されます。
`LatentStraddle` は二値分類専用であり、マルチクラスには使えません。

[30番の実行Notebook](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)と
[対応表](../models/multiclass-active-learning.md)も参照してください。


### はじめてのマルチクラス分類：入力と出力

たとえば設備の状態を「正常・注意・異常」に分けるとき、
ラベルはそれぞれ `0, 1, 2` とします。
入力 `train_X` は `[観測数, 特徴量数]`、ラベル `train_Y` は
`[観測数]` の整数テンソルです。

実行例では `MulticlassSingleTaskGPClassifier` に `num_classes=3` を指定し、
`fit()` の後に `predict_proba()` を呼び出します。
3点を予測した場合の出力形状は `[3, 3]`（3点 × 3クラス）です。
コードは[30番Notebook](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)にまとめています。

予測確率が `[0.7, 0.2, 0.1]` なら、
最も確からしいのはクラス0ですが、クラス1・2の可能性も残ります。
`predict_class` は確率が最大のクラスを選びます。
`posterior` の潜在値を直接「70%」などと解釈しないでください。

### 獲得関数を選ぶための考え方

| 方法 | 何を優先するか | 最初の選択として |
| --- | --- | --- |
| PredictiveEntropy | ラベル予測が曖昧な点 | シンプルな不確実性サンプリング |
| MarginUncertainty | 上位2クラスの確率が近い点 | クラス境界を重点的に探索 |
| ProbabilityVariance | GPサンプル間でクラス確率が変動する点 | モデル側の不確実性を調べる |
| BALD | 潜在GPについて情報を得られそうな点 | ノイズとモデル不確実性を区別したい場合 |
| qBALD | 複数候補を同時に測る際の情報量 | 小さなqで候補間の依存を考慮 |

いずれも「最も正しく分類できる点」を直接選ぶ関数ではありません。
**どの点を追加測定すると学習に役立ちそうか**を評価します。

### 二値分類から移行するときの注意

二値分類のBernoulliモデルでは成功確率1つでラベル分布を表せますが、
マルチクラスではクラス数 `C` 個の確率が必要です。
`LatentStraddle` の「潜在関数0が境界」という前提は
そのまま使えません。マルチクラスではクラス間の潜在値の差が境界を決めます。

Mixedモデルでは `cat_dims` を指定しますが、
探索側の `MixedVariableSpace` や `fixed_features_list` は別途設定します。
モデルをMixedにしただけでは、候補生成器がカテゴリ制約を自動的に理解するわけではありません。

### よくあるつまずき

- **確率の和が1にならない？** 最後のクラス軸で合計してください。
- **学習回数を増やせば必ず良くなる？** 過学習や不安定な最適化もあるため、検証データで確認してください。
- **Entropyが高いのにBALDが低い？** ラベル自体が曖昧でも、追加データでモデルが学習できる不確実性とは限りません。
- **qを増やしたらqBALDが失敗した？** 組合せ数 `C**q` の上限を確認してください。
- **Accuracyが高ければ十分？** クラス不均衡や確率校正を考慮し、混同行列・NLL・Brier scoreも確認してください。



## 回帰モデルとの合成

[61番Notebook](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)では、回帰GP（単目的・多目的）とマルチクラスGPを異種モデルとして構成し、許容クラス集合の確率で獲得関数を重み付けします。分類モデル単体のActive Learningと目的が異なるため、こちらに分離しています。

## 実装の参照

- `src/robotorchan/models/classification/multiclass/standard/single_task.py`
- `src/robotorchan/models/classification/posterior/probability.py`
- [二値分類の共通基礎ガイド](24_classification.md)


## 非定常・Spectral Mixtureカーネル

マルチクラス分類では各クラスの潜在GPの共分散を変更します。
既存の`MulticlassSingleTaskGPClassifier`を基礎として、
`NonstationaryMulticlassGPClassifier`と
`SpectralMixtureMulticlassGPClassifier`を提供します。
カテゴリ入力を含む場合は、それぞれ
`MixedNonstationaryMulticlassGPClassifier`、
`MixedSpectralMixtureMulticlassGPClassifier`を利用します。

非定常モデルは既存の`GibbsKernel`を使い、入力に応じた局所長さ尺度を学習します。
`local_lengthscale(X)`はクラスごとの値を返します。
Spectral Mixtureは複数周波数の定常共分散であり、非定常性と同義ではありません。
混合変数では連続列だけにこれらのカーネルを適用し、
カテゴリ列には既存の混合共分散構成を再利用します。

`fit()`、`predict_proba()`、分類用獲得関数の処理経路は
標準マルチクラスモデルと共通です。候補探索では
`cat_dims`だけでなく、候補側にも許容カテゴリ値を指定してください。
非定常カーネルの数学的背景は[非定常GP](../theory/17_nonstationary_gp.md)、
Spectral Mixtureの背景は[表現力の高いGP](../theory/22_expressive_gp.md)を参照してください。

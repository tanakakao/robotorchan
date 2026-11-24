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


## 高次元分類：PCA・PLS・Random Projection・ALEBO

この節では、入力を縮約するモデルとALEBOの座標系を区別します。
以下は**候補プールから予測エントロピーが最大の点を1点選ぶ**例です。
これは候補提案までの最小例であり、連続空間の獲得関数最適化ではありません。

**実行手順：** 訓練データ `train_X`（例：8次元）とクラスラベル `train_Y` を用意し、`n_components=3` でPCA分類モデルを構築します。`fit(epochs=10, lr=0.01)` で学習後、元の8次元の候補プールを `predict_proba(candidate_X)` に渡します。各候補のクラス確率から予測エントロピーを求め、最大の候補を選びます。学習・予測・候補提案を実際に実行するコードは、対応する分類Notebookにまとめます。

`PLSMulticlassGPClassifier` および
`RandomProjectionMulticlassGPClassifier` も
`n_components` を指定して同じ手順で利用できます。
ただし、PLSはラベルの数値表現に依存するため、特に多クラス分類では
クラス番号の置換に対する感度を検証してください。

ALEBOでは `ALEBOMulticlassGPClassifier` に
**埋め込み空間の学習座標と候補座標**を渡します。
元の探索空間の候補をそのまま渡さないでください。
埋め込みの構築、元の実験条件への射影、実行可能領域の処理は
`ALEBOStrategy` 側の責務です。

この例の学習ラベルはAPI説明用の人工データです。
モデル精度やALの改善を示すベンチマークではありません。
実験では保留データを分け、NLL・Brier score・クラス別指標を確認します。


## 高次元の疎ARDモデル：SAAS型とMAP-SAAS型

`SaasMulticlassGPClassifier` は `global_scale` を、
`MapSaasMulticlassGPClassifier` は `tau` を正の値として指定します。
どちらも入力は元の特徴量の次元のままで、PCA等の縮約を事前に要求しません。

使用の流れは次のとおりです。

1. 元の高次元入力 `train_X` とクラスラベル `train_Y` でモデルを構築する
2. `num_classes` を指定し、`model.fit(epochs=..., lr=...)` で学習する
3. 未測定の高次元候補を `model.predict_proba(candidate_X)` に渡す
4. 予測確率からエントロピーなどの獲得スコアを計算し、候補を選ぶ

これは**変分分類モデルに疎ARD事前分布を導入する実装**です。
完全Bayesian SAASのMCMC推論とは区別してください。
実行できる学習・予測・候補提案・可視化は
[30番Notebook](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)のPhase 6節を参照してください。


## タスク相関GP分類（MultiTask／Block形式）

`MultiTaskMulticlassGPClassifier` は、入力の1列を整数タスクIDとする**Long形式**に対応します。
`task_feature` でその列を、`rank` でICMタスク共分散のランクを指定します。
各行の `train_Y` は1つのクラス番号です。
未知のタスク番号での予測は学習時のタスク索引範囲を超えるため対象にしません。

`KroneckerMultiTaskMulticlassGPClassifier` は共有入力と、全タスクのラベルを並べた
`train_Y`（形状 `[n, m]`）を受け取り、内部でLong形式へ展開します。
名称は既存二値分類とのAPI対応を表すもので、
**厳密Kronecker GPの専用推論を意味しません**。

モデルの `fit(epochs=..., lr=...)` 後、Long形式では候補の各行にタスクIDを付け、
Block形式では元の入力候補だけを `predict_proba` に渡します。
Block形式の予測確率の形状は `[q, m, C]` で、タスクごとに最後のクラス軸が1へ正規化されます。
複数タスクに対する提案スコアは、実験目的に合わせてタスク間の平均・最大・重み付き和などを定めます。
潜在 `posterior()` は内部のLong形式（`q*m` 点）で評価するため、
**予測確率の形状と潜在Posteriorの形状は一致しません**。

学習、予測確率、タスク別グラフ、予測エントロピーに基づく候補提案は
[30番Notebook](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)のタスク相関節を参照してください。

### Block形式のProbability Posteriorと獲得関数

Block形式の `predict_proba(X)`、`sample_class_probabilities(X)`、`predictive_entropy(X)` はタスク軸を保持します。例えば候補が `q` 点、タスクが `m` 個、クラスが `C` 個なら、予測確率は `[q, m, C]`、予測エントロピーは `[q, m]` です。

一方、`posterior(X)` と `probability_posterior(X)` はBoTorchの候補軸を保つため、内部Long形式の `q*m` 点を使います。前者は潜在GPの事後分布、後者は確率事後分布であり、`probability_posterior(X).mean` の形状は `[q*m, C]` です。タスク別に表示する場合は `[q, m, C]` にreshapeします。

`mutual_information(X)` はタスクごとに `[q, m]` のBALDスコアを返します。複数タスクを同じ入力で観測する場合、エントロピーやBALDのタスク平均などを使って入力点を提案できます。ただし、この単純な集約は**タスク間の同時情報量を厳密に最適化したものではありません**。


## Multi-Fidelity分類：連続入力とMixed入力

`MultiFidelityMulticlassGPClassifier` は通常の設計変数と忠実度を含む入力を扱い、
`MixedMultiFidelityMulticlassGPClassifier` はカテゴリ設計変数も扱います。
`iteration_fidelity` には反復型忠実度の列インデックス、
`data_fidelities` にはデータ型忠実度の列インデックスを指定します。
少なくとも一つの忠実度列が必要で、カテゴリ列と忠実度列は重複できません。

入力列の一例は「連続設計値、カテゴリID、忠実度」です。
`train_Y` は0から始まるクラスラベルとし、`num_classes` を指定します。
`fit()`、`predict_proba()`、`probability_posterior()`、
`predictive_entropy()` などは分類モデルと同じ契約で使用できます。

候補提案では、評価したい忠実度を決めてから候補点プールの
`predict_proba(candidate_X)` を求め、予測エントロピーなどを比較できます。
評価コストが違う場合、単純にエントロピーが最大の点を選ぶだけでは
費用対効果を最適化したことにはなりません。低／高忠実度での
予測差と必要な評価費用を区別して解釈してください。

具体的な学習・確率予測・低高忠実度比較グラフ・候補提案は
[Notebook 30](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)を参照してください。


### 既存のActive Learning獲得関数との接続

`PredictiveEntropy(model)`、`qPredictiveEntropy(model)`、`qBALD(model)` は、
忠実度列を含む候補点をそのまま受け取ります。単一点獲得関数には候補形状 `[..., 1, d]`、
q-batch獲得関数には `[..., q, d]` を渡します。
q-batchではすべての候補に同じ高忠実度を与える例をNotebookに追加しています。
`qBALD` はクラス数 `C` と候補数 `q` に対して `C**q <= 256` の制約を持つため、
多数候補には周辺スコアを用いるかバッチ構成を検討してください。

予測エントロピーとBALDはラベルの不確実性を指標としますが、
低忠実度を測定した場合に高忠実度の予測がどれだけ改善するかという
**忠実度をまたぐ情報価値**や**評価コスト**を最適化する指標ではありません。


## Deep GP分類：確率的階層と候補提案

`MulticlassSingleTaskDeepGPClassifier` は複数GP層から成る確率的な潜在モデルです。`hidden_dims` で中間層の幅、`num_inducing` で誘導点の数、`posterior_samples` で予測時の潜在サンプル数を指定します。

- `fit(epochs=..., lr=..., num_likelihood_samples=...)`：各GP層の変分パラメータと分類尤度を学習
- `posterior(X)`：潜在関数のBoTorch互換事後分布（非ガウス・サンプル表現）
- `predict_proba(X)`：潜在事後をクラス確率へ変換し、モンテカルロ平均した `[q,C]`
- `probability_posterior(X)`：クラス確率の期待値と事後分散を持つPosterior
- `PredictiveEntropy(model)`、`qPredictiveEntropy(model)`、`qBALD(model)`：既存のActive Learning獲得関数による候補提案

Deep GPの潜在事後は一般にガウスではないため、通常のGPにおける正規分布の閉形式公式を使いません。予測確率や獲得関数の値にはサンプル誤差があり、精密な比較には乱数シード・サンプル数・学習収束の確認が必要です。

入力はまず連続設計変数の `[n,d]` を対象とします。Mixed入力を使う場合はカテゴリ変数を単に実数へ変換しただけで同等とは見なしません。カーネル型モデルとDeep GPの性能比較は保留データで行ってください。

実行例は[Notebook 30](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)を参照してください。


## InfiniteWidthBNN（無限幅NNGPカーネル）分類

`InfiniteWidthBNNMulticlassGPClassifier`（連続入力）と
`MixedInfiniteWidthBNNMulticlassGPClassifier`（連続・カテゴリ混在入力）が利用できます。
`depth`（ReLUの層数）、`weight_variance`、`bias_variance`、
`ard`、`eps` は、既存の二値InfiniteWidthBNN分類器と同じ意味です。
Mixed版では `cat_dims` にカテゴリ特徴の列インデックスを与えます。
少なくとも1つの連続特徴とカテゴリ特徴が必要です。

通常の分類モデルと同様に、`fit()`、`predict_proba()`、
`probability_posterior()`、`predict_class()`を使用できます。
候補提案には既存の`PredictiveEntropy`や`qPredictiveEntropy`を利用します。
`qPredictiveEntropy`は周辺スコアの平均であり、候補間依存を考慮した
同時情報量の最適化ではありません。

**InfiniteWidthBNNとDeepGPは別モデルです。** 前者は単一GPのカーネルとして
無限幅ReLUネットワークから導かれた共分散を使用します。後者は複数の
確率的なGP層を合成し、非ガウス潜在事後をサンプルで扱います。
NNGPカーネルを用いた分類も、非ガウス尤度に対する変分推論が必要です。

学習・クラス確率と候補提案・Mixed入力の例は
[Notebook 30](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)
を参照してください。


## Contextual・Hierarchical分類モデル

`ContextualMulticlassGPClassifier` は、コンテキストを表す整数IDを入力列の一つとして受け取ります。例えば `train_X=[design_x, context_id]` として `context_feature=-1` を指定し、`rank` でコンテキスト間の相関の階数を制御します。

`HierarchicalMulticlassGPClassifier` は、グループとサブグループの階層を受け取ります。例えば `train_X=[design_x, group_id, subgroup_id]` として `group_feature=-2`、`subgroup_feature=-1` を指定します。サブグループIDはグループをまたいで再利用できません（グローバルに一意）。同じサブグループIDが複数グループに属するとエラーになります。

両モデルとも `num_classes` を指定し、既存の分類APIで `fit()`、`predict_proba()`、`probability_posterior()`、`PredictiveEntropy(model)`、`qPredictiveEntropy(model)` を利用します。候補は**既知のコンテキスト／グループIDを明示**して提案し、その条件における分類の不確実性を評価します。コンテキストIDやグループIDを連続最適化してはいけません。新しい未知IDへの外挿は保証しません。

共通＋偏差という加法共分散を使うため、単純なOne-hot入力やカテゴリを連続量として解釈する手法とは異なります。詳しい動作例とグラフは[Notebook 30](../../examples/notebooks/30_multiclass_classification_active_learning.ipynb)を参照してください。

# 31. 順序GP分類：理論と実装の対応

数学的な背景は[理論31章](../theory/31_ordinal_classification.md)を参照してください。モデルの構築・学習・不確実性評価・候補提案は[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)で実行できます。

## クラス確率と潜在GPの事後分布

`posterior(X)` は潜在GPの事後分布、`predict_proba(X)` は観測ラベルの事後予測確率、`probability_posterior(X)` はクラス確率の不確実性を表します。これらは別の量です。

`probability_posterior(X)` の `mean` / `variance` / `credible_interval` は**クラスごとの周辺的な量**であり、全クラスが同時に指定された信用区間に入るという意味ではありません。候補間の依存関係を捨てない評価では、同時潜在サンプルの扱いを確認します。

## Ordered-probitに基づく分類確率Posterior

順序GPでは `predict_proba(X)` の周辺平均クラス確率をGaussian潜在事後から解析的に求めます。`probability_posterior(X)` の確率分散はGauss–Hermite求積、信用区間はクラスごとのモンテカルロ周辺分位点を用います。中間クラスの確率は潜在値に対して一般に単調ではないため、二値probitと同じ単調変換の区間公式は使えません。

## 順序GPの公開APIと獲得関数

### 順序分類を選ぶ判断基準

製造品質の「不良・要確認・良好」のようにラベル間に**順序**があり、
隣接クラスの関係を学習に使いたい場合はOrdinal GPを選びます。
設備A・B・Cのように順序のないラベルなら、通常のマルチクラス分類を検討します。
Ordinal GPは出力ラベルの順序を使うモデルであって、
入力にカテゴリ変数があること自体を意味しません。

現行の公開モデルは次の2つです。

| モデル | 入力 | registry ID | 尤度・潜在関数 |
| --- | --- | --- | --- |
| `OrdinalSingleTaskGPClassifier` | 連続変数 | `ordinal.standard` | Ordered probit・単一潜在GP |
| `MixedOrdinalSingleTaskGPClassifier` | 連続＋カテゴリ | `ordinal.mixed` | 同じOrdered probit・Mixed kernel |

順序ラベルには0から始まる整数 `0, ..., C-1` を渡し、
`num_classes=C`（3以上）を指定します。
`train_Y` は `[n]` または `[n, 1]` を受け付けます。
クラスラベルを文字列のまま渡す暗黙エンコードは行いません。
業務上のラベル名は別の対応表で保持してください。

### 数式から公開メソッドまで

| 理論上の量 | 公開API / 実装 | 重要な注意 |
| --- | --- | --- |
| 変分GPによる潜在分布 $q(f)$ | `posterior(X)` / `latent_posterior(X)` | クラス確率ではない |
| 閾値 $b_1,\ldots,b_{C-1}$ | `model.likelihood.thresholds` | $b_1=0$、残りは増分で単調化 |
| $P(y=c\mid f)$ | `OrderedProbitLikelihood.class_probabilities` | 潜在値を固定した条件付き確率 |
| $P(y=c\mid X,\mathcal D)$ | `predict_proba(X)` | Gaussian潜在分布を積分した予測確率 |
| 予測ラベル分布 | `predictive_distribution(X)` | `Categorical` |
| 予測クラス | `predict_class(X)` | 最大確率クラス（argmax） |
| 潜在関数サンプル | `sample_latent(X)` | 通常は最終次元1 |
| 条件付き確率サンプル | `sample_class_probabilities(X)` | 出力最終次元はクラス数 |
| 学習 | `fit(epochs=..., lr=...)` | 非Gaussian尤度に対する変分学習 |

`OrderedProbitLikelihood` はGPyTorchの `Likelihood` を継承し、
単調な閾値を `softplus` の累積和で構築します。
`expected_log_prob` では `GaussHermiteQuadrature1D` を利用します。
`OrdinalSingleTaskGPClassifier` は既存の
`SingleTaskVariationalGP` を基底として学習と誘導点管理を再利用します。
`MixedOrdinalSingleTaskGPClassifier` はさらに
`make_mixed_covar_module` を使い、
Ordinal尤度や事後確率の計算を重複実装しません。

`predict_proba` では潜在事後の平均 $\mu$ と分散 $s^2$ を使い、
$\Phi((b-\mu)/\sqrt{1+s^2})$ の差から各クラス確率を計算します。
そのため、潜在平均を単に閾値関数へ代入する計算とは異なります。
一方 `sample_class_probabilities` は潜在GPの同時サンプルから
条件付き確率を作るため、BALDやqBALDへ利用できます。

### 最初の利用手順

1. 実測された順序ラベルを `0,1,\ldots,C-1` に対応付ける。
2. 入力 `train_X` とラベル `train_Y` を作り、モデルを初期化する。
3. `fit()` で学習し、`likelihood.thresholds` の順序を確認する。
4. `predict_proba(X)` で確率とクラス順を確認する。
5. 未評価候補で獲得関数を評価し、候補生成器へ渡す。
6. **提案後に初めて**ラベルを観測し、データへ追加して再学習する。

**候補を提示しただけではActive Learningは完了しません。**
提案・ラベル観測・追加学習の一連の流れは
[31番 Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)
で確認できます。

### Active Learningとq-batch

| 獲得関数 | q=1 | q>1で使う関数 | 候補間相関 |
| --- | --- | --- | --- |
| 予測Entropy | `PredictiveEntropy` | `qPredictiveEntropy` | 周辺平均のみ |
| Margin | `MarginUncertainty` | `qMarginUncertainty` | 周辺平均のみ |
| 確率分散 | `ProbabilityVariance` | `qProbabilityVariance` | 周辺平均のみ |
| 情報利得 | `BALD` | `qBALD` | 潜在同時サンプルを利用 |

`qBALD` のjoint情報量はモンテカルロ推定であり、
ラベルベクトルの列挙上限は `C**q <= 256` です。
単にq個の予測確率が得られることは、どの獲得関数でも
joint q最適化できることを保証しません。
`LatentStraddle` は二値の潜在ゼロ境界用なので、
Ordinalモデルへは適用できません。

### Mixed入力と候補探索

`MixedOrdinalSingleTaskGPClassifier(..., cat_dims=[1])` は
入力の2列目をカテゴリとみなし、共通Mixed kernelを使います。
モデルに `cat_dims` を渡しても、
最適化器の探索空間に許容カテゴリ値は自動登録されません。
探索側で `MixedVariableSpace` を設定し、
`optimize_acqf_sampling` の `variable_space` に渡します。

`CandidateConstraints` は候補入力の可行域を制限します。
**予測した品質クラスに対する安全確率制約ではありません。**
後者は独立に確率制約用の獲得関数を設計する必要があります。

### 検証・適用の制約

Notebookの小規模学習は実行可能な使い方を確認するためのものです。
少数のepochで精度・確率校正・ラベル効率が保証されるわけではありません。
順序ラベルにはNLL・Brier score・クラス別Recallを確認し、
順序差に意味があるときに限りMAEも参考にします。
候補提案の性能評価は、独立したテストデータと
Random Samplingを含む複数シードのラベル予算比較で実施します。

理論上の導出は
[順序GP理論](../theory/31_ordinal_classification.md)、
公開モデルの選び分けは
[分類モデル一覧](../models/classification.md)を参照してください。

## 回帰モデルとの合成

[61番Notebook](../../examples/notebooks/61_heterogeneous_classification_constraints.ipynb)では、回帰GP（単目的・多目的）と順序GPを合成し、許容される順序クラス集合の確率で獲得関数を重み付けします。

## 実装の参照

- `src/robotorchan/models/classification/ordinal/standard/single_task.py`
- `src/robotorchan/models/classification/ordinal/likelihood.py`
- `src/robotorchan/models/classification/posterior/probability.py`
- [二値分類の共通基礎ガイド](24_classification.md)


## 非定常・Spectral Mixtureカーネル

順序分類はクラスごとに独立したGPを置くのではなく、
**単一の潜在GPと順序付き閾値**を利用します。
これらのモデルでは、その潜在GPの共分散のみを差し替えます。

- `NonstationaryOrdinalGPClassifier`：入力依存の局所長さ尺度を持つGibbsカーネル
- `SpectralMixtureOrdinalGPClassifier`：周期的・複数周波数の定常共分散
- `MixedNonstationaryOrdinalGPClassifier`：連続＋カテゴリ入力の非定常共分散
- `MixedSpectralMixtureOrdinalGPClassifier`：連続＋カテゴリ入力のSpectral Mixture共分散

標準の順序モデルと同じ`fit()`、`predict_proba()`、
`posterior()`を使用します。非定常の連続入力モデルでは
`local_lengthscale(X)`で局所的な滑らかさを確認できます。
カテゴリ変数の探索可能値はモデルの`cat_dims`とは別に候補生成側で管理します。

理論上、Gibbsカーネルは場所による滑らかさの変化を表し、
Spectral Mixtureは定常な周波数構造を表します。
順序尤度の閾値の意味はどちらのカーネルでも変わりません。
詳細は[非定常GP](../theory/17_nonstationary_gp.md)と
[表現力の高いGP](../theory/22_expressive_gp.md)を参照してください。


## 高次元分類：PCA・PLS・Random Projection・ALEBO

この節では、入力を縮約するモデルとALEBOの座標系を区別します。
以下は**候補プールから予測エントロピーが最大の点を1点選ぶ**例です。
これは候補提案までの最小例であり、連続空間の獲得関数最適化ではありません。

**実行手順：** 訓練データ `train_X`（例：8次元）とクラスラベル `train_Y` を用意し、`n_components=3` でPCA分類モデルを構築します。`fit(epochs=10, lr=0.01)` で学習後、元の8次元の候補プールを `predict_proba(candidate_X)` に渡します。各候補のクラス確率から予測エントロピーを求め、最大の候補を選びます。学習・予測・候補提案を実際に実行するコードは、対応する分類Notebookにまとめます。

`PLSOrdinalGPClassifier` および
`RandomProjectionOrdinalGPClassifier` も
`n_components` を指定して同じ手順で利用できます。
ただし、PLSはラベルの数値表現に依存するため、特に多クラス分類では
クラス番号の置換に対する感度を検証してください。

ALEBOでは `ALEBOOrdinalGPClassifier` に
**埋め込み空間の学習座標と候補座標**を渡します。
元の探索空間の候補をそのまま渡さないでください。
埋め込みの構築、元の実験条件への射影、実行可能領域の処理は
`ALEBOStrategy` 側の責務です。

この例の学習ラベルはAPI説明用の人工データです。
モデル精度やALの改善を示すベンチマークではありません。
実験では保留データを分け、NLL・Brier score・クラス別指標を確認します。


## 高次元の疎ARDモデル：SAAS型とMAP-SAAS型

`SaasOrdinalGPClassifier` は `global_scale` を、
`MapSaasOrdinalGPClassifier` は `tau` を正の値として指定します。
どちらも入力は元の特徴量の次元のままで、PCA等の縮約を事前に要求しません。

使用の流れは次のとおりです。

1. 元の高次元入力 `train_X` とクラスラベル `train_Y` でモデルを構築する
2. `num_classes` を指定し、`model.fit(epochs=..., lr=...)` で学習する
3. 未測定の高次元候補を `model.predict_proba(candidate_X)` に渡す
4. 予測確率からエントロピーなどの獲得スコアを計算し、候補を選ぶ

これは**変分分類モデルに疎ARD事前分布を導入する実装**です。
完全Bayesian SAASのMCMC推論とは区別してください。
実行できる学習・予測・候補提案・可視化は
[31番Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)のPhase 6節を参照してください。


## タスク相関GP分類（MultiTask／Block形式）

`MultiTaskOrdinalGPClassifier` は、入力の1列を整数タスクIDとする**Long形式**に対応します。
`task_feature` でその列を、`rank` でICMタスク共分散のランクを指定します。
各行の `train_Y` は1つのクラス番号です。
未知のタスク番号での予測は学習時のタスク索引範囲を超えるため対象にしません。

`KroneckerMultiTaskOrdinalGPClassifier` は共有入力と、全タスクのラベルを並べた
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
[31番Notebook](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)のタスク相関節を参照してください。

### Block形式のProbability Posteriorと獲得関数

Block形式の `predict_proba(X)`、`sample_class_probabilities(X)`、`predictive_entropy(X)` はタスク軸を保持します。例えば候補が `q` 点、タスクが `m` 個、クラスが `C` 個なら、予測確率は `[q, m, C]`、予測エントロピーは `[q, m]` です。

一方、`posterior(X)` と `probability_posterior(X)` はBoTorchの候補軸を保つため、内部Long形式の `q*m` 点を使います。Block形式モデルの `predict_proba(X)` は通常は元入力の `[q,d]` を受け取り `[q,m,C]` を返しますが、内部計算との整合性のため明示的なタスクID列を付けたLong形式 `[q,d+1]` も受け取り、この場合は `[q,C]` を返します。前者は潜在GPの事後分布、後者は確率事後分布であり、`probability_posterior(X).mean` の形状は `[q*m, C]` です。タスク別に表示する場合は `[q, m, C]` にreshapeします。

`mutual_information(X)` はタスクごとに `[q, m]` のBALDスコアを返します。複数タスクを同じ入力で観測する場合、エントロピーやBALDのタスク平均などを使って入力点を提案できます。ただし、この単純な集約は**タスク間の同時情報量を厳密に最適化したものではありません**。


## Multi-Fidelity分類：連続入力とMixed入力

`MultiFidelityOrdinalGPClassifier` は通常の設計変数と忠実度を含む入力を扱い、
`MixedMultiFidelityOrdinalGPClassifier` はカテゴリ設計変数も扱います。
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
[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)を参照してください。


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

`OrdinalSingleTaskDeepGPClassifier` は複数GP層から成る確率的な潜在モデルです。`hidden_dims` で中間層の幅、`num_inducing` で誘導点の数、`posterior_samples` で予測時の潜在サンプル数を指定します。

- `fit(epochs=..., lr=..., num_likelihood_samples=...)`：各GP層の変分パラメータと分類尤度を学習
- `posterior(X)`：潜在関数のBoTorch互換事後分布（非ガウス・サンプル表現）
- `predict_proba(X)`：潜在事後をクラス確率へ変換し、モンテカルロ平均した `[q,C]`
- `probability_posterior(X)`：クラス確率の期待値と事後分散を持つPosterior
- `PredictiveEntropy(model)`、`qPredictiveEntropy(model)`、`qBALD(model)`：既存のActive Learning獲得関数による候補提案

Deep GPの潜在事後は一般にガウスではないため、通常のGPにおける正規分布の閉形式公式を使いません。予測確率や獲得関数の値にはサンプル誤差があり、精密な比較には乱数シード・サンプル数・学習収束の確認が必要です。

入力はまず連続設計変数の `[n,d]` を対象とします。Mixed入力を使う場合はカテゴリ変数を単に実数へ変換しただけで同等とは見なしません。カーネル型モデルとDeep GPの性能比較は保留データで行ってください。

実行例は[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)を参照してください。


## InfiniteWidthBNN（無限幅NNGPカーネル）分類

`InfiniteWidthBNNOrdinalGPClassifier`（連続入力）と
`MixedInfiniteWidthBNNOrdinalGPClassifier`（連続・カテゴリ混在入力）が利用できます。
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
[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)
を参照してください。


## Contextual・Hierarchical分類モデル

`ContextualOrdinalGPClassifier` は、コンテキストを表す整数IDを入力列の一つとして受け取ります。例えば `train_X=[design_x, context_id]` として `context_feature=-1` を指定し、`rank` でコンテキスト間の相関の階数を制御します。

`HierarchicalOrdinalGPClassifier` は、グループとサブグループの階層を受け取ります。例えば `train_X=[design_x, group_id, subgroup_id]` として `group_feature=-2`、`subgroup_feature=-1` を指定します。サブグループIDはグループをまたいで再利用できません（グローバルに一意）。同じサブグループIDが複数グループに属するとエラーになります。

両モデルとも `num_classes` を指定し、既存の分類APIで `fit()`、`predict_proba()`、`probability_posterior()`、`PredictiveEntropy(model)`、`qPredictiveEntropy(model)` を利用します。候補は**既知のコンテキスト／グループIDを明示**して提案し、その条件における分類の不確実性を評価します。コンテキストIDやグループIDを連続最適化してはいけません。新しい未知IDへの外挿は保証しません。

共通＋偏差という加法共分散を使うため、単純なOne-hot入力やカテゴリを連続量として解釈する手法とは異なります。詳しい動作例とグラフは[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)を参照してください。


## ラベルノイズを考慮したGP分類

`LabelNoiseOrdinalGPClassifier` は、真のクラスから観測ラベルへ移る確率を表す
**クラス遷移行列**を指定して学習します。
`flip_probability=0.2` は各真のクラスが確率0.2で他のクラスへ
一様に置き換わる対称ノイズです。非対称な誤記録率をモデル化するときは
`transition_matrix`（行が真のクラス、列が観測クラス）を指定します。
二つの指定は同時に使えません。遷移行列は各要素が0から1の有限値であり、
各行の総和が1でなければなりません。

`fit()` は**観測ラベル確率による学習目的関数**を使用します。
`predict_clean_proba(X)` は誤記録前の真のクラス分布、
`predict_proba(X)` と `probability_posterior(X)` は誤記録後の
観測ラベル分布を返します。予測クラスや既存の
`PredictiveEntropy` / `qPredictiveEntropy` は既定で
**観測ラベルの不確実性**を評価します。実際に知りたい対象が真の
クラスなのか、誤記録された観測なのかを混同しないでください。

このモデルは遷移行列が**既知かつ固定**であることを前提とします。
ノイズ率の自動推定や、入力位置によって変化する誤記録確率はこのモデルの対象外です。
真のラベルのないデータだけから任意の遷移行列を識別できると想定してはいけません。
Notebook 31 にクラス確率の比較・候補提案の実行例があります。


### 汚染ラベルへの対応

`ContaminatedOrdinalGPClassifier` は、確率 `contamination_probability` で真のラベルとは無関係な
`contaminant_distribution` から観測ラベルが生成される状況を扱います。
これはクラス遷移行列の特殊形であり、既存のラベルノイズ学習処理を再利用します。

汚染率と汚染ラベル分布は既知・固定として指定します。
`predict_clean_proba()` は汚染前、`predict_proba()` は汚染後の予測です。


## 入力不確実性を考慮する分類

`ContinuousUncertainInputOrdinalGPClassifier` は、候補点が正規摂動する状況を扱います。従来の分類モデルと同じ `fit()` で学習した後、`predict_proba_with_input_uncertainty(X, input_std=..., num_samples=...)` または `input_covar=...` を指定して予測確率を周辺化します。標準偏差は `X` と同じ形・dtype・device、共分散は `[...,d,d]` の半正定値行列とします。

`UncertainCategoricalOrdinalGPClassifier` はMixed入力に対して、1列のカテゴリ値が確率的に変わる場合を扱います。`cat_dims` と `uncertain_cat_dim` を指定し、`predict_proba_with_category_uncertainty(X, category_values=..., category_probabilities=...)` でカテゴリごとのクラス確率を確率加重平均します。各候補点のカテゴリ確率の行和は1が必要です。

- 両モデルとも**学習入力自体の誤差を考慮するモデルではありません**。候補点を実際に評価するときの入力不確実性を予測上で周辺化します。
- 通常の `predict_proba(X)` と `PredictiveEntropy(model)(X.unsqueeze(-2))` は、公称入力に対する結果です。入力不確実性を含む候補選択を行うには、周辺化済みの予測確率からクラスエントロピーなどを計算してください。既存の獲得関数が自動的に入力を摂動するわけではありません。
- 連続摂動の周辺化はモンテカルロ近似です。カテゴリ分布の周辺化は有限サポートの正確な重み付き和です。学習モデルの不確実性と入力不確実性は異なります。

[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb) に、単一モデルの学習、候補確率、可視化と候補提案を追加しています。


## Ensembleと非GP分類器

`RandomForestOrdinalClassifier`、`ExtraTreesOrdinalClassifier`、`GradientBoostingOrdinalClassifier`、`HistGradientBoostingOrdinalClassifier` は、scikit-learnの分類器を使用する公開アダプターです。`train_X`、`train_Y`、`num_classes`を渡し、`fit()`後に`predict_proba(X)`、`predict_class(X)`、`predictive_entropy(X)`を利用します。scikit-learnは任意依存です。

`OrdinalClassificationEnsemble` は、学習済み分類器2つ以上を位置引数に渡し、`weights`でメンバーの重みを設定します。メンバーのクラス数・クラスラベル順序が一致している必要があります。`predict_proba()`は重み付き平均、`probability_posterior()`はメンバー確率に関する経験的Posterior、`mutual_information()`はモデル間の不一致を返します。GPと非GP分類器も同時に組み込めますが、**共通潜在Posteriorはありません**。

非GP単体での`probability_posterior()`、`mutual_information()`、`latent_posterior()`は未定義であり、明示的なエラーとなります。Active Learningの予測エントロピーは単一モデルでも利用できますが、BALD型のメンバー不一致はEnsemble側で評価してください。

順序分類用の非GPツリーモデルはラベルをカテゴリとして扱い、Ordered Probitと同じ閾値構造は仮定しません。順序に特有の尤度を利用したい場合は既存のOrdinal GPを使用してください。

実行例・予測グラフ・候補提案については[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)を参照してください。


### Bootstrap Ensemble

`BootstrapGradientBoostingOrdinalClassifier` は既存の二値Bootstrap Ensembleに対応した拡張です。`n_members`個の独立したブートストラップ標本でGradient Boostingモデルを学習します。`fit()`後は`probability_posterior()`で各モデルのクラス確率分布を、`predict_proba()`で平均を取得できます。クラス数が少ない場合にも各標本が全クラスを含むよう再抽出しますが、希少クラスが極端に少ない場合は標本の作成に失敗することがあります。重み付け可能な任意のモデルの統合には`OrdinalClassificationEnsemble`を使用してください。


## 校正・Conformal予測集合・費用を考慮した判断・Active Learning

既存の`TemperatureScalingCalibrator`はクラス数に依存しません。学習に使っていない検証データで`fit(probabilities, labels)`してから、`CalibratedOrdinalClassifier(model, calibrator)`に渡します。校正済みラッパーは`predict_proba()`、`sample_class_probabilities()`、`probability_posterior()`を提供し、基礎GPの`latent_posterior()`は変更しません。`probability_posterior()`は校正後サンプルの**経験的Posterior**であり、基礎GPの解析的潜在Posteriorではありません。単一非GPモデルの認識論的サンプリングは未対応としてエラーにします。

`SplitConformalClassifier(calibrated_model, alpha=0.1)`はマルチクラス・順序分類でも既存実装を利用できます。基礎モデル学習・温度校正に使っていない**別の保留データ**を`calibrate(calibration_X, calibration_Y)`に渡し、`prediction_set(X)`でクラスごとの真偽値マスクを取得します。交換可能性のもとでの周辺被覆率の保証であり、クラス条件付き被覆や分布シフトに対する保証ではありません。

誤分類コストでは、`multiclass_expected_decision_cost(probabilities, cost_matrix=costs)`と`multiclass_cost_sensitive_prediction(...)`を利用します。`cost_matrix[a, y]`は真のラベルが`y`で行動`a`を選ぶコストです。行動軸と真のクラス軸を取り違えないでください。順序ラベルでは `ordinal_cost_matrix(num_classes, power=1 または 2)` と `ordinal_cost_sensitive_prediction(probabilities, power=...)` を使用します。絶対距離または二乗距離に基づく決定です。

既存の`PredictiveEntropy`、`BALD`、`qPredictiveEntropy`は、校正ラッパーの公開クラス確率APIを通じて利用できます。`qPredictiveEntropy`はバッチ内の周辺エントロピー平均であり、厳密な同時情報利得ではありません。非GP単体にはBALD用Posteriorはありません。予測クラスのコスト、Conformal予測集合の大きさ、予測エントロピーはそれぞれ別の基準です。

[Notebook 31](../../examples/notebooks/31_ordinal_classification_active_learning.ipynb)で、データ分割、モデル学習、温度校正、Conformal集合、費用に基づく判断と候補点提案を確認できます。

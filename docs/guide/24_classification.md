# Classification：latent GP posteriorとclass probabilityを分離する

## 分類GPの確率モデル

二値ラベル`y∈{0,1}`を扱うとき、実数値潜在関数`f(x)`へGP priorを置き、Bernoulli likelihood経由でラベルを生成する。
[理論第24章](../theory/24_classification.md)のとおり、Gaussian priorにBernoulli尤度は非共役なので、通常は近似推論（variationalなど）が必要となる。

**`posterior(X)`はlatent posteriorであって、成功確率ではない。** `predict_proba(X)`はclass probabilities、
`predict_class(X)`は決定規則に従うラベルを返す。これらを混同すると、constrained BOやActive Learningの数式の対象が変わってしまう。

## 理論と実装の対応

| 数学的対象 | 公開契約 | 対応する意味 |
| --- | --- | --- |
| 潜在関数`f(x)`の近似分布 | `model.posterior(X)` | BoTorch-compatibleなlatent値・不確実性 |
| `E[link(f(x)) \| D]` | `model.predict_proba(X)` | 予測クラス確率 |
| 確率からの分類判断 | `model.predict_class(X)` | 閾値・クラス選択に基づく判断 |
| 不確実性の学習 | `model.make_mll()`など | Bernoulli likelihoodに対するvariational目的（Exact Gaussian MLLではない） |
| 目的／制約へ接続 | `robotorchan.acquisition.composition` | 確率objective・classification feasibilityのbridge |

## モデル群の選択

`BinarySingleTaskGPClassifier`は連続入力の基礎モデル。
`MixedBinarySingleTaskGPClassifier`ではcategorical列と連続設計列を分ける。
`MultiTaskBinaryGPClassifier`と`KroneckerMultiTaskBinaryGPClassifier`ではlong-format /
block-designの入力構造が異なる。

高次元ではSAAS-style分類がvariationalであり、回帰のNUTSベース`SaasFullyBayesianSingleTaskGP`と同じ推論方式ではない。
DeepGP分類は確率的な層を持つ。Non-GP分類器がclass probabilityを返す場合でもGaussian latent posteriorが存在するとは限らず、
`posterior`要件のある獲得関数へ無条件に渡せない。

## Active Learning・制約付きBOでの解釈

`PredictiveEntropy`は予測ラベルのentropy、`BALD`は情報量、`LatentStraddle`はlatent decision boundary近傍の探索である。
これらは単一の「uncertainty」として代替できない。`ProbabilityVariance`は予測確率の変動を扱い、latent posterior varianceとは異なる。
分類モデルを可行性`P(pass\|x)`として使う場合、objectiveが直接潜在GP値ではなく**実際の分類確率**を見るようにbridgeを設計する。

## 確率の解釈とモデルの制限

Brier score、NLL、ECEなどはclass probabilityの予測校正を評価するための指標であり、latent varianceの値を直接比較する指標ではない。
現行文書の分類モデルはbinary-firstであり、全multiclass対応を主張しない。出力の全familyやmulti-fidelity分類などの対応可否は
[Classificationモデル一覧](../models/classification.md)で個別に確認する。

## Bernoulli likelihoodがlatent posteriorをクラス確率へ変える

BinarySingleTaskGPClassifierはBoTorchのVariational GPを基礎に、
GPyTorch BernoulliLikelihoodとVariationalELBOを使う。近似潜在分布q(f(x))に対して

$$
P(y=1\mid x,\mathcal D)
=\mathbb E_{q(f(x))}[P(y=1\mid f(x))]
$$

とするため、一般にlatent posterior meanを単にsigmoidやprobit変換した値と同一視できない。
src/robotorchan/models/classification/binary/standard/single_task.pyではpredictive_distribution()が
latent_posterior(X).distributionをself.likelihoodへ渡し、
predict_proba()は結果のpositive probabilityから[P(y=0) ,P(y=1)]を構築する。
sample_class_probabilities()は別経路としてlatent samplesから条件付き確率を生成する。

predict_class()はpredict_proba()[...,1]に閾値を適用するだけであり、クラス確率の校正を保証しない。閾値は0〜1の有限数である必要があり、範囲外を拒否する。
Bernoulli likelihoodに対するELBOを利用する点はExact GPのGaussian MLLと異なる。

## HeterogeneousModelで回帰と分類の意味論を保つ

src/robotorchan/models/heterogeneous.pyのHeterogeneousModelはtorch.nn.Moduleのcontainerであり、
BoTorchの単一Modelとして全outputを結合したposterior()を提供しない。entry_posterior(key,X)は子のposteriorをそのまま返すため、
分類子ではlatent distributionになる。entry_predict_proba(key,X)はClassificationModelMixinの子にのみ使える。
回帰子で呼ぶとTypeErrorになる。

分類子と回帰子を併用する制約付きBOでは、回帰acquisitionの価値とP(pass|x)を異なる数理量として扱う。分類出力をGaussian regressionの1列として連結しない。
global output indexは各子のnum_outputsに応じて割り当て、output_ownerが子モデルのindexとlocal output indexを返す。
output_namesやnamesは別名であり、位置indexが基準となる。

ModelListGPはBoTorch ModelListGPの独立GP合成とSumMarginalLogLikelihoodを用いるもので、
分類確率と回帰posteriorの意味論を管理するHeterogeneousModelとは責務が異なる。ModelListGPを異種モデルの単一posteriorとして説明してはいけない。

## 分類確率を目的と制約で使い分ける

同じpredict_proba(X)[..., c]でも、最大化する対象と可行性の判定に用いる対象では意味が異なる。
ProbabilityObjective(output, class_index, direction)は対象クラスの予測確率をスカラー目的として評価する。
例えば良品になる確率自体を最大化する場合である。一方、ClassificationProbabilityOfFeasibility(model,
feasible_class)はクラス確率をP(feasible|X,D)として取得し、回帰目的の獲得関数に外部重みとして用いる。

- **分類確率を最大化・最小化**
  公開契約：ProbabilityObjective.evaluate(model,X)
  実装上の区別：指定クラス列を抽出しObjectiveDirectionを適用。RegressionObjectiveではない

- **分類確率から潜在posteriorに基づくサンプルを生成**
  公開契約：ProbabilityObjective.sample(model,X,sample_shape=...)
  実装上の区別：classifier.sample_class_probabilitiesに依存。全分類器で利用可能とは限らない

- **分類を制約として利用**
  公開契約：ClassificationProbabilityOfFeasibility(X)
  実装上の区別：predict_probaからfeasible_classの列を抽出。一般のjoint可行性ではない

- **サンプル単位の可行確率**
  公開契約：SampleProbabilityOfFeasibility
  実装上の区別：epistemic sampleの確率。平均PoFをそのままMC sampleへ置き換えない

- **観測値に基づく連続制約**
  公開契約：SampleResidualFeasibility
  実装上の区別：posterior sampleから残差を計算し、残差 <= 0で可行

- **予測確率そのものの閾値制約**
  公開契約：ProbabilityResidualFeasibility
  実装上の区別：決定論的な確率残差。P(別の制約が成立)とは異なる

ProbabilityObjectiveのdirection=MINIMIZEは対象確率へ方向変換を適用するが、確率の物理的定義が負になるわけではない。分類制約では可行クラス番号を明示し、
ラベルが0を可行とする問題で既定値feasible_class=1を誤用しない。class_indexはclassification metadataから解決され、
未対応ラベル・回帰モデルへの指定は拒否される。

### posterior predictiveとepistemic sampleは別の期待値

~~~text
latent posterior q(f|D)
    ├─ likelihoodによる周辺化 → predict_proba(X) → deterministic PoF
    └─ latent sample → sample_class_probabilities(X) → sample-wise PoF
~~~

予測確率の平均と、潜在関数posterior由来の確率sampleは異なる。
src/robotorchan/acquisition/composition.py::make_sample_classification_feasibility_bridgeは
classification registryのsupports_probability_samplesをチェックし、非対応ならTypeErrorで拒否する。
サンプル化できない分類器に平均PoFを複製して代替しない。

## 関連する実装資料

ライブラリ固有のモデル群、予測API、対応する組合せについては
[分類モデルガイド](../models/classification.md)を参照してください。

## 変分GPを分類器へ拡張する継承

[binary/standard/single_task.py][binary]の`BinarySingleTaskGPClassifier`は
`BinaryClassificationMixin`と`SingleTaskVariationalGP`を継承します。近似GPの計算を再利用し、
Bernoulli likelihoodと分類用ELBOへ接続します。`predictive_distribution()`がlatent分布をlikelihoodへ渡し、
`predict_proba()`がその正例確率から2クラス分のtensorを構成します。

Theoryのlink関数を介した期待値は、このlatent分布から予測ラベル分布への変換に対応します。
潜在sampleからの`sample_class_probabilities()`は別の経路であり、平均確率を複製してepistemic sampleにする処理ではありません。
クラス確率の構成・分類の判断はローカル拡張で、Bernoulli分布と変分GPの演算はGPyTorch・BoTorchの担当です。

[binary]: ../../src/robotorchan/models/classification/binary/standard/single_task.py

使用手順と結果の可視化は[39番Notebook](../../examples/notebooks/39_classification_active_learning.ipynb)を参照してください。

## 不確実性・確率校正・予測集合の責務分離

[分類理論の第6.2節](../theory/24_classification.md)では、
ラベルの誤記録、候補入力の揺らぎ、Ensembleによる確率のばらつき、
確率校正、OOD、Conformalの違いを説明しています。
これらはすべて「不確実性」と関連しますが、
操作する確率分布もモデルの契約も異なります。

| 問題 | 対応する実装 | 予測への影響 |
| --- | --- | --- |
| 観測ラベルの誤記録 | LabelNoiseBinarySingleTaskGPClassifier | 観測ラベルのBernoulli尤度を変更 |
| 候補入力の揺らぎ | ContinuousUncertainInputBinarySingleTaskGPClassifier | 摂動後の入力に対する予測確率を平均 |
| モデル間の不一致 | ClassificationEnsemblePosterior | メンバー確率の平均と分散を計算 |
| 予測確率の校正 | TemperatureScalingCalibrator | クラス確率を温度で変換 |
| 校正後の分類器 | CalibratedBinaryClassifier | 確率の予測と確率サンプルへ校正を適用 |
| 予測集合の構成 | SplitConformalClassifier | 保留データに基づいて予測集合を返す |

LabelNoiseBinarySingleTaskGPClassifierでは、
観測されたラベルの確率とラベル反転前の確率を区別します。
標準のpredict_probaは観測過程を含む予測です。
ラベル反転前の確率はpredict_clean_probaを参照します。

ContinuousUncertainInputBinarySingleTaskGPClassifierの
predict_proba_with_input_uncertaintyは、
候補時点の連続入力にGaussian摂動を加えて確率を平均します。
これは訓練入力自体の測定誤差を自動で修正する手法ではありません。

Temperature ScalingとSplit Conformalはどちらも
保留データを使いますが、前者は確率を変換し、
後者は指定された被覆率を目指す予測集合を構成します。
Conformalの被覆率は交換可能性などの仮定に依存し、
各入力における条件付き被覆率を保証するわけではありません。

対応する実行例は
[51番Notebook](../../examples/notebooks/40_classification_robust_uncertain.ipynb)
と[41番Notebook](../../examples/notebooks/41_classification_probability_layers.ipynb)
です。41番ではConformalの概念を説明しますが、
予測集合の実際の算出は行っていません。

## 単一点とq-batchの分類Active Learning

[分類獲得関数の理論](../theory/acquisition/13_classification_active_learning.md)
では、予測Entropy、Margin、Probability Variance、BALD、
潜在Straddleの数学的な違いを説明しています。
[獲得関数Guide](acquisition/13_classification_active_learning.md)
では、それぞれが要求するモデルの公開メソッドと
q-batchの対応方式を確認できます。

単一点のPredictiveEntropy等はq=1用です。
qPredictiveEntropy・qMarginUncertainty・qProbabilityVariance・qLatentStraddleは
周辺スコアを平均します。一方qBALDは同時ラベル情報利得を推定し、
二値ではq=1〜8、マルチクラスと順序分類ではクラス数Cについて
`C**q <= 256`の上限があります。
候補間の重複を直接考慮するかどうかが異なります。

## 拡張二値分類モデルの使い分け

評価忠実度が異なる場合は`MultiFidelityBinaryGPClassifier`、
連続とカテゴリの混在には`MixedMultiFidelityBinaryGPClassifier`を選びます。
周期的な境界にはSpectral Mixture、Contextごとの差にはContextual、
親子構造を持つデータにはHierarchicalを検討します。
ReLUネットワーク由来の非定常なカーネルを試す場合はInfinite-width BNNを選びます。

どのモデルでも、学習には`fit()`、クラス確率には`predict_proba()`、
潜在関数には`posterior()`を使用します。各モデルの構築引数と選択表は
[モデル一覧](../models/classification.md)、実行コードは
[分類拡張Notebook](../../examples/notebooks/47_binary_classification_extensions.ipynb)を参照してください。


## マルチクラス分類GPの利用

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

[51番の実行Notebook](../../examples/notebooks/51_multiclass_classification_active_learning.ipynb)と
[対応表](../models/multiclass-active-learning.md)も参照してください。


### はじめてのマルチクラス分類：入力と出力

たとえば設備の状態を「正常・注意・異常」に分けるとき、
ラベルはそれぞれ `0, 1, 2` とします。
入力 `train_X` は `[観測数, 特徴量数]`、ラベル `train_Y` は
`[観測数]` の整数テンソルです。

実行例では `MulticlassSingleTaskGPClassifier` に `num_classes=3` を指定し、
`fit()` の後に `predict_proba()` を呼び出します。
3点を予測した場合の出力形状は `[3, 3]`（3点 × 3クラス）です。
コードは[51番Notebook](../../examples/notebooks/51_multiclass_classification_active_learning.ipynb)にまとめています。

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


## Ordinal GP：順序ラベルと実装の対応

### 順序分類を選ぶ判断基準

製造品質の「不良・要確認・良好」のようにラベル間に**順序**があり、
隣接クラスの関係を学習に使いたい場合はOrdinal GPを選びます。
設備A・B・Cのように順序のないラベルなら、通常のマルチクラス分類を検討します。
Ordinal GPは出力ラベルの順序を使うモデルであって、
入力にカテゴリ変数があること自体を意味しません。

現行の公開モデルは次の2つです。

| モデル | 入力 | registry ID | 尤度・潜在関数 |
| --- | --- | --- | --- |
| \`OrdinalSingleTaskGPClassifier\` | 連続変数 | \`ordinal.standard\` | Ordered probit・単一潜在GP |
| \`MixedOrdinalSingleTaskGPClassifier\` | 連続＋カテゴリ | \`ordinal.mixed\` | 同じOrdered probit・Mixed kernel |

順序ラベルには0から始まる整数 \`0, ..., C-1\` を渡し、
\`num_classes=C\`（3以上）を指定します。
\`train_Y\` は \`[n]\` または \`[n, 1]\` を受け付けます。
クラスラベルを文字列のまま渡す暗黙エンコードは行いません。
業務上のラベル名は別の対応表で保持してください。

### 数式から公開メソッドまで

| 理論上の量 | 公開API / 実装 | 重要な注意 |
| --- | --- | --- |
| 変分GPによる潜在分布 $q(f)$ | \`posterior(X)\` / \`latent_posterior(X)\` | クラス確率ではない |
| 閾値 $b_1,\ldots,b_{C-1}$ | \`model.likelihood.thresholds\` | $b_1=0$、残りは増分で単調化 |
| $P(y=c\mid f)$ | \`OrderedProbitLikelihood.class_probabilities\` | 潜在値を固定した条件付き確率 |
| $P(y=c\mid X,\mathcal D)$ | \`predict_proba(X)\` | Gaussian潜在分布を積分した予測確率 |
| 予測ラベル分布 | \`predictive_distribution(X)\` | \`Categorical\` |
| 予測クラス | \`predict_class(X)\` | 最大確率クラス（argmax） |
| 潜在関数サンプル | \`sample_latent(X)\` | 通常は最終次元1 |
| 条件付き確率サンプル | \`sample_class_probabilities(X)\` | 出力最終次元はクラス数 |
| 学習 | \`fit(epochs=..., lr=...)\` | 非Gaussian尤度に対する変分学習 |

\`OrderedProbitLikelihood\` はGPyTorchの \`Likelihood\` を継承し、
単調な閾値を \`softplus\` の累積和で構築します。
\`expected_log_prob\` では \`GaussHermiteQuadrature1D\` を利用します。
\`OrdinalSingleTaskGPClassifier\` は既存の
\`SingleTaskVariationalGP\` を基底として学習と誘導点管理を再利用します。
\`MixedOrdinalSingleTaskGPClassifier\` はさらに
\`make_mixed_covar_module\` を使い、
Ordinal尤度や事後確率の計算を重複実装しません。

\`predict_proba\` では潜在事後の平均 $\mu$ と分散 $s^2$ を使い、
$\Phi((b-\mu)/\sqrt{1+s^2})$ の差から各クラス確率を計算します。
そのため、潜在平均を単に閾値関数へ代入する計算とは異なります。
一方 \`sample_class_probabilities\` は潜在GPの同時サンプルから
条件付き確率を作るため、BALDやqBALDへ利用できます。

### 最初の利用手順

1. 実測された順序ラベルを \`0,1,\ldots,C-1\` に対応付ける。
2. 入力 \`train_X\` とラベル \`train_Y\` を作り、モデルを初期化する。
3. \`fit()\` で学習し、\`likelihood.thresholds\` の順序を確認する。
4. \`predict_proba(X)\` で確率とクラス順を確認する。
5. 未評価候補で獲得関数を評価し、候補生成器へ渡す。
6. **提案後に初めて**ラベルを観測し、データへ追加して再学習する。

**候補を提示しただけではActive Learningは完了しません。**
提案・ラベル観測・追加学習の一連の流れは
[52番 Notebook](../../examples/notebooks/52_ordinal_classification_active_learning.ipynb)
で確認できます。

### Active Learningとq-batch

| 獲得関数 | q=1 | q>1で使う関数 | 候補間相関 |
| --- | --- | --- | --- |
| 予測Entropy | \`PredictiveEntropy\` | \`qPredictiveEntropy\` | 周辺平均のみ |
| Margin | \`MarginUncertainty\` | \`qMarginUncertainty\` | 周辺平均のみ |
| 確率分散 | \`ProbabilityVariance\` | \`qProbabilityVariance\` | 周辺平均のみ |
| 情報利得 | \`BALD\` | \`qBALD\` | 潜在同時サンプルを利用 |

\`qBALD\` のjoint情報量はモンテカルロ推定であり、
ラベルベクトルの列挙上限は \`C**q <= 256\` です。
単にq個の予測確率が得られることは、どの獲得関数でも
joint q最適化できることを保証しません。
\`LatentStraddle\` は二値の潜在ゼロ境界用なので、
Ordinalモデルへは適用できません。

### Mixed入力と候補探索

\`MixedOrdinalSingleTaskGPClassifier(..., cat_dims=[1])\` は
入力の2列目をカテゴリとみなし、共通Mixed kernelを使います。
モデルに \`cat_dims\` を渡しても、
最適化器の探索空間に許容カテゴリ値は自動登録されません。
探索側で \`MixedVariableSpace\` を設定し、
\`optimize_acqf_sampling\` の \`variable_space\` に渡します。

\`CandidateConstraints\` は候補入力の可行域を制限します。
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
[分類GP理論](../theory/24_classification.md)、
公開モデルの選び分けは
[分類モデル一覧](../models/classification.md)を参照してください。

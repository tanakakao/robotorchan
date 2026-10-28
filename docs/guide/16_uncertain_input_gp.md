# Uncertain-input GP：理論と実装

## 背景と理論の入口

観測した説明変数の位置が不確かなtraining-input uncertaintyと、提案した候補を実行するときのdecision-time perturbationを区別する。Gaussian入力不確実性をRBF kernelに積分すると、入力分布の平均と共分散に依存するexpected kernelを得る。これは入力へ単発のjitterを付ける近似とは異なる。

数式と確率的仮定の正本は[対応する理論章](../theory/16_uncertain_input_gp.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/uncertain/uncertain_input.py`の`GaussianUncertainInputKernel`は、平均とflattenした入力共分散を含む表現から解析的期待RBF共分散を計算する。Choleskyによる線形系とlog determinantを利用する。
- **Implemented:** `_validate_input_uncertainty`は`train_X_std`と`train_X_covar`をちょうど一方指定する契約を検査し、共分散の対称性と半正定値性をチェックする。
- **Extended:** Mixed版の期待kernelでは連続部の不確実性とカテゴリ等値kernelを区別する。カテゴリIDを正規分布の連続座標と扱わない。

## APIと実装を読むための最小例

```python
import torch
from robotorchan.models import UncertainInputSingleTaskGP

train_X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(5 * train_X)
train_X_std = torch.full_like(train_X, 0.02)
model = UncertainInputSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    train_X_std=train_X_std,
)
candidate_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
posterior = model.posterior(candidate_X)
assert posterior.mean.shape[-2:] == (2, 1)
```

コード例は公開APIを示す静的な例であり、このセッションで実行して成功したとの主張ではない。

## 適用条件、テスト、限界

この解析結果はGaussian入力不確実性×RBF kernelの仮定に限定される。零入力分散で通常のRBFへ戻る極限、共分散対称性、PSD、dtype/device、zero/noisy casesを別々に検証する。candidate-time robust decisionの保証を自動的に付与しない。

旧[モデルMapping](../../.github/project-records/theory-guide/mapping/models.md)・[数学的検証Mapping](../../.github/project-records/theory-guide/mapping/verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../models/robust_noise.md)には利用者向けの比較を記す。

## Gaussian入力不確実性の積分と公開posterior座標

訓練入力の真の位置をz_i~N(mu_i,Sigma_i)とし、RBF共分散を仮定する。2つの独立なGaussian位置分布の共分散を平均化したexpected kernelは、lengthscale平方の対角行列LambdaとS=Sigma_i+Sigma_jを使い、

$$
\bar k(i,j)=|\Lambda|^{1/2}|\Lambda+S|^{-1/2}
\exp[-\tfrac12(\mu_i-\mu_j)^T(\Lambda+S)^{-1}(\mu_i-\mu_j)]
$$

となる（単位outputscaleのRBFの場合）。GaussianUncertainInputKernelは入力を[mean(d), flattened covariance(d*d)]という内部表現に拡張し、Cholesky solveとlogdetによって上式を評価する。ゼロ分散の極限で通常のRBFに戻る。

UncertainInputSingleTaskGPの公開constructorはtrain_X_stdとtrain_X_covarをちょうど一方だけ受け取る。stdはtrain_Xと同じshapeで非負・有限、full covarianceは[...,d,d]で有限・対称・PSDでなければならない。posterior(X)では元のd次元deterministic候補を受け取り、zero covarianceを連結して内部posteriorへ渡す。**不確かな訓練位置**と**不確かな実行候補**を同時に周辺化するAPIと解釈しない。

MixedUncertainInputSingleTaskGPはカテゴリ列の不確実性を正規分布として近似しない。連続列だけにGaussian uncertaintyを与え、native CategoricalKernelを積として組み合わせる。少なくとも1つの連続設計列が必要で、train_X_std/covarのshapeは連続列数に合わせる。

根拠：src/robotorchan/models/uncertain/uncertain_input.py。解析式とpublic→augmented座標変換のソース照合に基づき、zero-variance limitの独立数値計算やposterior精度の再検証は今回行っていない。

## 理論章から分離した実装上の条件

### 12. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- continuous training-input uncertaintyはcaller-supplied Gaussianとして扱う
- uncertainty metadata自体をtrain_X / train_Yから自動推定するモデルではない
- RBF covariance を入力分布について解析的に周辺化する
- diagonal uncertainty は `train_X_std`
- correlated uncertainty は `train_X_covar`
- covariance は finite / symmetric / positive semidefinite を要求
- training representation は `[mean, flattened covariance]`
- `posterior(X)` の candidate は deterministic
- Mixed 版では continuous uncertainty と categorical identity を分離
- categorical uncertainty は専用モデルで扱う
- candidate perturbation / environmental scenarioは別layer
- robust candidate evaluationでは `InputPerturbation` とrisk objectiveを独立に合成できる
- current public uncertain-input familyからMultiTask等の未実装cross-productを推測しない

実装仕様は [uncertain-input GP](../models/uncertain_input_gp.md)、
[full-covariance uncertain-input GP](../models/full_covariance_uncertain_input_gp.md)、
[mixed uncertain-input GP](../models/mixed_uncertain_input_gp.md)、
[uncertain categorical GP](../models/uncertain_categorical_gp.md) を参照してください。

#### 直感：実装上の補足

この問題は candidate perturbation や environmental scenario と関連しますが、robotorchan では
training-input uncertainty と decision-time perturbation を明確に分けます。

#### 2. Gaussian uncertainty と RBF kernel：実装上の補足

robotorchan の `GaussianUncertainInputKernel` は Gaussian input uncertainty と RBF kernel の
組み合わせに対する解析的な expected covariance を実装します。

#### 5. Training uncertaintyを学習しているわけではない：実装上の補足

現行 `UncertainInputSingleTaskGP` の `train_X_std` / `train_X_covar` は、
callerが与えるuncertainty metadataです。

#### 6. Prediction candidate は deterministic：実装上の補足

現在の `UncertainInputSingleTaskGP.posterior(X)` は candidate `X` を deterministic として
扱います。内部では candidate covariance をゼロにして augmented representation を作ります。

#### 8. Categorical uncertainty：実装上の補足

robotorchan の `UncertainCategoricalSingleTaskGP` は、continuous coordinate uncertainty
とは分離してこの問題を扱います。

#### Uncertain categoricalはdeterministic categoryのMixed GPとも違う：実装上の補足


のような入力です。現行実装のexpected categorical covarianceは、学習されるPSDな
category covariance matrixをこの確率分布で両側から平均化します。

#### Uncertain categoricalはdeterministic categoryのMixed GPとも違う：実装上の補足


ではありません。また現行public modelは「複数のuncertain categorical columnsを
一般的に自動処理するAPI」と解釈しないことが重要です。

#### 10. Training-input uncertainty と candidate perturbation：実装上の補足

robotorchan では次を区別します。

#### 10. Training-input uncertainty と candidate perturbation：実装上の補足


とする場合、現行のrobust BO pipelineではBoTorchの `InputPerturbation` が
posterior評価時にnominal candidateをscenario axisへ展開し、ExpectationやCVaRなどの
risk objectiveがそのaxisを集約できます。

#### 現行scopeを越えて推測しない：実装上の補足

#### 現行scopeを越えて推測しない

#### 13. この章で覚えておくこと：実装上の補足

- output noiseとinput uncertaintyは異なる
- training-input uncertaintyとcandidate-time perturbationも異なる
- 現行continuous modelはcaller-supplied Gaussian input uncertaintyをexpected RBF kernelへ入れる
- analytic expected-kernel式はGaussian uncertainty × RBFという仮定に依存する
- `train_X_std` / `train_X_covar` は学習される量ではなく入力metadataである
- `posterior(X)` のquery candidateは現行実装ではdeterministicである
- Mixed版ではGaussian uncertaintyはcontinuous dimensionsだけに適用する
- uncertain categorical inputはcategory probability distributionとして別modelで扱う
- candidate robustnessはscenario / risk-objective layerで定義する

## 数値 API と利用上の補足

この区別により、外部 API では元の feature dimension を維持できます。

詳細は
[uncertain categorical GP](../models/uncertain_categorical_gp.md) を参照してください。

- output noiseとinput uncertaintyは異なる
- training-input uncertaintyとcandidate-time perturbationも異なる
- 現行continuous modelはcaller-supplied Gaussian input uncertaintyをexpected RBF kernelへ入れる
- analytic expected-kernel式はGaussian uncertainty × RBFという仮定に依存する
- `posterior(X)` のquery candidateは現行実装ではdeterministicである
- Mixed版ではGaussian uncertaintyはcontinuous dimensionsだけに適用する
- uncertain categorical inputはcategory probability distributionとして別modelで扱う
- candidate robustnessはscenario / risk-objective layerで定義する

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

ここで行っているのは「観測された `train_X` を一度だけrandom jitterして通常GPをfitする」
処理ではありません。各training pointを入力分布として扱い、その分布間でkernelを周辺化します。

`UncertainInputSingleTaskGP` は2種類の training-input uncertainty 表現を受け取ります。

```text
train_X_std.shape == train_X.shape
```


```text
Sigma_x = diag(train_X_std^2)
```


```text
train_X_covar.shape == train_X.shape[:-1] + (d, d)
```


`train_X_std` と `train_X_covar` は同時には指定せず、必ずどちらか一方を指定します。

これはユーザー向け raw feature space を変更するための feature engineering ではありません。
`raw_train_X` は元の入力を保持し、augmentation は model 内部の covariance 計算のための
private representation です。

つまりモデルが `train_X` と `train_Y` だけから各点の入力誤差分散を自動推定するわけではありません。

~~~text
caller
  → observed train_X
  → train_X_std または train_X_covar
  → uncertain-input kernel

model
  → 与えられた入力分布を使ってcovarianceを計算
~~~


`MixedUncertainInputSingleTaskGP` は

`MixedUncertainInputSingleTaskGP` のcategoryは確定値です。一方、
`UncertainCategoricalSingleTaskGP` は1つのcategorical featureについて
category probability vectorを明示的に受け取ります。

- `UncertainInputSingleTaskGP`
- `MixedUncertainInputSingleTaskGP`
- `UncertainCategoricalSingleTaskGP`

## 専用処理の補足

これはユーザー向け raw feature space を変更するための feature engineering ではありません。
`raw_観測入力` は元の入力を保持し、augmentation は model 内部の covariance 計算のための
private representation です。

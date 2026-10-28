# Heteroskedastic / Replicate Noise：理論と実装

## 背景と理論の入口

潜在関数の不確かさ`Var[f(x)|D]`、入力に依存する観測分散`σ²(x)`、反復測定の平均に対する観測分散`s²/r`を区別する。単純に予測分散が大きいことをnoiseが大きいことと読み替えない。

数式と確率的仮定の正本は[対応する理論章](../theory/15_heteroskedastic_noise.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/robust/robust.py`の`HeteroskedasticSingleTaskGP`はresponse GPとresidualから推定するnoise GPを反復更新する経路を持つ。これは完全joint posterior推論と同義ではない。
- **Implemented:** `src/robotorchan/models/robust/joint_heteroskedastic.py`の`JointHeteroskedasticSingleTaskGP`はresponseとlog noiseの2つのvariational GPを持つ。`supports_mll=False`で、`make_mll()`は意図的に例外を返すため、`training_loss()`を使用する。
- **Extended:** replicate-noise系列は反復観測を集約して平均値とその観測分散を扱う。

## APIと実装を読むための最小例

```python
import torch
from robotorchan.models import JointHeteroskedasticSingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(5 * train_X)
model = JointHeteroskedasticSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing=4,
    num_mc_samples=4,
)
loss = model.training_loss()
candidate_X = train_X[:2]
response = model.posterior(candidate_X)
log_noise = model.noise_posterior(candidate_X)
observation_variance = model.predicted_noise(candidate_X)
assert loss.ndim == 0
assert response.mean.shape[-2:] == (2, 1)
assert log_noise.mean.shape[-2:] == (2, 1)
assert observation_variance.shape[-2:] == (2, 1)
```

この例は2つの潜在過程の予測を区別する。responseは関数値のposterior、
log_noiseは対数観測分散のposteriorを表す。predicted_noise()は
対数正規分布のモーメントを用いた観測分散の推定で、responseのposterior分散とは異なる。
学習ステップは実行していない。make_mll()ではなくtraining_loss()を使用する。

## 適用条件、テスト、限界

jointモデルの`train_Y`は1出力を要求し、`num_inducing`、`num_mc_samples`、各betaは正数であることが必要。反復観測の`s²`と`s²/r`の区別は独立数値参照で確認する。noise posteriorは観測ノイズの実測校正と等価ではない。

旧[モデルMapping](../../.github/project-records/theory-guide/mapping/models.md)・[数学的検証Mapping](../../.github/project-records/theory-guide/mapping/verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../models/robust_noise.md)には利用者向けの比較を記す。

## 観測ノイズの予測と応答posteriorの違い

不均一分散の観測モデルは y(x)=f(x)+epsilon(x)、Var[epsilon(x)]=v(x) と分ける。response posteriorのVar[f(x)|D]は観測ノイズv(x)の推定値とは同じではない。

| 経路 | 学習アルゴリズムと実装 | 重要な注意 |
| --- | --- | --- |
| HeteroskedasticSingleTaskGP | models/robust/robust.py。response GPの残差平方からlog noiseをモデル化し、fit_heteroskedastic(iterations)で交互に更新 | 反復的なplug-in推定であり、2つの潜在GPをjointに積分したposteriorではない |
| JointHeteroskedasticSingleTaskGP | models/robust/joint_heteroskedastic.py。response_modelとnoise_modelを別のSingleTaskVariationalGPとして保持 | supports_mll=False。make_mll()は例外、学習目的はtraining_loss()で計算する |

単純な交互更新側のnoise_posterior(X)はlog観測分散のlatent posteriorを返し、predicted_noise(X)はその平均をexpしてnoise_floorで切り上げる。一方joint variational側のpredicted_noise(X)はlog noise posteriorの平均mu_vと分散s_v^2から

$$
\widehat v(x)=\exp(\mu_v(x)+\tfrac12 s_v^2(x))
$$

を用いるため、exp(mu_v)と同じではない。Noise GPに不確実性がある場合は両式の差が生じる。どちらも実測測定分散との校正を保証しない。

Joint版のposterior(X)はresponse processを、noise_posterior(X)はlog observation varianceのlatent processを返す。train_Yは1出力、num_inducing・num_mc_samples・beta_response・beta_noiseはいずれも正値を要求する。学習時にはMLLを流用せずtraining_loss()を最小化する。

根拠：src/robotorchan/models/robust/robust.py、src/robotorchan/models/robust/joint_heteroskedastic.py。今回の調査は静的source inspectionで、noiseの較正やjoint inferenceの数値再実行ではない。

## 理論章から分離した実装上の条件

### 11. Posterior APIで何が返るか

noise-aware modelでも、BOのacquisitionへ渡したい中心量は通常latent responseです。

~~~text
posterior(X)
  → response process f(x) のposterior

noise_posterior(X)
  → log observation variance processのposterior

predicted_noise(X)
  → observation noise varianceの推定値
~~~

したがって `posterior(X).variance` と `predicted_noise(X)` を同じ量として解釈しません。
前者はresponse posterior uncertainty、後者はobservation mechanismのvarianceです。

### 13. 実装との対応

理論章で保持すべき実装契約は次です。

- heteroskedasticityはoutput noiseの入力依存性
- iterative heteroskedastic modelはresidualからlog-noise GPを交互にfit
- heteroskedastic MultiTask / Mixed MultiTaskではtask featureをstructural featureとして分離
- heteroskedastic MultiFidelityではfidelity coordinateをnoise processにも保持
- joint modelはresponse / noise processを同じvariational objectiveで扱う
- joint modelは `training_loss()` を使い、`make_mll()` を一律には使わない
- `posterior(X)` と `predicted_noise(X)` は異なるquantity
- replicate aggregation は exact duplicate raw rows に基づく
- replicate variance は unbiased sample variance
- GP に渡す mean variance は `replicate_variance / replicate_count`
- mean variance には `noise_floor` を適用
- 各 replicate group は2観測以上を要求
- Mixed 版は categorical identity を保持
- MultiFidelity 版は fidelity を replicate identity の一部として保持

対応モデルの仕様は [Robust / Noise model guide](../models/robust_noise.md)、
[Replicate noise GP](../models/replicate_noise_gp.md)、
[Joint heteroskedastic GP](../models/joint_heteroskedastic_gp.md) を参照してください。

#### 直感：実装上の補足

robotorchan では noise の扱いを少なくとも次の3種類に分けます。

#### 最初に3つのvarianceを分ける：実装上の補足


replicateが独立で同じ分散を持つなら、最後の量はおおよそ `s^2(x) / r` です。
robotorchanのreplicate modelが `train_Yvar` に渡すのは個々の測定値のvarianceではなく、
この**平均値の観測variance**です。

#### 2. Heteroskedastic GP：実装上の補足

robotorchanの `HeteroskedasticSingleTaskGP` は、一般論としての完全なjoint inferenceではなく、
**実用的なiterative two-GP procedure**です。`fit_heteroskedastic()` は概ね

#### 2. Heteroskedastic GP：実装上の補足

さらに現行実装には `HeteroskedasticMultiTaskGP` と
`MixedHeteroskedasticMultiTaskGP` があります。MultiTask版ではtask featureを
design-sideのcategorical featureとして扱わず、response GPとnoise GPの両方で
task covarianceを持つstructural featureとして扱います。

#### 3. Joint heteroskedastic inference：実装上の補足

ここで「joint」とは、responseとnoiseの2つのlatent processを同じMonte Carlo
variational objectiveで学習するという意味です。現行実装では
`make_mll()` ではなく `training_loss()` を使います。

#### 4. Replicate noise：実装上の補足

robotorchan の replicate-noise 実装は exact duplicate の design row を group 化し、
group mean と unbiased sample variance を計算します。GP に渡す `train_Yvar` は
replicate variance そのものではなく **mean variance** であり、

#### 5. Replicate の成立条件：実装上の補足

robotorchan の aggregation は raw input row の exact duplicate を同一条件として扱います。
したがって連続値に丸め誤差がある場合、意味的には同一条件でも別 group になる可能性があります。

#### 7. Multi-fidelity replicate noise：実装上の補足

robotorchan には `ReplicateNoiseMultiFidelityGP` もあります。ここでは fidelity feature も
raw design row の一部なので、

## 数値 API と利用上の補足


という処理を行います。`noise_posterior(X)` はlog observation varianceのlatent posterior、
`predicted_noise(X)` はその平均から変換したnoise varianceを返します。

モデル固有の training contract は
[Joint heteroskedastic GP](../models/joint_heteroskedastic_gp.md) を参照してください。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

`HeteroskedasticMultiFidelityGP` ではfidelity coordinateもnoise processの入力に残るため、
responseだけでなくobservation noiseがfidelityによって変化する構造を表現できます。

`JointHeteroskedasticSingleTaskGP` は response と log-noise の latent process を joint
objective で学習する構成です。概念的には

```text
train_Yvar = max(s^2(x) / r, noise_floor)
```


`MixedReplicateNoiseSingleTaskGP` では、continuous / categorical を含む raw row の同一性を
保ったまま replicate aggregation を行います。

`train_Yvar` を外部から与えられる場合は、そのvarianceを既知として使うfixed-noise modelingです。
これは未知のnoise surfaceをresidualから推定するheteroskedastic GPとは異なります。

~~~text
known train_Yvar
  → noiseを観測・推定済みとしてresponse GPへ渡す

heteroskedastic GP
  → Xからnoise process自体を学習する

replicate GP
  → replicateから各group meanのvarianceを推定してtrain_Yvarへ渡す
~~~


- latent posterior varianceとobservation noise varianceを分ける
- replicate GPの `train_Yvar` は個々の測定varianceではなくvariance-of-the-meanである
- known `train_Yvar` とlearned heteroskedastic noiseは別の情報源である
- iterative modelとjoint variational modelは同じheteroskedasticityを異なる推論契約で扱う
- replicate groupingはraw rowのexact equalityに基づき、各groupに2観測以上必要である
- task / fidelity / categoryはnoise modelでもそれぞれのstructural semanticsを保つ
- noisy acquisitionを選ぶこと自体はheteroskedastic surrogateを学習することではない

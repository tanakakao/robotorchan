# Heteroskedastic / Replicate Noise：理論と実装

## 背景と理論の入口

潜在関数の不確かさ`Var[f(x)|D]`、入力に依存する観測分散`σ²(x)`、反復測定の平均に対する観測分散`s²/r`を区別する。単純に予測分散が大きいことをnoiseが大きいことと読み替えない。

数式と確率的仮定の正本は[対応する理論章](../../../theory/15_heteroskedastic_noise.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/robust/robust.py`の`HeteroskedasticSingleTaskGP`はresponse GPとresidualから推定するnoise GPを反復更新する経路を持つ。これは完全joint posterior推論と同義ではない。
- **Implemented:** `src/robotorchan/models/robust/joint_heteroskedastic.py`の`JointHeteroskedasticSingleTaskGP`はresponseとlog noiseの2つのvariational GPを持つ。`supports_mll=False`で、`make_mll()`は意図的に例外を返すため、`training_loss()`を使用する。
- **Extended:** replicate-noise系列は反復観測を集約して平均値とその観測分散を扱う。

## APIと実装を読むための最小例

```python
from robotorchan.models import JointHeteroskedasticSingleTaskGP

print(JointHeteroskedasticSingleTaskGP.__name__)
```

コード例は公開APIを示す静的な例であり、このセッションで実行して成功したとの主張ではない。

## 適用条件、テスト、限界

jointモデルの`train_Y`は1出力を要求し、`num_inducing`、`num_mc_samples`、各betaは正数であることが必要。反復観測の`s²`と`s²/r`の区別は独立数値参照で確認する。noise posteriorは観測ノイズの実測校正と等価ではない。

旧[モデルMapping](../models.md)・[数学的検証Mapping](../verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../../../models/robust_noise.md)には利用者向けの比較を記す。

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

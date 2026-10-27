# Robust Gaussian Process：理論と実装

## 背景と理論の入口

外れ値に頑健であること、ノイズが入力依存であること、入力自体が不確かなことは別問題である。観測の大きな残差にはStudent-t heavy-tail、contamination mixture、sparse relevance pursuitなど異なる確率機構を使う。データを無条件に削除する処理とは異なり、尤度・異常観測の重み・潜在応答への影響をモデル化する。

数式と確率的仮定の正本は[対応する理論章](../../../theory/14_robust_gaussian_process.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Delegated/Extended:** `src/robotorchan/models/robust/robust.py`の`RobustRelevancePursuitMultiTaskGP`はBoTorchの`RobustRelevancePursuitMixin`を利用し、multi-taskのtask列とローカルなraw観測契約を維持する。
- **Implemented:** Student-t系列は非Gaussian尤度とvariational objectiveを使う。標準exact Gaussian MLLを流用しない。
- **Implemented/Extended:** contaminated系列は名目観測と混入観測の混合をモデル化する。明示的な外れ値ラベルの正解を返すとは限らない。
- RRPではsparse outlier relevanceの学習契約を個別に確認し、通常の`SingleTaskGP.fit()`と同じ計算路を仮定しない。

## APIと実装を読むための最小例

```python
import torch
from robotorchan.models import StudentTSingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(6 * train_X)
model = StudentTSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing=4,
    df=4.0,
)
loss = model.training_loss()
posterior = model.posterior(train_X[:2])
assert loss.ndim == 0
assert posterior.mean.shape[-2:] == (2, 1)
```

この例はvariational objectiveを計算し、潜在応答posteriorを取得する。optimizer stepは行っていない。
`make_mll()`はStudent-t系列では使用できず、学習時には`training_loss()`を最小化する。

## 適用条件、テスト、限界

Student-tでは有限分散の自由度条件、観測ノイズとlatent uncertaintyの分離、variational fittingの収束を確認する。relevance pursuitは支持集合の復元精度を別途測る。モデル名やshape testのみから異常検知性能、校正、頑健性を結論付けない。

旧[モデルMapping](../models.md)・[数学的検証Mapping](../verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../../../models/robust_noise.md)には利用者向けの比較を記す。

## Robust relevance pursuitの実装と推論の切り分け

外れ値に頑健なモデルを比較する際は、尤度のheavy-tail、混合汚染、観測ごとのrelevance追跡を区別する。どれもGaussian GP posteriorの標準分散を増やすだけの同一アルゴリズムではない。

src/robotorchan/models/robust/robust.pyのRobustRelevancePursuitSingleTaskGPはBoTorchRobustRelevancePursuitSingleTaskGPを継承する。一方、RobustRelevancePursuitMultiTaskGPはMultiTaskGPとBoTorchのRobustRelevancePursuitMixinを組み合わせ、base likelihoodにoutlier-related noise構造を付加する。MultiTask版はlong-formatのtask_featureを明示的に受け取る。

MultiTask版のto_standard_model()は現在の学習／評価モードを保持しつつ、対応するMultiTaskGPを再構築する補助経路である。raw observations・task列・選択したcovar_moduleなどは継承するが、robust推論の全操作が通常のSingleTaskGP.fit()と同一になるわけではない。

MixedRobustRelevancePursuitMultiTaskGPはtask_featureを構造列として扱うため、cat_dimsと重ねるとnormalize_feature_dimsが拒否する。カテゴリ共分散の構築にはmake_mixed_covar_moduleを利用する。外れ値重みの推論やsupport選択は通常の最大周辺尤度の一段階最適化と混同しない。

**検証境界：** robust.py内のclass継承、mixin初期化、task列とカテゴリ列の検証を確認した。外れ値支持集合の復元精度、likelihood同定、実測ノイズ下の較正や数値的頑健性の比較は今回実行していない。


## Contamination Mixture：対数尤度と外れ値診断

ContaminatedSingleTaskGPは2成分のGaussian observation modelを使う。外れ値混入率をπ、
inlier/outlierの標準偏差をそれぞれσ_in、σ_outとすると、残差rの密度は次の形となる。

$$
p(r)=(1-\pi)\,\mathcal N(r;0,\sigma_{\mathrm{in}}^2)
+\pi\,\mathcal N(r;0,\sigma_{\mathrm{out}}^2).
$$

実装ではinlier/outlier log probabilityの加重和をlogsumexpで安定的に評価する。
training_loss()は潜在GPのMonte Carlo sampleで期待対数尤度を近似し、KL項を加える。
model.make_mll()はRuntimeErrorで拒否される。posterior(X)は潜在応答の予測であり、
汚染ラベルの確定値ではない。contamination_diagnostic(X,Y)は潜在平均を代入して
汚染確率を近似する診断量で、外れ値ラベルの正解や完全Bayesianな周辺確率ではない。

~~~python
import torch
from robotorchan.models import ContaminatedSingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(6 * train_X)
model = ContaminatedSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing=4,
    num_likelihood_samples=4,
)
loss = model.training_loss()
posterior = model.posterior(train_X[:2])
diagnostic = model.contamination_diagnostic(train_X[:2], train_Y[:2])
assert loss.ndim == 0
assert posterior.mean.shape[-2:] == (2, 1)
assert diagnostic.shape == (2, 1)
~~~

## Robust Relevance Pursuit：疎な外れ値構造

RobustRelevancePursuitSingleTaskGPはBoTorchの
RobustRelevancePursuitSingleTaskGPを継承し、robotorchan側で元の教師データと
make_mll()を提供する。ただし、通常のExact GPと同じハイパーパラメータ最適化だけで
outlier support探索が終了するとは限らない。support選択のための上流契約を確認する。

Student-tとContaminatedモデルが非Gaussianな観測モデルをvariational inferenceで扱うのに対し、
RRPは外れ値に関連した疎なnoise構造を持つ。RRPのmake_mll()が利用可能であっても、
Student-tやContaminatedのtraining_loss()と交換可能ではない。

根拠：src/robotorchan/models/robust/{student_t,contaminated,robust}.py。
ここでの数式とAPIの照合は、外れ値検出性能・support回復精度の数値実証ではない。

# Robust Gaussian Process：理論と実装

## 背景と理論の入口

外れ値に頑健であること、ノイズが入力依存であること、入力自体が不確かなことは別問題である。観測の大きな残差にはStudent-t heavy-tail、contamination mixture、
sparse relevance pursuitなど異なる確率機構を使う。データを無条件に削除する処理とは異なり、尤度・異常観測の重み・潜在応答への影響をモデル化する。

数式と確率的仮定の正本は[対応する理論章](../theory/14_robust_gaussian_process.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Delegated/Extended:** `src/robotorchan/models/robust/robust.py`の
  `RobustRelevancePursuitMultiTaskGP`はBoTorchの`RobustRelevancePursuitMixin`を利用し、
  multi-taskのtask列とローカルなraw観測契約を維持する。
- **Implemented:** Student-t系列は非Gaussian尤度とvariational objectiveを使う。標準exact Gaussian MLLを流用しない。
- **Implemented/Extended:** contaminated系列は名目観測と混入観測の混合をモデル化する。明示的な外れ値ラベルの正解を返すとは限らない。
- RRPではsparse outlier relevanceの学習契約を個別に確認し、通常の`SingleTaskGP.fit()`と同じ計算路を仮定しない。

## モデル化と推論上の制限

Student-tでは有限分散の自由度条件、観測ノイズとlatent uncertaintyの分離、variational fittingの収束を確認する。
relevance pursuitは支持集合の復元精度を別途測る。モデル名やshape testのみから異常検知性能、校正、頑健性を結論付けない。

## Robust relevance pursuitの実装と推論の切り分け

外れ値に頑健なモデルを比較する際は、尤度のheavy-tail、混合汚染、観測ごとのrelevance追跡を区別する。
どれもGaussian GP posteriorの標準分散を増やすだけの同一アルゴリズムではない。

src/robotorchan/models/robust/robust.pyのRobustRelevancePursuitSingleTaskGPは
BoTorchRobustRelevancePursuitSingleTaskGPを継承する。一方、
RobustRelevancePursuitMultiTaskGPはMultiTaskGPとBoTorchのRobustRelevancePursuitMixinを組み合わせ、
base likelihoodにoutlier-related noise構造を付加する。MultiTask版はlong-formatのtask_featureを明示的に受け取る。

MultiTask版のto_standard_model()は現在の学習／評価モードを保持しつつ、対応するMultiTaskGPを再構築する補助経路である。
raw observations・task列・選択したcovar_moduleなどは継承するが、robust推論の全操作が通常のSingleTaskGP.fit()と同一になるわけではない。

MixedRobustRelevancePursuitMultiTaskGPはtask_featureを構造列として扱うため、
cat_dimsと重ねるとnormalize_feature_dimsが拒否する。カテゴリ共分散の構築にはmake_mixed_covar_moduleを利用する。
外れ値重みの推論やsupport選択は通常の最大周辺尤度の一段階最適化と混同しない。

## Contamination Mixture：対数尤度と外れ値診断

ContaminatedSingleTaskGPは2成分のGaussian observation modelを使う。外れ値混入率をπ、inlier/outlierの標準偏差をそれぞれσ_in、
σ_outとすると、残差rの密度は次の形となる。

$$
p(r)=(1-\pi)\,\mathcal N(r;0,\sigma_{\mathrm{in}}^2)
+\pi\,\mathcal N(r;0,\sigma_{\mathrm{out}}^2).
$$

実装ではinlier/outlier log probabilityの加重和をlogsumexpで安定的に評価する。
training_loss()は潜在GPのMonte Carlo sampleで期待対数尤度を近似し、KL項を加える。model.make_mll()はRuntimeErrorで拒否される。
posterior(X)は潜在応答の予測であり、汚染ラベルの確定値ではない。contamination_diagnostic(X,Y)は潜在平均を代入して汚染確率を近似する診断量で、
外れ値ラベルの正解や完全Bayesianな周辺確率ではない。

## Robust Relevance Pursuit：疎な外れ値構造

RobustRelevancePursuitSingleTaskGPはBoTorchのRobustRelevancePursuitSingleTaskGPを継承し、
robotorchan側で元の教師データとmake_mll()を提供する。ただし、通常のExact GPと同じハイパーパラメータ最適化だけでoutlier support探索が終了するとは限らない。
support選択のための上流契約を確認する。

Student-tとContaminatedモデルが非Gaussianな観測モデルをvariational inferenceで扱うのに対し、RRPは外れ値に関連した疎なnoise構造を持つ。
RRPのmake_mll()が利用可能であっても、Student-tやContaminatedのtraining_loss()と交換可能ではない。

根拠：src/robotorchan/models/robust/{student_t,contaminated,robust}.py。ここでの数式とAPIの照合は、
外れ値検出性能・support回復精度の数値実証ではない。

### 実装との対応

対応する公開実装の契約は次のとおりです。

- Student-t / contamination は variational inferenceで `training_loss()` を使う
- Student-t は `df > 2`
- contamination は `0 < pi < 1` と `sigma_out > sigma_in > 0`
- posterior は acquisition から利用可能な latent response posterior
- Mixed 版は native categorical covariance を保持
- MultiTask 版は task feature を構造列として分離
- contamination parametersは現行実装では固定constructor parameters
- contamination diagnosticはposterior mean residualに基づく近似診断で、自動削除APIではない
- relevance pursuitはsingle-taskだけでなくlong-format multi-task / Mixed版も公開

API の詳細はコードとモデルガイドを優先してください。

### Student-t・contamination・relevance pursuitの違い

`StudentTSingleTaskGP`は`StudentTLikelihood`とvariational GPを組み合わせる。
`training_loss()`は期待log likelihoodとinducing-variable KLを評価する。`df > 2`は有限な観測分散を持つ範囲にAPIを限定する条件である。
`num_likelihood_samples`は期待値のMonte Carlo近似数、`beta`はKL項の重みとなる。

`ContaminatedSingleTaskGP`はlatent functionのMonte Carlo sampleに対してmixture log likelihoodの期待値を近似する。
`contamination_probability`、`inlier_scale`、`outlier_scale`は固定constructor parametersであり、
これらを全て自動推定するモデルではない。`contamination_diagnostic(X, Y)`はposterior meanの残差に基づく近似的なcomponent確率を返す。
診断結果は観測を自動削除する指示や、真の外れ値ラベルではない。

relevance pursuitのsparse-support更新と最適化は上流の契約に従う。観測側の疎な外れ値構造と、高次元モデルの入力側のsparsityを区別する。Student-t、
contamination、relevance pursuitを同じ学習目的へ置き換えない。

### Mixed・MultiTaskの構造と公開モデル

- **Student-t**
  単一タスク：`StudentTSingleTaskGP`
  Mixed：`MixedStudentTSingleTaskGP`
  MultiTask：`StudentTMultiTaskGP`
  Mixed MultiTask：`MixedStudentTMultiTaskGP`

- **Contamination**
  単一タスク：`ContaminatedSingleTaskGP`
  Mixed：`MixedContaminatedSingleTaskGP`
  MultiTask：`ContaminatedMultiTaskGP`
  Mixed MultiTask：`MixedContaminatedMultiTaskGP`

- **Relevance pursuit**
  単一タスク：`RobustRelevancePursuitSingleTaskGP`
  Mixed：`MixedRobustRelevancePursuitSingleTaskGP`
  MultiTask：`RobustRelevancePursuitMultiTaskGP`
  Mixed MultiTask：`MixedRobustRelevancePursuitMultiTaskGP`

Mixed版はnative categorical covarianceを保つ。MultiTask版の`task_feature`はタスクを表す構造列であり、カテゴリ設計変数と区別する。

Student-t / contaminationの`posterior(X)`がBOへ公開するのはlatent response posteriorである。
heavy-tailed noiseや観測mixture全体を、そのままacquisitionのsampleへ加える意味ではない。入力不確かさ、入力依存ノイズ、
潜在関数の非定常性も別の生成機構として扱う。APIの詳細と関連する検証は[Robust / Noiseモデルガイド](../models/robust_noise.md)を参照する。

## 非Gaussian尤度の期待値とKLを結ぶ実装

[student_t.py](../../src/robotorchan/models/robust/student_t.py)の
`_StudentTGPBase`は`RawDataMixin`と`nn.Module`を継承し、変分response modelを保持します。
`training_loss()`はlatent sampleに対するStudent-t尤度の期待値とKLを組み合わせます。尤度の確率分布や変分GPの演算はGPyTorch・BoTorchを利用し、
robotorchanはこのモデル固有の目的とデータ保持を接続します。

[contaminated.py](../../src/robotorchan/models/robust/contaminated.py)の
`ContaminatedSingleTaskGP`も`RawDataMixin`と`nn.Module`を基礎とし、
`_component_log_prob()`・`_mixture_log_prob()`で成分尤度と混合確率を計算します。
`training_loss()`はsample方向に平均したlog likelihoodへ変分KLを加えます。`contamination_diagnostic()`は別経路で、
posterior meanの残差から混合成分の近似確率を求めます。この診断を潜在変数の厳密な事後推論と取り違えない設計です。

使用手順と結果の可視化は[対応するNotebook](../../examples/notebooks/15_robust_observation_models.ipynb)を参照してください。

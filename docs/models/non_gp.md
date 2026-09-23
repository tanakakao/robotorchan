# Non-GP probabilistic surrogates

robotorchan の non-GP surrogate は、Gaussian Process を装うのではなく、各モデルが持つ
予測サンプルの意味を BoTorch の `Posterior` contract に接続します。現在の公開モデルは
`RandomForestSurrogate`、`ExtraTreesSurrogate`、`GradientBoostingSurrogate`、
`HistGradientBoostingSurrogate` に加え、オプション依存の `NGBoostSurrogate` です。

## 共通 contract

これらのモデルは constructor に与えた `train_X` / `train_Y` を raw training data として
保持し、学習は明示的な `fit()` で行います。GP の marginal likelihood は存在しないため
`supports_mll = False` です。posterior は BoTorch の `EnsemblePosterior` として返し、
MC acquisition function から利用できます。

sklearn による予測経路は candidate `X` に対して微分可能ではありません。そのため
acquisition optimization には `TreeEnsembleSearchStrategy` などの gradient-free search を
使用します。analytic Gaussian acquisition の分布仮定を満たすとは扱いません。

## Random Forest / Extra Trees

RF と Extra Trees では、各 fitted tree の予測を empirical function sample として扱います。
tree 間のばらつきは ensemble disagreement であり、厳密な Bayesian posterior variance や
calibrated observation noise ではありません。

mixed input では公開 API に raw input と `cat_dims` を渡します。現在の sklearn tree backend
は数値配列を消費するため、カテゴリ表現の統計的意味は native categorical model と同一では
ありません。カテゴリコードを nominal covariance とみなす GP と混同しないでください。

## Gradient Boosting / HistGradientBoosting

boosting の individual stage は逐次加法成分であり、posterior sample ではありません。
robotorchan は bootstrap resample ごとに完全な boosting model を fit し、各 complete model の
予測を ensemble member とします。したがって posterior の sample dimension は boosting stage
ではなく bootstrap member に対応します。

multi-output の場合は各 bootstrap member 内で出力ごとの estimator を学習し、同じ bootstrap
member index を共有します。`output_indices` による出力選択も利用できます。

## Acquisition と制約

non-GP ensemble posterior は qEI や qEHVI など BoTorch の MC acquisition と組み合わせます。
制約は surrogate 内部に feasibility policy を埋め込まず、BoTorch の objective / constraint
callable 側で表現します。surrogate と意思決定規則を分離するためです。

## 不確実性の解釈

RF / Extra Trees の tree disagreement と bootstrap boosting の member disagreement は生成過程が
異なります。両者の posterior standard deviation を同じ calibration 指標として直接比較する
ことは避けてください。Phase 13 benchmark の posterior spread も診断値であり、calibrated
credible interval を意味しません。

## 使い分け

滑らかな低次元問題では GP baseline を残し、non-smooth、局所的相互作用、比較的大きな表形式
データでは tree / boosting surrogate を比較対象にできます。モデル選択は固定的な優劣ではなく、
予測誤差、uncertainty diagnostics、BO の regret、計算時間を同じ問題設定で比較して判断します。

Notebook: [Non-GP surrogate](../../examples/notebooks/25_non_gp_surrogates.ipynb)  
Theory: [Empirical ensemble surrogate](../theory/23_non_gp_surrogates.md)


## NGBoostSurrogate

`NGBoostSurrogate` は boosting stage を ensemble member とみなさず、NGBoost が返す Gaussian
予測分布を sampleable な BoTorch `Posterior` に変換します。初期 contract は単一出力の数値回帰
のみです。候補点予測は CPU/numpy backend を通るため入力勾配はサポートせず、MC acquisition と
勾配不要探索を使用します。予測分散を使う回帰 active learning は適用可能ですが、通常の
NGBoost 予測分布だけでは epistemic / aleatoric uncertainty を分離できないため BALD 対応は
主張しません。

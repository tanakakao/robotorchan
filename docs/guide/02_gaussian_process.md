# Gaussian Process：数式から学習・予測の処理を読む

## この章で解決する問題

[GP の理論](../theory/02_gaussian_process.md)では、観測に条件付けた平均と共分散を求めました。この章では、観測をモデルへ渡し、学習目的を構成してパラメータを推定し、候補点の予測分布を取り出すまでをコードと対応付けます。単一タスク・単一出力の Gaussian 観測モデルが対象です。

最初に処理の境界を押さえましょう。モデルの生成は確率モデルとデータを結び付ける操作で、学習そのものではありません。学習は平均・kernel・ノイズのパラメータを更新し、予測はそれらを固定して観測に条件付けます。得られた分布を獲得関数へ渡す段階も、実際の目的関数を観測する段階とは別です。

## 数式に対応するオブジェクト

学習点 X、予測点 X_*、平均 m、kernel k、観測ノイズ共分散 Σ_noise に対して、K_y=K_XX+Σ_noise と書くと、潜在関数の予測は次です。

$$
\mu_*=m(X_*)+K_{*X}K_y^{-1}(y-m(X)),
\qquad
\Sigma_*=K_{**}-K_{*X}K_y^{-1}K_{X*}.
$$

| 数理上の対象 | 実装上の入口 | 役割 |
| --- | --- | --- |
| m(x) | `model.mean_module` | 入力における事前平均を計算する |
| k(x,x') | `model.covar_module` | 点間の事前共分散を計算する |
| 観測ノイズ | `model.likelihood`、`train_Yvar` | 潜在分布から観測分布を作る |
| 対数周辺尤度 | `model.make_mll()` | `ExactMarginalLogLikelihood` を構成する |
| パラメータ推定 | `model.fit()`、`fit_gpytorch_mll(mll)` | 学習目的を上流の optimizer で最適化する |
| 観測への条件付け | `model.posterior(test_X)` | 予測点全体の joint posterior を返す |

平均と kernel は単に予測値を返すものではありません。学習点での m(X)、学習・予測点間の K_*X、予測点間の K_** を構成し、観測残差 y−m(X) を共分散を通じて伝えます。likelihood は Gaussian 観測ノイズを加えて K_y を定めます。逆行列との積や共分散の補正は、これらを使う上流の予測処理で計算されます。

## モデル生成時に何が起きるか

入口は [SingleTaskGP](../../src/robotorchan/models/standard/single_task.py) です。`SingleTaskGP(FittableExactGPModelMixin, BoTorchSingleTaskGP)` として、共通の学習機能と BoTorch のモデルを継承します。

コンストラクタはまず `train_X`、`train_Y`、任意の `train_Yvar` を `detach().clone()` し、その後で元の引数を親モデルへ渡します。親側で変換や前処理が行われても、呼び出し時点のデータを保持するためです。最後に `_store_supervised_training_data()` がコピーを登録します。

予測モデルを定義するのは親の [BoTorch SingleTaskGP](https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/gp_regression.py) です。ここで入力形状の検証、入出力 transform、likelihood、平均、kernel が設定されます。BoTorch 0.18.1 では、指定がなければ平均は `ConstantMean`、共分散は次元に応じた事前分布を持つ ARD RBF、出力変換は `Standardize` です。別の平均や kernel は `mean_module`・`covar_module` で指定できます。この既定 kernel を常に独立した `ScaleKernel` 付きと仮定しないでください。

`train_Yvar` がなければ `GaussianLikelihood` でノイズを推定し、あれば `FixedNoiseGaussianLikelihood` に既知の観測分散を渡します。値は標準偏差ではなく分散です。出力変換を使うと Y と Yvar も変換され、予測結果は元の応答スケールへ戻されます。入力の正規化は暗黙に行われるとは限らないため、必要な `input_transform` は明示してください。

## 学習目的を作り、パラメータを推定する

[ExactGPModelMixin.make_mll()](../../src/robotorchan/models/base.py) は `ExactMarginalLogLikelihood(self.likelihood, self)` を返します。この操作は目的オブジェクトの構成であり、学習の実行ではありません。

理論の周辺尤度は、潜在関数を積分した観測 y の密度です。[GPyTorch の ExactMarginalLogLikelihood.forward()](https://github.com/cornellius-gp/gpytorch/blob/v1.15.2/gpytorch/mlls/exact_marginal_log_likelihood.py) は、潜在の分布へ likelihood を適用して観測分布を作り、`log_prob(target)` を計算します。その後、登録された追加損失・対数事前分布を加え、データ数で割ります。したがって、既定モデルで返る値を理論の「対数尤度の総和」そのものと比較してはいけません。事前分布を使う場合の目的は、純粋な最尤推定とも異なります。

`FittableExactGPModelMixin.fit()` は `supports_fit` を確認し、`fit_gpytorch_mll(self.make_mll(), optimizer_kwargs=...)` を呼びます。戻り値は `None` です。最適化を独自実装するのではなく、[BoTorch の fitting](https://github.com/pytorch/botorch/blob/v0.18.1/botorch/fit.py) に委譲しています。標準の Exact MLL 経路は既定で SciPy optimizer を使い、結果・警告に応じた再試行を管理します。学習の成功は大域的な最適解や予測の較正を保証しません。

`supports_mll = True` と、クラスが `supports_fit = True` を明示する契約も区別してください。特殊な subclass に学習入口が無条件に有効になる設計ではありません。Variational GP の ELBO、Fully Bayesian GP の hyperparameter 積分、DeepGP の学習目的へ、この Exact GP の目的をそのまま適用できません。

## 最小例：生成から学習・予測へ

Python 3.11 以上でリポジトリを `pip install -e ".[dev]"` により導入した環境を想定します。次の入力 X は `[n, d]=[6, 1]`、応答 Y は `[n, m]=[6, 1]` です。

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
Y = (X - 0.4).square()
model = SingleTaskGP(train_X=X, train_Y=Y)
mll = model.make_mll()
fit_gpytorch_mll(mll)
test_X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
p = model.posterior(test_X)
assert p.mean.shape[-2:] == (2, 1)
assert p.variance.shape[-2:] == (2, 1)
```

この例ではノイズ分散を指定せず、既定の likelihood が推定します。`mll` の構成後に `fit_gpytorch_mll` を呼ぶことで、初めてパラメータが更新されます。`model.fit()` を使う場合も同じ共通入口から fitting へ進みます。

予測点は `[q, d]=[2, 1]` で、`p.mean` と `p.variance` は `[2, 1]` です。後者は潜在関数の共分散行列の対角成分です。複数候補の依存を見るには `p.distribution.covariance_matrix` を参照しますが、大きな行列の実体化にはメモリ費用がかかります。batch 付き入力では、先頭に互換な batch 軸が加わります。全 Tensor の dtype・device も整合させてください。

## posterior() の内部を数式に沿って追う

このモデルの予測入口は、BoTorch の [BatchedMultiOutputGPyTorchModel.posterior()](https://github.com/pytorch/botorch/blob/v0.18.1/botorch/models/gpytorch.py) です。評価モードへ切り替えて入力を変換した後、`self(X)` を呼びます。この呼び出しは prior を返す `forward()` だけで完結せず、継承した [GPyTorch ExactGP.__call__()](https://github.com/cornellius-gp/gpytorch/blob/v1.15.2/gpytorch/models/exact_gp.py) による条件付けへ進みます。

訓練側の予測キャッシュがなければ prediction strategy を構成し、学習・予測間の共分散と事前平均を渡します。[DefaultPredictionStrategy](https://github.com/cornellius-gp/gpytorch/blob/v1.15.2/gpytorch/models/exact_prediction_strategies.py) の `_mean_cache()` は likelihood を加えた訓練共分散に対し、中心化した観測を右辺とする線形方程式を解きます。これは数式の α=K_y^{-1}(y−m(X)) に対応します。`exact_predictive_mean()` は K_*X α と予測点の事前平均を加えます。`exact_predictive_covar()` は K_** から条件付けによる補正を引きます。

ここで「Exact GP」は確率モデルの条件付けの区分であり、計算経路を常に密行列 Cholesky に固定する意味ではありません。GPyTorch と `linear_operator` の設定・行列サイズにより solve、キャッシュ、`fast_pred_var` の計算経路が変わります。小規模な直接計算は、そうした最適化を含む実装と理論式を照合するための参照になります。数値安定化用の jitter も測定ノイズのモデルとは区別します。

最後に BoTorch は `GPyTorchPosterior` へ包み、出力 transform を逆変換します。必要なら `posterior_transform` も適用します。この流れ全体を robotorchan 独自の Gaussian conditioning 実装として説明することはできません。

## 固定パラメータで条件付き分布を照合する

学習の収束と予測式の対応を分けて確認するため、次は平均 0、lengthscale 0.4、既知のノイズ分散 0.025 を固定します。標準化を無効にし、数式とモデルが同じ応答スケールを使うようにします。独立に作った RBF 行列を Cholesky 分解し、GPyTorch の予測と比較します。

```python
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from robotorchan.models import SingleTaskGP

train_x = torch.tensor([[0.05], [0.3], [0.7], [0.95]], dtype=torch.double)
train_y = torch.tensor([[0.2], [-0.4], [0.8], [0.1]], dtype=torch.double)
test_x = torch.tensor([[0.15], [0.55], [0.85]], dtype=torch.double)
noise = 0.025
lengthscale = 0.4
kernel = RBFKernel().double()
kernel.lengthscale = lengthscale
fixed_model = SingleTaskGP(
    train_x,
    train_y,
    train_Yvar=torch.full_like(train_y, noise),
    covar_module=kernel,
    mean_module=ZeroMean(),
    outcome_transform=None,
)
fixed_model.eval()


def covariance(left, right):
    return torch.exp(-torch.cdist(left, right).square() / (2 * lengthscale**2))


k_xx = covariance(train_x, train_x)
k_xs = covariance(train_x, test_x)
k_ss = covariance(test_x, test_x)
system = k_xx + noise * torch.eye(len(train_x), dtype=torch.double)
L = torch.linalg.cholesky(system)
reference_mean = k_xs.T @ torch.cholesky_solve(train_y, L)
reference_covariance = k_ss - k_xs.T @ torch.cholesky_solve(k_xs, L)
with torch.no_grad():
    fixed_posterior = fixed_model.posterior(test_x)
    noisy_posterior = fixed_model.posterior(
        test_x, observation_noise=torch.full_like(test_x, noise)
    )
torch.testing.assert_close(fixed_posterior.mean, reference_mean, atol=1e-6, rtol=1e-6)
torch.testing.assert_close(
    fixed_posterior.distribution.covariance_matrix,
    reference_covariance,
    atol=1e-6,
    rtol=1e-6,
)
torch.testing.assert_close(
    noisy_posterior.variance,
    fixed_posterior.variance + noise,
    atol=1e-6,
    rtol=1e-6,
)
```

3 点の予測平均は `[3, 1]`、共分散は `[3, 3]` です。ここで検査しているのは、固定した kernel とノイズにおける平均・共分散および独立な観測ノイズの加算です。任意の kernel、学習結果、誤差分布の妥当性を認定する例ではありません。

`posterior(..., observation_noise=False)` は潜在応答、`True` または Tensor はノイズ付き観測の予測です。ただし未知の場所の測定分散は、訓練分散を渡しただけで自動的に同定されません。確認した BoTorch 0.18.1 の fixed-noise 経路では、`True` は訓練分散の平均を使います。場所ごとの既知分散を指定したい場合は Tensor を使い、出力 transform が有効ならその Tensor が変換後のスケールである点に注意してください。上の例は `outcome_transform=None` のため両スケールが一致します。

## 所有権・データ保持・適用上の境界

| 区分 | このモデルでの責務 |
| --- | --- |
| Delegated | BoTorch・GPyTorch の確率モデル、尤度、変換、周辺尤度計算、最適化、予測 |
| Extended | `SingleTaskGP` の公開入口に raw データ保持・学習 capability を追加 |
| Implemented | `RawDataMixin` のコピー保持、MLL オブジェクト構成、fit への接続と対応検査 |
| Approximated | 上流の線形代数・分散計算の設定に応じた数値計算。robotorchan 固有の GP 近似ではない |
| Unsupported | この Exact MLL を Variational・Fully Bayesian・DeepGP に一律適用すること |
| Unverified | あらゆる条件での収束、較正、hyperparameter の同定、大域最適性 |

[RawDataMixin](../../src/robotorchan/models/base.py) はコピーを非永続 buffer として登録します。`raw_train_X`・`raw_train_Y`・`raw_train_Yvar` は dtype・device の変更には追従しますが、`state_dict` には含まれません。また `condition_on_observations()` や `fantasize()` の後の最新訓練状態へ書き換える目的のデータではありません。現在の条件付け状態が必要なら BoTorch の `train_inputs`・`train_targets` を確認してください。

## 検証と読み進め方

[掲載例テスト](../../tests/docs/test_theory_guide_core_model_examples.py)はこの Markdown の Python を直接実行します。[独立 Cholesky 参照](../../tests/models/standard/test_phase7_exact_gp_reference.py)は別経路の RBF 行列による条件付き平均・共分散を比較し、[fit 契約](../../tests/models/contracts/test_exact_gp_fit.py)と[raw データ来歴](../../tests/models/contracts/test_raw_data_provenance.py)はそれぞれの責務を確認します。過去の[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)も参照できますが、現在の実行結果とは区別してください。テストの存在、実行成功、数値照合、数学的証明は別の水準です。

次は [Kernel の実装ガイド](03_kernel.md)で共分散の選択を学び、[BO のガイド](01_bayesian_optimization.md)で予測を意思決定へ接続してください。使い方は[標準モデル](../models/standard.md)、動く一連の例は [GP Notebook](../../examples/notebooks/01_single_task_gp.ipynb)で確認できます。GP の中心的な推論は上流へ委譲されるため、数式のどの部分がモデル定義で、どこから条件付け・最適化になるかを区別してソースを読むことが重要です。

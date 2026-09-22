# robotorchan

[![CI](https://github.com/tanakakao/robotorchan/actions/workflows/ci.yml/badge.svg)](https://github.com/tanakakao/robotorchan/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

`robotorchan` は [BoTorch](https://botorch.org/) をベースにした、ベイズ最適化・Active Learning 向けの研究開発用 Python ライブラリです。

BoTorch-native なモデル構成を維持しながら、raw training data の保持、`make_mll()` などの共通インターフェースと、高次元・Mixed・Robust・Multi-task / Multi-output 問題向けのモデルや探索戦略を提供します。

## 主な特徴

- **BoTorch-native**: BoTorch / GPyTorch のモデル・posterior・acquisition function と組み合わせやすい設計
- **統一モデル API**: raw training data、`supports_mll`、`make_mll()` などを共通化
- **幅広い surrogate**: 標準 GP、Mixed、Multi-Fidelity、Multi-task / Multi-output、Robust / Noise / Input uncertainty をサポート
- **高次元・表現力**: PCA / PLS / Random Projection、AE / VAE、SAAS、DeepGP、Infinite-width BNN GP、Spectral Mixture GP を提供
- **Non-GP surrogate**: Random Forest、Extra Trees、Gradient Boosting 系を共通の探索フローで利用可能
- **Active Learning 拡張**: variance / standard deviation、Straddle、Boundary Variance、EPIG などを提供
- **探索戦略の分離**: REMBO、HeSBO、ALEBO、TuRBO、BAxUS、tree ensemble search などを `SearchStrategy` として利用可能
- **実行可能な資料**: モデル選択ガイド、理論ドキュメント、Jupyter Notebook を同じリポジトリで管理

## Quick Start

開発版をインストール:

```bash
git clone https://github.com/tanakakao/robotorchan.git
cd robotorchan
pip install -e .
```

最小の GP fitting / posterior prediction:

```python
import torch
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import SingleTaskGP

torch.set_default_dtype(torch.double)

train_X = torch.rand(20, 2)
train_Y = torch.sin(2 * torch.pi * train_X[:, :1]) + 0.2 * train_X[:, 1:2]

model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
fit_gpytorch_mll(model.make_mll())

posterior = model.posterior(torch.rand(5, 2))
print(posterior.mean)
```

より詳しいモデル選択と実行例は [モデルガイド](docs/models.md) と [examples](examples/README.md) を参照してください。

## ドキュメント

目的に応じて次の順に進むと、モデル選択から理論、実行例まで辿れます。

1. **モデルを選ぶ:** [モデル概要・使い所ガイド](docs/models.md)
2. **理論を確認する:** [理論ガイド](docs/theory/README.md)
3. **コードを動かす:** [Examples / Notebook一覧](examples/README.md)

より専門的な課題では次のドキュメントを参照してください。

- **理論から理解する:** [`docs/theory/README.md`](docs/theory/README.md)
- **獲得関数を理解する:** [`docs/theory/acquisition/README.md`](docs/theory/acquisition/README.md)
- **獲得関数・探索を使う:** [`docs/optimization/README.md`](docs/optimization/README.md)
- **どのモデルを選ぶか:** [`docs/models.md`](docs/models.md)
- **高次元モデルを選ぶ:** [`docs/models/high_dimensional_model_selection.md`](docs/models/high_dimensional_model_selection.md)
- **Robust / Noise / Uncertaintyを選ぶ:** [`docs/models/robust_noise.md`](docs/models/robust_noise.md)
- **高次元出力モデルを選ぶ:** [`docs/models/high_dimensional_outputs.md`](docs/models/high_dimensional_outputs.md)
- **実際にどう使うか:** [`examples/README.md`](examples/README.md)
- **高次元の探索戦略:** [`docs/optimization/high_dimensional_search.md`](docs/optimization/high_dimensional_search.md)
- **設計方針・内部構造:** [`docs/development/architecture.md`](docs/development/architecture.md)
- **リリース手順:** [`docs/development/releasing.md`](docs/development/releasing.md)

モデルごとの実行可能な Jupyter Notebook は [`examples/notebooks/`](examples/notebooks/) にあります。

ベイズ最適化を初めて学ぶ場合は、[`ベイズ最適化とは`](docs/theory/01_bayesian_optimization.md) → [`Gaussian Process`](docs/theory/02_gaussian_process.md) → [`Kernel`](docs/theory/03_kernel.md) → [`Acquisition Function`](docs/theory/04_acquisition_function.md) → [`Mixed Variables`](docs/theory/05_mixed_variables.md) → [`Multi-Fidelity`](docs/theory/06_multi_fidelity.md) → [`Multi-task / Multi-output`](docs/theory/07_multitask_multioutput.md) → [`High-dimensional GP`](docs/theory/08_high_dimensional_gp.md) → [`Variational GP`](docs/theory/09_variational_gp.md) → [`Preference Learning`](docs/theory/10_preference_learning.md) → [`Structured Output`](docs/theory/11_structured_output.md) → [`Hierarchical / Contextual GP`](docs/theory/12_hierarchical_contextual_gp.md) → [`Model Selection`](docs/theory/13_model_selection.md) までを基礎編として読み、その後は課題に応じて [`Robust Gaussian Process`](docs/theory/14_robust_gaussian_process.md) 〜 [`Advanced High-dimensional GP Models`](docs/theory/21_advanced_high_dimensional_models.md)、[`Expressive GP Models`](docs/theory/22_expressive_gp.md)、[`Non-GP Surrogates`](docs/theory/23_non_gp_surrogates.md) を参照してください。獲得関数の詳細理論は [`docs/theory/acquisition/`](docs/theory/acquisition/README.md) に独立して整理しています。

## インストール

現在はリポジトリから開発版を利用します。

```bash
git clone https://github.com/tanakakao/robotorchan.git
cd robotorchan
pip install -e .
```

Fully Bayesian SAASを利用する場合:

```bash
pip install -e ".[fully-bayesian]"
```

Notebook 例も動かす場合:

```bash
pip install -e ".[examples]"
```

Fully Bayesian SAAS の例も動かす場合:

```bash
pip install -e ".[examples,fully-bayesian]"
```

## モデル共通規約

多くの supervised wrapper は、BoTorch の予測挙動を維持したまま次の情報を保持します。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.raw_data_names
model.supports_mll
model.make_mll()
```

`raw_*` は **wrapper 構築時に渡された入力のスナップショット** です。

`condition_on_observations()` や `fantasize()` 後の現在の学習状態を表すものではありません。現在状態には BoTorch ネイティブの `train_inputs` / `train_targets` を使用してください。

## 利用可能なモデル

robotorchan の public model API は多数の派生モデルを含むため、README ではモデルファミリーを示します。完全な public API と各モデルの Guide / Theory / Notebook 対応は [モデルガイド](docs/models.md) と [model coverage](docs/model_coverage.json) を参照してください。

| ファミリー | 代表的なモデル・機能 |
|---|---|
| Standard / Mixed / Multi-Fidelity | `SingleTaskGP`, `MixedSingleTaskGP`, `SingleTaskMultiFidelityGP` |
| Multi-task / Multi-output | `MultiTaskGP`, `KroneckerMultiTaskGP`, `ModelListGP`, Mixed / heterogeneous variants |
| High-dimensional / Reduction | PCA, PLS, Random Projection, AE / VAE, Joint Encoder, SAAS, MAP-SAAS, ALEBO |
| Robust / Noise / Uncertainty | Student-t, contaminated observation, heteroskedastic, replicate noise, uncertain input, nonstationary, relevance pursuit |
| Expressive GP | DeepGP, Infinite-width BNN GP, Spectral Mixture GP and Mixed / MultiTask variants |
| Structured / Contextual | Pairwise, HigherOrder, LatentKronecker, hierarchical / contextual GP |
| Non-GP surrogate | Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting |

特に MultiTask / Kronecker、Mixed、reduced、robust、expressive 系には複数の組み合わせがあります。クラス名をREADMEへ重複して固定せず、`robotorchan.models.__all__` を public API の source of truth とし、[model_coverage.json](docs/model_coverage.json) でドキュメント対応を追跡します。

### Acquisition / Active Learning

BoTorch 標準の acquisition function をそのまま利用できることを基本とし、robotorchan 固有の Active Learning / sampling 拡張として `PosteriorVariance`, `PosteriorStd`, `Straddle`, `RandomizedStraddle`, `BoundaryVariance`, `ExpectedPredictiveInformationGain`、Thompson candidate selection などを提供します。

理論は [Acquisition Function Theory](docs/theory/acquisition/README.md)、利用方法と適用範囲は [Optimization guides](docs/optimization/README.md) と [Acquisition integration status](docs/optimization/acquisition-status.md) を参照してください。

## 特殊な学習方法

すべてのモデルが通常の Exact MLL 学習を使うわけではありません。

### Variational GP

`SingleTaskVariationalGP.make_mll()` は `VariationalELBO` を返します。minibatch 学習では全データ件数を `num_data` として扱います。

### Fully Bayesian SAAS

Fully Bayesian SAAS は MLL fitting ではなく NUTS を使います。

```python
from botorch.fit import fit_fully_bayesian_model_nuts

fit_fully_bayesian_model_nuts(model)
```

このため `supports_mll=False` です。

### PairwiseGP

Preference data を通常の回帰 target として扱わず、`datapoints` と `comparisons` を使います。

### LatentKroneckerGP

`train_X` / `train_Y` に加えて、時間・波長・位置などの出力軸 `train_T` を明示的に持ちます。


## 高次元の探索戦略

robotorchan は surrogate model と acquisition function に加えて、acquisition function をどの空間・方法で探索するかを `SearchStrategy` として分離します。

```python
from botorch.acquisition.analytic import LogExpectedImprovement
from robotorchan.optim import HeSBOStrategy

acq = LogExpectedImprovement(model=model, best_f=train_Y.max())
strategy = HeSBOStrategy(bounds, embedding_dim=5, seed=0)
result = strategy.optimize(acq, q=1)

candidate = result.candidates
acquisition_value = result.acquisition_value
```

利用可能な主な strategy は `OriginalSpaceStrategy`、`RandomSearchStrategy`、`LatentSpaceStrategy`、`REMBOStrategy`、`HeSBOStrategy`、`ALEBOStrategy`、`TuRBOStrategy`、`BAxUSStrategy`、`BAxUSThompsonSamplingStrategy`、`TreeEnsembleSearchStrategy` です。ALEBO 用には full Mahalanobis metric を持つ `ALEBOGP` も提供します。

`SearchResult.candidates` は常に public/original input space の候補を返します。`acquisition_value` は選択された joint q-batch に対する scalar tensor です。TuRBO / BAxUS は stateful strategy のため、目的関数を評価した後に観測値を strategy へ戻します。

詳細、各 strategy の役割、BAxUS の state 更新、benchmark の設計は [`docs/optimization/high_dimensional_search.md`](docs/optimization/high_dimensional_search.md) を参照してください。

## 設計目標

- **BoTorch-native**: 可能な限り標準の BoTorch / PyTorch object を利用する
- **薄い wrapper**: upstream のモデル挙動を維持し、robotorchan 固有機能だけを追加する
- **Research-friendly**: 新しい獲得関数や最適化手法を試しやすくする
- **Composable**: model / acquisition / objective / transform / optimizer を分離する
- **Tested**: tensor shape、API、数値挙動を自動テストする
- **Clean-room implementation**: 公開アルゴリズム・公開 API を基にゼロから実装する

## 対応環境

- Python >= 3.11
- BoTorch >= 0.18.1, < 0.19


## 開発・引用・ライセンス

- 開発への参加方法: [CONTRIBUTING.md](CONTRIBUTING.md)
- 変更履歴: [CHANGELOG.md](CHANGELOG.md)
- 研究・論文での引用情報: [CITATION.cff](CITATION.cff)
- ライセンス: [BSD-3-Clause](LICENSE)

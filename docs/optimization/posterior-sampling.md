# 事後分布のサンプリング

BoTorch を基盤とし、モデルの名前ではなく posterior の契約から sampler を選びます。

## Posterior と sampler の対応

| Posterior の意味 | BoTorch sampler | 代表モデル |
| --- | --- | --- |
| Gaussian | `SobolQMCNormalSampler` | SingleTask、MultiTask、Kronecker、Mixed、多忠実度、reduced、robust Gaussian、NGBoost adapter |
| 経験 ensemble | `IndexSampler` | RandomForest、ExtraTrees、GradientBoosting、HistGradientBoosting |
| 確率的・経験的軌跡 | `StochasticSampler` | DeepGP |
| 実際の `PosteriorList` | `ListSampler` | 返り値が `PosteriorList` の場合のみ |

`ModelListGP` という名前だけで `ListSampler` を選びません。同種の Gaussian submodel なら joint Gaussian の多出力 posterior が返り、normal sampler が適切です。

`make_model_sampler` は capability metadata に基づく統合 helper です。獲得関数のトップレベル公開 API ではなく、BoTorch sampler の直接利用を置き換えません。

## Joint sampling

多出力・MultiTask・Kronecker の共分散を持つ posterior は joint にサンプルします。出力・タスクを独立 sampler へ分割しないでください。

`DeepGPPosterior` は予測軌跡全体を保持します。`StochasticSampler` が `posterior.rsample()` に委譲し、Gaussian や QMC-normal とみなさず軌跡単位で再抽出します。

## 形状と再現性

BoTorch の形状契約に従います。

```text
sample_shape x batch_shape x q x output_shape
```


Gaussian normal sampler は共通乱数のため base sample を再利用できます。経験 ensemble と確率的 posterior は各 sampler 固有の契約に従い、同一動作を仮定しません。

## 候補に対する勾配

sampling の互換性と勾配最適化の互換性は別です。Gaussian GP、Mixed の連続次元、MultiTask、Kronecker、reduced、robust Gaussian、現在の DeepGP 軌跡はテスト対象経路で候補勾配を保ちます。scikit-learn tree ensemble と NGBoost は NumPy・外部推定器を経由するため、候補入力に対して微分できません。モデルの境界に合う探索戦略を使ってください。

## MC 獲得関数

モデル出力の不確実性を積分します。

```text
model -> posterior -> MCSampler -> objective / outcome constraint -> acquisition
```


出力制約はモデルの sample から評価します。候補最適化が担う既知の入力制約とは別です。

## 入力摂動

候補摂動は posterior sampler の責務ではありません。

```text
candidate X
-> input perturbation scenarios
-> model posterior
-> posterior sampling
-> risk aggregation across scenarios
```


scenario 軸と posterior MC 軸を区別し、Expectation・MeanVariance・CVaR などは posterior sampling 後に scenario 軸を集約します。

## Thompson sampling

有限プールからの候補生成は MC 積分とは別です。`select_thompson_candidates` を通じて BoTorch `MaxPosteriorSampling` に委譲します。

```text
model -> posterior over choices -> MaxPosteriorSampling -> selected candidates
```


互換 Gaussian・経験 ensemble posterior に使えます。多出力では明示的なスカラー化 objective が必要です。

## Custom sampler

BoTorch の既存機能では正しく sample できない統計的な意味があり、それを定義・検証できる場合に限り追加します。現在の対象モデルでは独自 sampler は不要で、Gaussian、経験 ensemble、DeepGP 軌跡、有限プール Thompson は上流の機能で扱います。

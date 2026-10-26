# Non-GP Surrogates：経験分布をBoTorchへ接続する

## 1. GP以外のモデルがBayesian Optimizationに参加する条件

BOに必要なのは必ずしもGPではなく、獲得関数が消費できる予測値や**sampleable predictive representation**である。[理論第23章](../../../theory/23_non_gp_surrogates.md)では木・boosting・NGBoostにおけるuncertaintyの意味を区別する。

同じ`posterior()`インターフェースでも、**GPの条件付き確率posterior**と**木ごとの予測の経験分布**は数学的に同義ではない。

## 2. モデルとposteriorの実装対応

| Family | 公開API | サンプルの意味 |
| --- | --- | --- |
| Random Forest | `RandomForestSurrogate` | fitted treeごとの予測 |
| Extra Trees | `ExtraTreesSurrogate` | 強い分割ランダム性を含むtreeごとの予測 |
| Gradient Boosting | `GradientBoostingSurrogate` | bootstrapごとに学習した**完全なboosting model**の予測 |
| Histogram Boosting | `HistGradientBoostingSurrogate` | bootstrapごとの完全なhistogram boosting model |
| NGBoost | `NGBoostSurrogate` | 学習された条件付き予測分布からのサンプル |

**Extended:** robotorchanは`src/robotorchan/models/non_gp/`にあるsklearnベースのモデル出力をBoTorchの`EnsemblePosterior`へ接続する。`fit()`で明示的に学習し、GPのMLLは使わない（`supports_mll=False`）。NGBoostはオプション依存で、初期契約は単一出力の数値回帰である。

重要な誤解として、boostingの逐次stage `h_1,h_2,…` を独立posterior sampleにしてはいけない。stageは残差を順番に修正する加法項であり、robotorchanはbootstrap単位の**完成したモデル**をmemberにしている。

## 3. 予測から候補探索まで

```text
train_X / train_Y → non-GP surrogate.fit()
                             ↓
           model.posterior(candidate_X)
                             ↓
                  EnsemblePosterior
                             ↓
               compatible MC acquisition
                             ↓
               gradient-free search strategy
```

posteriorの軸は概念的に`… × ensemble_samples × q × outputs`であり、sample軸と`q`軸の取り違えはacquisitionの意味を壊す。sklearn経由のcandidate評価は一般にCPU/numpyを介しており、入力へのPyTorch gradientは期待できない。そのため`optimize_acqf`の勾配ベース経路ではなく、`TreeEnsembleSearchStrategy`等の勾配不要探索を検討する。

## 4. 公開APIの入口例

```python
import torch
from robotorchan.models import RandomForestSurrogate

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = train_X[:, :1].square()
model = RandomForestSurrogate(train_X=train_X, train_Y=train_Y)
model.fit()
posterior = model.posterior(train_X[:2])
print(posterior.mean.shape)
```

上記は設計意図を示す例で、今回の作業で実行したものではない。具体的なconstructorとposterior contractは実装・Notebookで検収する。

## 5. 制限と検証

MC qEI/qEHVIは条件を満たす場合の候補であり、全獲得関数が`EnsemblePosterior`を解釈できるという主張ではない。Gaussian analytic EIは一般に経験ensembleの仮定に合わない。tree disagreementはcalibrated epistemic posteriorではなく、NGBoostの予測分散からepistemic/aleatoricを自動的に分離できるわけでもない。

詳細：[Non-GPモデル](../../../models/non_gp.md)、[Non-GP Notebook](../../../../examples/notebooks/25_non_gp_surrogates.ipynb)、[Acquisition Foundation](04_acquisition_function.md)、[旧Mapping](../models.md)、[検証記録](../verification.md)。

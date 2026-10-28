# Student-t 観測 GP モデル

Student-t 尤度を使い、裾の重い観測残差を扱うモデルです。疎な外れ値補正や、入力依存の Gaussian 観測ノイズとは異なる仮定を置きます。

## 統計モデル
```text
f(x) ~ GP
y | f(x) ~ StudentT(df=nu, loc=f(x), scale=sigma)
```
コンストラクタでは観測分散が存在するよう `df > 2` を要求します。Student-t 尤度による変分推論を使い、Gaussian 尤度の厳密な周辺尤度では学習しません。

## 公開モデル

- `StudentTSingleTaskGP`：連続入力の単一タスク・スカラー回帰。
- `MixedStudentTSingleTaskGP`：`cat_dims` と混合共分散による連続・カテゴリ入力。
- `StudentTMultiTaskGP`：`task_feature` とタスク共分散による long-format の複数タスク。
- `MixedStudentTMultiTaskGP`：long-format の複数タスクと連続・カテゴリ入力。

タスク列とカテゴリのデータ列は異なる構造です。タスク ID やカテゴリ符号を連続量として解釈せず、入力構造に合うモデルを選んでください。

## 学習と予測分布
```python
import torch
from robotorchan.models import StudentTSingleTaskGP

train_X = torch.rand(20, 2, dtype=torch.double)
train_Y = torch.sin(train_X[:, :1])

model = StudentTSingleTaskGP(
    train_X,
    train_Y,
    num_inducing=16,
    df=4.0,
)
loss = model.training_loss()
posterior = model.posterior(train_X[:3])
```
`training_loss()` は Student-t 尤度の期待対数尤度と変分 KL 項を組み合わせます。微分可能な損失を適切な PyTorch optimizer で最小化してください。`supports_mll = False` で、`make_mll()` は厳密 Gaussian MLL を返さず例外になります。BoTorch 互換の `posterior()` は **潜在応答**の分布であり、Student-t の将来観測分布ではありません。

元の訓練入力・出力を保持し、state dictionary の読み込み時には変分キャッシュを消去します。

## 適用範囲と解釈

裾の重い尤度はモデルの仮定であり、すべてのデータで外れ値への頑健性を保証するものではありません。保留データで Gaussian モデルや他の robust モデルと予測性能・較正を比較してください。入力摂動と CVaR・最悪値による獲得効用の集約は別の最適化上の課題です。

モデル選択は [Robust / Noise](robust_noise.md) を参照してください。

# Joint neural GP の学習契約

`JointEncoderGP`、`HybridAutoEncoderGP`、`JointVAEGP` は、frozen reducerを使う `ReducedGP` 系とは異なり、encoderをGPと同時に最適化します。

## 推奨API

Joint系では `training_loss()` を最適化してください。

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(100):
    optimizer.zero_grad()
    loss = model.training_loss()
    loss.backward()
    optimizer.step()
```

モデルごとのlossは以下です。

| Model | `training_loss()` |
| --- | --- |
| `JointEncoderGP` | negative exact GP MLL |
| `HybridAutoEncoderGP` | negative GP MLL + `reconstruction_weight * reconstruction_loss` |
| `JointVAEGP` | negative GP MLL + `reconstruction_weight * reconstruction_loss + beta * KL` |

`HybridAutoEncoderGP` と `JointVAEGP` を含む共同学習モデルは、共通の `training_loss()` を使用します。

## `make_mll()` との違い

`make_mll()` はrobotorchanのExact GP wrapper共通APIとして、GPyTorchのMLLオブジェクトを返します。`JointEncoderGP` の目的はGP MLLだけなので概念的に一致しますが、Hybrid / JointVAEではreconstructionやKLがMLLの外側にあります。

したがって次のコードはHybrid / JointVAEの完全な学習目的にはなりません。

```python
fit_gpytorch_mll(model.make_mll())
```

Joint系をモデル種別に依存せず学習する場合は `training_loss()` を使用してください。

## Frozen neural modelとの違い

`AutoEncoderGP`、`VAEGP`、`SupervisedAutoEncoderGP`、`SupervisedVAEGP` はreducerを事前学習してfreezeした後にGPを学習します。この系統では通常どおり `make_mll()` と `fit_gpytorch_mll()` を使用できます。

```text
Frozen neural:
X[,Y] -> representation pretraining -> freeze -> GP fitting

Joint neural:
X -> encoder -> GP -> Y
     ^          |
     +----------+
       training_loss
```

この区別により、benchmarkや将来のtrainer実装でモデル固有の `hybrid` / `joint_vae` 分岐を持たず、Joint系を一つの学習契約で扱えます。

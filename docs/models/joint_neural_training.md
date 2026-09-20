# Joint neural GP と Deep Kernel Learning

`JointEncoderGP` は、ニューラル特徴写像 $\\phi_\\theta(x)$ の出力にExact GPを置き、GP MLLからencoderとGPを同時最適化するDeep Kernel Learning (DKL) の基本実装です。`DeepKernelGP` という重複クラスは追加せず、robotorchanでは `JointEncoderGP` をこの役割に使用します。

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


## DKLとしての特徴抽出器

既定では `hidden_dims` と `activation` からMLP encoderを構築します。`latent_dim` は入力次元以下に限定されず、圧縮・同次元・拡張のいずれも利用できます。

任意のPyTorch moduleを `feature_extractor` として渡すこともできます。出力最終次元は `latent_dim` と一致させてください。

```python
feature_extractor = torch.nn.Sequential(
    torch.nn.Linear(train_X.shape[-1], 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, 8),
)

model = JointEncoderGP(
    train_X,
    train_Y,
    latent_dim=8,
    feature_extractor=feature_extractor,
)
```

このmoduleはmodelのsubmoduleとして登録されるため、`model.parameters()` を使った `training_loss()` の最適化でGPと同時に更新されます。入力標準化はmodel側で行い、その後のテンソルがfeature extractorへ渡されます。

この設計により、DKLのためだけに別の互換APIを増やさず、将来CNNやTransformerなど別の特徴抽出器を接続する場合も同じ学習契約を維持できます。


## BoTorch acquisitionとの統合

`JointEncoderGP` は候補点を元の入力空間で受け取り、内部で特徴写像を適用します。そのため、獲得関数側でlatent空間へ手動変換する必要はありません。

Phase 3では、posteriorだけでなく `qLogExpectedImprovement`、`qLogNoisyExpectedImprovement`、`qUpperConfidenceBound` と `optimize_acqf` の元入力空間での動作をintegration testで固定しています。また、sequential BOで重要な `condition_on_observations()` と `fantasize()` についても元入力次元を維持する契約をテストします。

したがってBO loopでは通常のBoTorch modelと同様にmodelを獲得関数へ直接渡します。encoderを外側から呼び出して候補点を変換しないでください。


## DKL variant coverage

DKLの基本契約はsingle-outputの `JointEncoderGP` だけでなく、既存のjoint representation系へ揃えています。

- `MixedJointEncoderGP`: continuous featuresのみをencoderへ渡し、categorical featuresはnative mixed kernelへ保持します。
- `JointEncoderMultiTaskGP`: task featureをencoderから除外し、data featuresだけを学習表現へ写像します。
- `JointEncoderKroneckerMultiTaskGP`: block-design multi-output / multitask dataをjoint neural representationで扱います。

MultiTask系でも `feature_extractor` に任意の `nn.Module` を指定でき、`latent_dim` は圧縮に限定されません。task identityやcategorical identityをニューラルencoderへ無条件に埋め込まず、GP側の構造として保持することをrobotorchanのDKL variant共通方針とします。

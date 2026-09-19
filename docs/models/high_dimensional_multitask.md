# 高次元 MultiTask GP 実行ガイド

高次元入力と複数タスクを同時に扱う場合は、task identityを保ったままdata featureだけを低次元化します。

## long-format

`PCAMultiTaskGP` などのlong-formatモデルでは、最後の列など明示した `task_feature` をreducerから除外します。

```python
import torch
from botorch.fit import fit_gpytorch_mll

from robotorchan.models import PCAMultiTaskGP

torch.set_default_dtype(torch.double)
torch.manual_seed(0)

data_X = torch.rand(20, 12)
task0 = torch.zeros(20, 1)
task1 = torch.ones(20, 1)
train_X = torch.cat(
    (
        torch.cat((data_X, task0), dim=-1),
        torch.cat((data_X, task1), dim=-1),
    )
)
base = torch.sin(2 * torch.pi * data_X[:, 0])
train_Y = torch.cat((base, 0.8 * base + 0.2)).unsqueeze(-1)

model = PCAMultiTaskGP(
    train_X,
    train_Y,
    task_feature=-1,
    n_components=3,
)
fit_gpytorch_mll(model.make_mll())

test_X = train_X[:4]
posterior = model.posterior(test_X)
print(posterior.mean)
```

公開APIには常に元空間のXを渡します。内部ではdata featureのみPCAへ通し、task featureを潜在表現の後ろへ再結合します。

## Kronecker形式

Kronecker MultiTaskではtaskがY列に対応するため、Xにtask featureを追加しません。入力reducerはX全体を低次元化できます。

## Mixed入力

Mixed版では continuous feature のみを削減し、categorical feature はそのまま保持します。long-format の場合はさらに task feature も保持します。

現在 public な共通基盤は `MixedReducedMultiTaskGP` と
`MixedReducedKroneckerMultiTaskGP` です。SingleTask 側に
`MixedPCAGP` / `MixedPLSGP` / `MixedRandomProjectionGP` などの named wrapper が
存在しても、それを根拠に `MixedPCAMultiTaskGP` のような未実装 class が存在すると
解釈してはいけません。必要な reducer は共通 Mixed MultiTask 基盤へ明示的に構成します。

## 実装対応表

| reducer family | SingleTask | MultiTask | Kronecker | Mixed SingleTask | Mixed MultiTask |
| --- | --- | --- | --- | --- | --- |
| PCA / PLS / Random Projection | named | named | named | named | common reduced base |
| AE / VAE | named | named | named | named | common reduced base |
| Supervised AE / VAE | named | named | named | named | common reduced base |
| Joint Encoder / Hybrid AE / Joint VAE | named | named | named | named | common reduced base |

ここで `named` は public class があること、`common reduced base` は
`MixedReducedMultiTaskGP` / `MixedReducedKroneckerMultiTaskGP` で構成することを表します。
ドキュメントの便宜のためだけに named cross-product class は追加しません。

## モデル選択

線形な低次元構造が想定できる場合はPCAを基準にし、Yに関連する射影が必要ならPLS、安価な対照モデルとしてRandom Projectionを使います。非線形構造が必要な場合はAE/VAE系、目的変数に合わせて表現を学習する場合はSupervised/Joint系を比較します。

高次元だが元特徴量のごく一部だけが効く仮定なら、削減モデルとは別系統のSAASが候補です。探索空間自体に低次元線形構造を仮定するALEBOもsurrogate reductionとは別レイヤーとして扱います。

## Benchmark

`benchmarks/high_dimensional_multitask.py` でlong-formatの線形reducer群を共通条件で比較できます。benchmark結果だけでモデルの優劣を固定せず、実データでも同じsplit・seed・評価指標で比較してください。

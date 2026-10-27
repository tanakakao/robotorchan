# MultiTask / MultiOutput GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/07_multitask_multioutput.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

複数タスクの関係を使って情報を共有する。long-format multi-taskでは`train_X`の一列がtask IDを表し、`train_Y`には観測行ごとに一つの出力がある。これと同じ入力点で複数出力を観測するblock-design Kroneckerモデルは、データ構造やposteriorのshapeが異なる。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/standard/multitask.py`の`MultiTaskGP`はBoTorch `MultiTaskGP`からlong-format契約とtask covarianceを継承し、raw-data/MLL便宜機能を付加する。
- **Delegated:** task-conditioned GPのposterior計算はBoTorch/GPyTorch。
- **Extended:** 同モジュールの`MixedMultiTaskGP`は`task_feature`を構造列に維持し、`cat_dims`を設計側のカテゴリ列として扱う。
- `KroneckerMultiTaskGP`は別のblock-design実装であり、long-formatへの単純な別名ではない。

## 最小の利用例

```python
import torch
from robotorchan.models import MultiTaskGP

X = torch.tensor([[0.1, 0.0], [0.4, 0.0], [0.1, 1.0], [0.4, 1.0]], dtype=torch.double)
Y = torch.tensor([[0.2], [0.5], [0.3], [0.7]], dtype=torch.double)
model = MultiTaskGP(train_X=X, train_Y=Y, task_feature=1)
print(model.posterior(X[:2]).mean)
```

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

既存のKronecker数値参照は固定したinput/task covarianceブロックの配列順とPSD等を対象とし、学習後posteriorの全面的な同値性を保証しない。task-indexing、データの配置、出力形状、未知taskへの一般化の制約を個別に確認する。Notebook: [Multitask](../../../../examples/notebooks/04_multitask_gp.ipynb)。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。

## 追加：相関構造・出力軸・モデル合成の使い分け

複数出力GPで重要なのは、同じモデル数を持つことではなく、異なる出力間に共分散を置くかどうかである。典型的なICM表現では次のような積構造を考える。

$$
\operatorname{Cov}(f_t(x),f_{t'}(x'))=k_X(x,x')B_{tt'}.
$$

ただしこの式は概念的な構造であり、すべてのBoTorch multitask kernelや尤度をこの単一式で定義するものではない。MultiTaskGPはtask featureを入力の構造列として使い、train_Xの各行がどのtaskの観測かを表すlong-formatである。KroneckerMultiTaskGPは同じ入力で全taskを観測するblock-designでtrain_Yが[n,m]になる。後者は前者の別名ではなく、taskの位置とposteriorの形状の解釈が違う。

| モデル | 教師データ・出力軸 | 相関と実装所有権 |
| --- | --- | --- |
| MultiTaskGP | train_Xにtask列、train_Yは各観測1値 | BoTorch MultiTaskGPのtask covarianceへ委譲。robotorchanはraw tensorとexact MLLを補う |
| KroneckerMultiTaskGP | train_Xは設計列、train_Yは[n,m] | BoTorch KroneckerMultiTaskGPへ委譲。raw_train_Yvar=Noneで、前者のtrain_Yvar契約と異なる |
| ModelListGP | 子GPごとに訓練X/Yを保持 | BoTorch ModelListGPを継承。子モデルの独立性を前提とする合成であり、cross-output covarianceを学習するMultiTaskGPとは異なる |
| HeterogeneousModel | 子modelのoutput数に応じたglobal index | 回帰posteriorと分類latent posteriorの意味を混ぜないため、container自身の共通posterior()を持たない |

MixedMultiTaskGPではtask_featureは構造列であり、cat_dimsから除外する。categorical covarianceは設計列にのみ適用する。ModelListGP.make_mll()はSumMarginalLogLikelihood(self.likelihood,self)を返す一方、HeterogeneousModelは子ごとのentry_make_mll / entry_fitを提供する。異種モデル群全体に単一のGaussian posteriorや単一MLLがあると仮定しない。

### 異種モデルの出力所有権を確認する

~~~python
import torch
from robotorchan.models import HeterogeneousModel, SingleTaskGP
from robotorchan.models.classification import BinarySingleTaskGPClassifier

train_X = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.double)
strength_Y = torch.tensor([[1.0], [1.2], [1.5], [1.3]], dtype=torch.double)
pass_Y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
regression_model = SingleTaskGP(train_X=train_X, train_Y=strength_Y)
binary_classifier = BinarySingleTaskGPClassifier(train_X=train_X, train_Y=pass_Y)
models = HeterogeneousModel(
    regression_model,
    binary_classifier,
    names=("strength", "pass"),
    output_names=("strength", "pass"),
)
X = torch.tensor([[0.5]], dtype=torch.double)
assert models.output_owner(1) == (1, 0)
pass_probs = models.entry_predict_proba("pass", X)
strength_posterior = models.entry_posterior("strength", X)
assert pass_probs.shape[-1] == 2
assert strength_posterior.mean.shape[-1] == 1
~~~

この例はconstructor・予測APIと出力所有権の接続を示すもので、学習前の予測確率に実用的な校正精度があることを意味しない。output_ownerは(global index)を(entry index, child-local index)に写し、entry_predict_probaは分類子に対してのみ有効。分類モデルのposteriorがlatentであることを忘れて出力を目的値として結合しない。

根拠：src/robotorchan/models/standard/multitask.py、standard/model_list.py、models/heterogeneous.py。独立したcross-task covarianceの数値検算・multi-output acquisitionの最適化実行は今回未実施。

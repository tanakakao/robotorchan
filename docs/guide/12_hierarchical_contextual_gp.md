# Hierarchical / Contextual GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/12_hierarchical_contextual_gp.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

親変数の値で子変数の有効・無効が切り替わるconditional search spaceと、context間の情報を構造的に共有するcontextual GPを区別する。カテゴリ値を持つだけの通常Mixed入力と、入力の存在条件が変わるhierarchical変数は同じではない。

## 実装の責務と公開契約

- **Implemented/Extended（family別）:** `src/robotorchan/models/structured/hierarchical.py`には`HierarchicalConditionalKernelGP`系、`src/robotorchan/models/structured/contextual.py`には`SACGP`、`LCEAGP`、`LCEMGP`等のcontext系がある。
- **Delegated:** GPの基本的なposterior演算など、BoTorch/GPyTorchへ委譲する部分と、条件付きkernel・context embeddingを構築するlocal部分はコード経路ごとに分けて確認する。
- 同一のcontext IDを設計カテゴリ、task index、あるいはlatent embeddingとして扱うモデルを一括りにしない。

## 最小の利用例

```python
import torch
from robotorchan.models import MixedHierarchicalConditionalKernelGP

train_X = torch.tensor(
    [
        [0.0, 0.10, 0.0, 0.20],
        [1.0, 0.25, 0.0, 0.35],
        [0.0, 0.40, 1.0, 0.55],
        [1.0, 0.60, 1.0, 0.70],
        [0.0, 0.75, 0.0, 0.80],
        [1.0, 0.90, 1.0, 0.95],
    ],
    dtype=torch.double,
)
train_Y = 0.3 * train_X[:, :1] + train_X[:, 1:2] + 0.2 * train_X[:, 3:4]
dependencies = {2: {0: [1], 1: [3]}}
model = MixedHierarchicalConditionalKernelGP(
    train_X=train_X,
    train_Y=train_Y,
    hierarchical_dependencies=dependencies,
    cat_dims=[0],
    use_saas_prior=False,
    outcome_transform=None,
)
assert model.cat_dims == (0,)
assert model.raw_train_X.shape == train_X.shape
assert model.make_mll() is not None
```

この例で第2列（0-based index 2）は条件を切り替える親変数であり、
`dependencies`は親が0なら第1列、1なら第3列をactiveにする。
`cat_dims=[0]`は通常のカテゴリ設計列を表し、親列を含めない。
`make_mll()`の生成は学習収束やinactive値に対する不変性の検証ではない。

Contextual GPの`SACGP`は`train_Yvar`と`decomposition`を明示的に受け取る。
階層条件付きkernelの`hierarchical_dependencies`と同じAPIではない。


### Contextual Multi-Output：task列と設計カテゴリ列を分ける

LCEMGP系列ではtaskを区別する構造列と、設計条件のカテゴリ列は役割が異なる。
次のMixedLCEMGPはtask_featureを末尾列として指定し、cat_dimsには設計列だけを渡す。

```python
import torch
from robotorchan.models import MixedLCEMGP

train_X = torch.tensor(
    [
        [0.10, 0.0, 0.0],
        [0.20, 1.0, 0.0],
        [0.40, 0.0, 1.0],
        [0.55, 1.0, 1.0],
        [0.75, 0.0, 0.0],
        [0.90, 1.0, 1.0],
    ],
    dtype=torch.double,
)
train_Y = train_X[:, :1] + 0.25 * train_X[:, 1:2] + 0.4 * train_X[:, 2:3]
model = MixedLCEMGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,
    cat_dims=[1],
    outcome_transform=None,
)
assert model.cat_dims == (1,)
assert model.raw_task_feature == 2
assert model.make_mll() is not None
```

この例はtaskと設計カテゴリの分離、および学習目的関数を生成できることを示す。
context間の汎化や潜在embeddingの推定精度まで示すものではない。

## 検証・未検証事項

上の階層条件付きGPの例はモデル構築とMLL生成を示すものであり、fitやposteriorの数値検証は含まない。conditional inactive dimensionsの扱い、contextごとの入力構造、観測に存在しないcontextへの汎化条件はfamilyごとに制限がある。詳細は[モデル選択ガイド](../models/hierarchical_contextual.md)と[Hierarchical Notebook](../../examples/notebooks/12_hierarchical_gp.ipynb)、[Context Notebook](../../examples/notebooks/14_contextual_gp.ipynb)を参照する。

[旧Model Mapping](../../.github/project-records/theory-guide/mapping/models.md)および[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../models/README.md)を参照する。

## 条件付き入力とcontextのモデル化境界

hierarchical modelでは親変数pによって子変数x_jが有効になるかが変わる。これを表すhierarchical_dependenciesは親入力列と親カテゴリ値からactiveとなる子列群への辞書であり、単なるカテゴリ集合やtask_featureの別名ではない。

src/robotorchan/models/structured/hierarchical.pyのHierarchicalConditionalKernelGPはBoTorchHierarchicalConditionalKernelGPを継承する。hierarchical_dependencies、eval_hierarchical_features、separate_hierarchical_features、use_saas_prior、use_outputscale等を上流へ委譲し、robotorchanはraw tensor snapshotとExact GPのmake_mll契約を補う。

MixedHierarchicalConditionalKernelGPでは、各active blockに対してcategorical／continuous／両者のadditive + product covarianceを組むローカル_MixedHierarchicalConditionalKernelを利用する。親変数の列は構造的であるためcat_dimsと重ねられず、重複するとValueErrorとなる。hierarchical_dependenciesは入力変換後の列と整合させる必要がある。

Contextual GPは別familyであり、特定context間の共分散やembeddingで情報を共有するモデルである。親による子の無効化規則をcontext IDと取り違えず、SACGP・LCEAGP・LCEMGPでは個別にAPIを確認する。

**検証境界：** 上記はhierarchical.pyの継承・kernel構築・negative contractのソース照合。active/inactive invarianceの数値property、未観測contextへの予測の妥当性を一括認証しない。

## 数値 API と利用上の補足

- [Mixed Variables](../theory/05_mixed_variables.md)
- [Multi-task / Multi-output](../theory/07_multitask_multioutput.md)
- [High-dimensional GP](../theory/08_high_dimensional_gp.md)
- [Model Selection](../theory/13_model_selection.md)
- [Model documentation](../models/README.md)

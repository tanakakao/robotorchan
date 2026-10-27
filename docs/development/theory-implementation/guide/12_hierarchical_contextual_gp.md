# Hierarchical / Contextual GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/12_hierarchical_contextual_gp.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

親変数の値で子変数の有効・無効が切り替わるconditional search spaceと、context間の情報を構造的に共有するcontextual GPを区別する。カテゴリ値を持つだけの通常Mixed入力と、入力の存在条件が変わるhierarchical変数は同じではない。

## 実装の責務と公開契約

- **Implemented/Extended（family別）:** `src/robotorchan/models/structured/hierarchical.py`には`HierarchicalConditionalKernelGP`系、`src/robotorchan/models/structured/contextual.py`には`SACGP`、`LCEAGP`、`LCEMGP`等のcontext系がある。
- **Delegated:** GPの基本的なposterior演算など、BoTorch/GPyTorchへ委譲する部分と、条件付きkernel・context embeddingを構築するlocal部分はコード経路ごとに分けて確認する。
- 同一のcontext IDを設計カテゴリ、task index、あるいはlatent embeddingとして扱うモデルを一括りにしない。

## 最小の利用例

```python
import torch
from robotorchan.models import SACGP

# Context構造の詳細な入力契約はモデル別ガイドに従う。
print(SACGP.__name__)
```

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

上の例は公開シンボルの入口確認であり、学習・posterior実行例ではない。conditional inactive dimensionsの扱い、contextごとの入力構造、観測に存在しないcontextへの汎化条件はfamilyごとに制限がある。詳細は[モデル選択ガイド](../../../models/hierarchical_contextual.md)と[Hierarchical Notebook](../../../../examples/notebooks/12_hierarchical_gp.ipynb)、[Context Notebook](../../../../examples/notebooks/14_contextual_gp.ipynb)を参照する。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。

## 条件付き入力とcontextのモデル化境界

hierarchical modelでは親変数pによって子変数x_jが有効になるかが変わる。これを表すhierarchical_dependenciesは親入力列と親カテゴリ値からactiveとなる子列群への辞書であり、単なるカテゴリ集合やtask_featureの別名ではない。

src/robotorchan/models/structured/hierarchical.pyのHierarchicalConditionalKernelGPはBoTorchHierarchicalConditionalKernelGPを継承する。hierarchical_dependencies、eval_hierarchical_features、separate_hierarchical_features、use_saas_prior、use_outputscale等を上流へ委譲し、robotorchanはraw tensor snapshotとExact GPのmake_mll契約を補う。

MixedHierarchicalConditionalKernelGPでは、各active blockに対してcategorical／continuous／両者のadditive + product covarianceを組むローカル_MixedHierarchicalConditionalKernelを利用する。親変数の列は構造的であるためcat_dimsと重ねられず、重複するとValueErrorとなる。hierarchical_dependenciesは入力変換後の列と整合させる必要がある。

Contextual GPは別familyであり、特定context間の共分散やembeddingで情報を共有するモデルである。親による子の無効化規則をcontext IDと取り違えず、SACGP・LCEAGP・LCEMGPでは個別にAPIを確認する。

**検証境界：** 上記はhierarchical.pyの継承・kernel構築・negative contractのソース照合。active/inactive invarianceの数値property、未観測contextへの予測の妥当性を一括認証しない。

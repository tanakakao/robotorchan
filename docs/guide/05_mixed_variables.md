# Mixed Variables：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/05_mixed_variables.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

連続変数とカテゴリ変数を同時に扱う。整数のカテゴリラベルを連続量の距離と誤認すると、カテゴリの番号付けで共分散が変わってしまう。native categorical kernelは等値性に基づく共分散で、カテゴリを単純な数値距離とみなさない。学習前後でカテゴリ符号を一貫して保つ。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/standard/single_task.py`の`MixedSingleTaskGP`はBoTorch `MixedSingleTaskGP`のラッパーで、raw tensor保持と共通学習契約を追加する。
- **Delegated:** GP推論、native mixed covarianceとposteriorの主要処理はBoTorch/GPyTorch。
- **Implemented:** `src/robotorchan/models/base.py`の`normalize_feature_dims`と`make_mixed_covar_module`は他のmixed構造モデル向けの共通補助機能。全モデルが同じmixed kernelを用いるわけではない。
- `cat_dims`はraw入力列に対応し、task/fidelity/context等の構造列と同一視しない。

## 最小の利用例

```python
import torch
from robotorchan.models import MixedSingleTaskGP

X = torch.tensor([[0.1, 0.0], [0.3, 1.0], [0.8, 0.0], [0.9, 1.0]], dtype=torch.double)
Y = torch.tensor([[0.2], [0.8], [0.6], [0.4]], dtype=torch.double)
model = MixedSingleTaskGP(train_X=X, train_Y=Y, cat_dims=[1])
print(model.posterior(X[:2]).mean)
```

コードは確認した公開APIを示す最小例であり、すべての条件での成功を保証するものではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

カテゴリ名の置換に対する共分散の不変性は、同じ条件でのカテゴリIDの付け替えを比較するpropertyであり、全mixed variantの検証完了を意味しない。BoTorch-native categorical kernel、内部one-hot、embedding、passthroughをモデルごとに区別する。関連Notebook: [Mixed GP](../../examples/notebooks/02_mixed_single_task_gp.ipynb)。

[旧Model Mapping](../../.github/project-records/theory-guide/mapping/models.md)および[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../models/README.md)を参照する。

## カテゴリ共分散と符号化の実装境界

カテゴリcを数値的な距離として扱うRBFと、カテゴリ同一性に基づくcategorical kernelは異なる。カテゴリ番号を付け替えても同じカテゴリ同士の同一性が保存される場合、native categorical kernelの共分散構造は変わらないというのが重要なpropertyである。連続設計座標x_cとカテゴリcを組み合わせると、専用mixed covariance helperは概念的に連続項・カテゴリ項・交互作用項を足し合わせる。

src/robotorchan/models/standard/single_task.pyのMixedSingleTaskGPはBoTorchMixedSingleTaskGPを直接継承し、BoTorch-native mixed kernelとposterior推論を上流へ委譲する。一方、src/robotorchan/models/base.pyのmake_mixed_covar_moduleはrobotorchanが他の構造化mixed family向けに加法・積kernelを構成するための実装で、MixedSingleTaskGPの内部kernelを必ず上書きするものではない。

| モデル系 | カテゴリの取扱い | 確認するべき契約 |
| --- | --- | --- |
| MixedSingleTaskGP | BoTorch native mixed categorical kernel | train_Xはraw [n,d]、cat_dimsはraw列番号。raw snapshotsとMLL便宜APIを追加 |
| MixedMultiTaskGP | 設計変数上のnative mixed covariance | task_featureはstructuralでありcat_dimsに入れない |
| MixedSingleTaskMultiFidelityGP | 設計変数上のnative mixed covarianceとBoTorch fidelity kernel | fidelity列をcat_dimsから除外。linear_truncated=Trueは拒否 |
| MixedSaasFullyBayesianSingleTaskGP | モデル所有のone-hot入力変換 | native mixed kernel経路とは異なる。カテゴリ列とencoded次元を区別 |

raw train_Xのカテゴリ列を実験側で任意に丸めるのではなく、学習・posterior・optimizerの全境界で同じ符号と候補値を使う。負のcat_dimsやtask/fidelityとの重複を扱う場合は、各モデルのnormalize_feature_dims呼出しを確認する。これはBoTorchMixedSingleTaskGPの全入力検証を代替する主張ではない。

**検証境界：** 上記の継承と共分散構成はソース照合。カテゴリラベル付け替えの数値不変性・探索候補の可行性は別途独立実験が必要。

## 数値 API と利用上の補足

- [Kernel](../theory/03_kernel.md)
- [Acquisition Function](../theory/04_acquisition_function.md)
- [Hierarchical / Contextual GP](../theory/12_hierarchical_contextual_gp.md)
- [Multi-task / Multi-output](../theory/07_multitask_multioutput.md)
- [Mixed-variable model documentation](../models/README.md)

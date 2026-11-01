# Hierarchical / Contextual GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/12_hierarchical_contextual_gp.md)は数学的仮定・導出を担当する。本章では理論の構成要素がクラスと内部処理へ対応する経路を説明する。

親変数の値で子変数の有効・無効が切り替わるconditional search spaceと、context間の情報を構造的に共有するcontextual GPを区別する。
カテゴリ値を持つだけの通常Mixed入力と、入力の存在条件が変わるhierarchical変数は同じではない。

## 実装の責務と公開契約

- **Implemented/Extended（family別）:** `src/robotorchan/models/structured/hierarchical.py`には
  `HierarchicalConditionalKernelGP`系、`src/robotorchan/models/structured/contextual.py`には`SACGP`、
  `LCEAGP`、`LCEMGP`等のcontext系がある。
- **Delegated:** GPの基本的なposterior演算など、BoTorch/GPyTorchへ委譲する部分と、
  条件付きkernel・context embeddingを構築するlocal部分はコード経路ごとに分けて確認する。
- 同一のcontext IDを設計カテゴリ、task index、あるいはlatent embeddingとして扱うモデルを一括りにしない。

## 実装上の制限

conditional inactive dimensionsの扱い、contextごとの入力構造、観測に存在しないcontextへの汎化条件はfamilyごとに制限がある。詳細は
[モデル選択ガイド](../models/hierarchical_contextual.md)と
[Hierarchical Notebook](../../examples/notebooks/12_hierarchical_gp.ipynb)、
[Context Notebook](../../examples/notebooks/14_contextual_gp.ipynb)を参照する。

## 条件付き入力とcontextのモデル化境界

hierarchical modelでは親変数pによって子変数x_jが有効になるかが変わる。
これを表すhierarchical_dependenciesは親入力列と親カテゴリ値からactiveとなる子列群への辞書であり、単なるカテゴリ集合やtask_featureの別名ではない。

src/robotorchan/models/structured/hierarchical.pyのHierarchicalConditionalKernelGPは
BoTorchHierarchicalConditionalKernelGPを継承する。hierarchical_dependencies、eval_hierarchical_features、
separate_hierarchical_features、use_saas_prior、use_outputscale等を上流へ委譲し、
robotorchanはraw tensor snapshotとExact GPのmake_mll契約を補う。

MixedHierarchicalConditionalKernelGPでは、
各active blockに対してcategorical／continuous／両者のadditive + product
covarianceを組むローカル_MixedHierarchicalConditionalKernelを利用する。親変数の列は構造的であるためcat_dimsと重ねられず、
重複するとValueErrorとなる。hierarchical_dependenciesは入力変換後の列と整合させる必要がある。

Contextual GPは別familyであり、特定context間の共分散やembeddingで情報を共有するモデルである。親による子の無効化規則をcontext IDと取り違えず、
SACGP・LCEAGP・LCEMGPでは個別にAPIを確認する。

## 関連する実装資料

- [Mixed Variables](../theory/05_mixed_variables.md)
- [Multi-task / Multi-output](../theory/07_multitask_multioutput.md)
- [High-dimensional GP](../theory/08_high_dimensional_gp.md)
- [Model Selection](../theory/13_model_selection.md)
- [Model documentation](../models/README.md)

## 条件付きkernelの構成と上流への委譲

[階層モデルの実装](../../src/robotorchan/models/structured/hierarchical.py)では、
標準版が`ExactGPModelMixin`とBoTorchの階層GPを継承します。
Mixed版の`_MixedHierarchicalConditionalKernel`は上流の`HierarchicalConditionalKernel`を継承し、
`construct_individual_kernels()`を差し替えます。各active block内の連続列とカテゴリ列を分離して、対応するkernel因子を組み立てます。
親の値からactive blockを選ぶ仕組みは上流の階層kernelに残します。

この分担により、Theoryの「無効な子の値へ共分散が依存しない」構造を保ちつつ、有効なblock内のカテゴリを通常の連続距離として扱わずに済みます。
`_parent_dims()`は依存関係から親列を求め、Mixedの設計カテゴリとの重なりを拒否します。

[context系の実装](../../src/robotorchan/models/structured/contextual.py)では、
`SACGP`・`LCEAGP`・`LCEMGP`がそれぞれ同名のBoTorchモデルと`ExactGPModelMixin`を継承します。
context分解・embeddingによる共分散は上流が担当します。`MixedLCEMGP`は設計側のkernelを拡張し、task列と設計カテゴリ列を分離します。
階層モデルのactive / inactive規則をcontext embeddingで代用してはいません。

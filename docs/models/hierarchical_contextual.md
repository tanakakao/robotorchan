# Hierarchical / Heterogeneous / Contextual models

## Conditional search spaces

`HierarchicalConditionalKernelGP` と `MixedHierarchicalConditionalKernelGP` は、親変数の値によって有効になる子変数が変わる探索空間を扱います。

Multi-task版として `HierarchicalConditionalKernelMultiTaskGP` と `MixedHierarchicalConditionalKernelMultiTaskGP` があります。通常の Mixed model はカテゴリを扱えても、「ある条件では変数自体が非アクティブ」という構造を自動的には表しません。

## Heterogeneous tasks

`HeterogeneousMTGP` と `MixedHeterogeneousMTGP` は task ごとに特徴量構成が異なる場合を扱います。これは conditional search space と似て見えても、task-specific feature availability をモデル化する別の問題です。

## Contextual GP

`SACGP` は context ごとの加法構造、`LCEAGP` は context 間の潜在 embedding を使う additive model、`LCEMGP` と `MixedLCEMGP` は context / task を multi-output として扱います。

context が単なるカテゴリ列なのか、分解可能な構造・taskとして意味を持つのかを確認して選択してください。

Notebook: [Hierarchical](../../examples/notebooks/12_hierarchical_gp.ipynb)、[Heterogeneous](../../examples/notebooks/13_heterogeneous_multitask_gp.ipynb)、[Contextual](../../examples/notebooks/14_contextual_gp.ipynb)  
Theory: [Hierarchical / Contextual](../theory/12_hierarchical_contextual_gp.md)

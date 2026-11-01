# Mixed / Discrete：獲得関数最適化の実装ガイド

## この章の問い

整数は順序・距離があるがunordered category IDに自然な距離はない。fixed features列挙とtype-aware変異を選び分ける。

数理的な正本は[理論の対応章](../../theory/optimization/04_mixed_discrete.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

categorical codeを連続補間して丸める方法は人工的な順序を持ち込む。q-batchでのカテゴリ割当も個別に確認する。

## Native mixed BoTorchとmixed one-shotの数学的差異

$$
\max_{c\in C}\;\max_{x_{\mathrm{continuous}}}
\alpha(x_{\mathrm{continuous}},c).
$$

固定するカテゴリ値は fixed_features_list として渡される。
これは11種類の named optimizer を選択する optimize_acqf(optimizer=" botorch") とは異なり、
後者は mixed MixedVariableSpace を明示的に拒否する。数式は探索方針の概念化であり、BoTorch内部の全枝刈りやq-batch処理の完全な再記述ではない。

- **native mixed**
  実装入口：backends/botorch.py::optimize_acqf_mixed_botorch
  主要な制約：fixed_features_list を受け、inter-point nonlinear constraintsを拒否

- **mixed GA**
  実装入口：dispatch.pyの optimizer="ga" と MixedVariableSpace
  主要な制約：is_mixedならmixed_genetic_algorithm.pyに分岐

- **mixed one-shot**
  実装入口：one_shot/mixed.py::optimize_mixed_one_shot_acqf
  主要な制約：fantasy補助行を含む拡張batchに行別カテゴリ割当てが必要。q=1のみ

native mixedではnonlinear candidate constraintsを使用すると、
feasibleな batch_initial_conditions またはic_generator が必要となる。inter-point nonlinear制約はValueErrorで拒否され、
q点間の制約を点単位の独立制約へ無断変換しない。候補の数値精度や全制約可行性は、委譲したという事実だけから保証できない。

通常のカテゴリIDを連続座標として補間することと、カテゴリごとの固定割当てを明示することは異なる。特にone-shotのfantasy行には別々のカテゴリ値が入り得るため、
[Batch/one-shot](08_batch_one_shot.md)の拡張batch契約を参照する。

## 固定割当てとtype-aware探索の実装境界

[backends/botorch.py](../../../src/robotorchan/optim/backends/botorch.py)の
`optimize_acqf_mixed_botorch()`は、カテゴリの固定割当てと制約を検査し、上流の`optimize_acqf_mixed()`へ渡します。
連続部分の最適化は上流が担当します。

一方、[mixed_genetic_algorithm.py][mixed-ga]は、整数・カテゴリの許容値を保つローカルの探索経路です。
共通の[MixedVariableSpace][space]が変数型を表し、
[dispatch.py](../../../src/robotorchan/optim/dispatch.py)はGAとmixed空間の組合せで
この専用経路へ分岐します。カテゴリの固定列挙と、型に応じた集団探索を区別する設計です。

[mixed-ga]: ../../../src/robotorchan/optim/backends/mixed_genetic_algorithm.py
[space]: ../../../src/robotorchan/optim/domains/variable_space.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/02_mixed_single_task_gp.ipynb)を参照してください。

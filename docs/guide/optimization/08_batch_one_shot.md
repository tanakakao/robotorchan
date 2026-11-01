# Batch / One-shot：実候補とfantasy変数を分離する

## Joint-qとone-shotの意思決定変数

[理論の対応章](../../theory/optimization/08_batch_one_shot.md)が数理的な正本である。通常のjoint-q最適化では実際に評価する候補数が`q`である。
一方one-shot acquisition（Knowledge Gradientやmulti-step lookahead）は、最適化する拡張変数に**fantasy補助点**を含む。
拡張サイズ`augmented_q = acq_function.get_augmented_q_batch_size(q)`は実際の評価点数とは異なり、
返す実候補には`acq_function.extract_candidates`が必要である。

`X_pending`は評価中でまだ観測を受け取っていない候補であり、one-shotのfantasy変数とも完了観測値とも混同しない。

## 一般one-shotの初期点生成

`src/robotorchan/optim/one_shot/initialization.py`の`gen_augmented_one_shot_initial_conditions`は、
`get_augmented_q_batch_size(q)`で決めたサイズをBoTorch `gen_batch_initial_conditions`へ渡す。
戻り値の形状は`[num_restarts, augmented_q, d]`である。

このhelperは専用initializerのないone-shot（例：`qMultiStepLookahead`）用であり、
Knowledge Gradientに専用のBoTorch initializerがある場合はそちらを使う。一般helperがKGに常に最良の初期値を与えると主張しない。

## Mixed one-shotの実装：行ごとに異なるカテゴリを保持

`src/robotorchan/optim/one_shot/mixed.py::optimize_mixed_one_shot_acqf`は`OneShotAcquisitionFunction`
を受け取る。現行実装では**q=1のみ**許可し、その他のqに対して`NotImplementedError`を出す。

実装は次の手順をとる。

1. カテゴリ次元と選択肢を検証し、各選択肢がbounds内にあることを確かめる。
2. 拡張batch内の**各行ごと**にカテゴリ割当てを列挙する。`m`を1行あたりの組合せ数とすると、探索する割当て数は`m ** augmented_q`となる。
   `max_assignments`を超えれば`ValueError`で停止する。
3. 各割当てで`[augmented_q,d]`をflattenし、`[1,augmented_q*d]`の連続最適化問題へ変換する。元の行ごとに別々の`fixed_features`を設定して、
   BoTorchの`optimize_acqf`へ委譲する。
4. 最良の拡張点に`extract_candidates`を適用し、**実評価する候補だけ**を返す。fantasy点を実験へ渡してはいけない。

通常のmixed optimizerにaugmented batchを直接渡すと、行ごとの異なるカテゴリ選択が保存されない可能性がある。
`check_acquisition_optimizer_compatibility`は`CandidateOptimizer.MIXED`とone-shotの組合せを拒否し、
`MIXED_ONE_SHOT`では逆にnon-one-shotを拒否する。これは事前の意味論チェックであり、実行成功の保証ではない。

## 実装の責務と数値的な制限

- **通常のBoTorch one-shot**
  実装・委譲：BoTorchのlookahead / fantasy machineryへDelegated
  主な制限：`q`とaugmented batchを区別する

- **一般one-shot初期値**
  実装・委譲：`gen_augmented_one_shot_initial_conditions`がExtended
  主な制限：KG専用initializerとは区別

- **Mixed one-shot**
  実装・委譲：`optimize_mixed_one_shot_acqf`が行別カテゴリ割当て列挙をImplemented
  主な制限：`q=1`、`max_assignments`による列挙上限

- **連続部分問題**
  実装・委譲：BoTorch `optimize_acqf`にDelegated
  主な制限：得られた値は一般に大域最適性の証明ではない

## 拡張batchに対する組合せ数と連続サブ問題

$$
M=\prod_j K_j,
\qquad N_{\mathrm{assignments}}=M^{q_{\mathrm{aug}}}.
$$

ここで q_aug は acq_function.get_augmented_q_batch_size(q) であり、実験候補数 q と区別する。例えばカテゴリ1列が3択、
augmented_q=4であれば81通りを列挙する。max_assignmentsより大きい場合はValueErrorを出すので、カテゴリ数・fantasy数を増やすほど組合せ爆発が生じる。

各割当てで [q_aug,d] を1個の連続候補 [1,q_aug*d] にflattenする。カテゴリ列 j の各行 r に相当するflatten後の列番号は r*d+j なので、
この列を固定することでfantasy行ごとのカテゴリ差異を保存する。連続部分をBoTorch optimize_acqfへ渡し、得られた拡張変数を [q_aug,d] に戻し、
extract_candidates によって実験するq行だけを返す。

## 数理モデルとクラス構造

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

未完了の実験点を`X_pending`で表す。新しい候補を選ぶ際は、そのpending点を考慮するacquisitionへ渡す。

したがって asynchronous support は optimizer だけの性質ではなく、acquisition が X_pending をどう扱うかにも依存します。

candidate を1点選ぶ-> X_pending に追加-> 同じ acquisition を再最適化

| 目的 | Optimization structure |
| --- | --- |
| q点を同時決定 | joint q-batch |
| q点を貪欲に構築 | sequential + X_pending |
| 実行中評価を考慮 | asynchronous + X_pending-aware acquisition |
| KG | one-shot + specialized KG initialization |
| multi-step lookahead | one-shot augmented tree |
| mixed one-shot | row-specific categorical handling |

## 補助行ごとのカテゴリを扱う専用経路

[one_shot/mixed.py](../../../src/robotorchan/optim/one_shot/mixed.py)の
`optimize_mixed_one_shot_acqf()`は、acquisitionから拡張qを取得し、各補助行に対するカテゴリ割当てを列挙します。
行ごとに固定値を適用するacquisitionへ包んで連続部分を最適化し、最良解から`extract_candidates()`で実験する行だけを取り出します。

一つのカテゴリを拡張batch全体へ固定する通常のmixed探索とは変数構造が異なります。割当数はカテゴリの組合せと拡張qに依存して増えるため、`max_assignments`で上限を持ち、
現在は実候補q=1へ限定します。列挙・行ごとの固定化はローカル拡張で、fantasyモデルと終端価値はBoTorchが担当します。

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/32_batch_async_fantasization.ipynb)を参照してください。

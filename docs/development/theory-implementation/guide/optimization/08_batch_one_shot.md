# Batch / One-shot：実候補とfantasy変数を分離する

## 1. Joint-qとone-shotの意思決定変数

[理論の対応章](../../../../theory/optimization/08_batch_one_shot.md)が数理的な正本である。通常のjoint-q最適化では実際に評価する候補数が`q`である。一方one-shot acquisition（Knowledge Gradientやmulti-step lookahead）は、最適化する拡張変数に**fantasy補助点**を含む。拡張サイズ`augmented_q = acq_function.get_augmented_q_batch_size(q)`は実際の評価点数とは異なり、返す実候補には`acq_function.extract_candidates`が必要である。

`X_pending`は評価中でまだ観測を受け取っていない候補であり、one-shotのfantasy変数とも完了観測値とも混同しない。

## 2. 一般one-shotの初期点生成

`src/robotorchan/optim/one_shot/initialization.py`の`gen_augmented_one_shot_initial_conditions`は、`get_augmented_q_batch_size(q)`で決めたサイズをBoTorch `gen_batch_initial_conditions`へ渡す。戻り値の形状は`[num_restarts, augmented_q, d]`である。

このhelperは専用initializerのないone-shot（例：`qMultiStepLookahead`）用であり、Knowledge Gradientに専用のBoTorch initializerがある場合はそちらを使う。一般helperがKGに常に最良の初期値を与えると主張しない。

## 3. Mixed one-shotの実装：行ごとに異なるカテゴリを保持

`src/robotorchan/optim/one_shot/mixed.py::optimize_mixed_one_shot_acqf`は`OneShotAcquisitionFunction`を受け取る。現行実装では**q=1のみ**許可し、その他のqに対して`NotImplementedError`を出す。

実装は次の手順をとる。

1. カテゴリ次元と選択肢を検証し、各選択肢がbounds内にあることを確かめる。
2. 拡張batch内の**各行ごと**にカテゴリ割当てを列挙する。`m`を1行あたりの組合せ数とすると、探索する割当て数は`m ** augmented_q`となる。`max_assignments`を超えれば`ValueError`で停止する。
3. 各割当てで`[augmented_q,d]`をflattenし、`[1,augmented_q*d]`の連続最適化問題へ変換する。元の行ごとに別々の`fixed_features`を設定して、BoTorchの`optimize_acqf`へ委譲する。
4. 最良の拡張点に`extract_candidates`を適用し、**実評価する候補だけ**を返す。fantasy点を実験へ渡してはいけない。

通常のmixed optimizerにaugmented batchを直接渡すと、行ごとの異なるカテゴリ選択が保存されない可能性がある。`check_acquisition_optimizer_compatibility`は`CandidateOptimizer.MIXED`とone-shotの組合せを拒否し、`MIXED_ONE_SHOT`では逆にnon-one-shotを拒否する。これは事前の意味論チェックであり、実行成功の保証ではない。

## 4. 実装の責務と数値的な制限

| 対象 | 実装・委譲 | 主な制限 |
| --- | --- | --- |
| 通常のBoTorch one-shot | BoTorchのlookahead / fantasy machineryへDelegated | `q`とaugmented batchを区別する |
| 一般one-shot初期値 | `gen_augmented_one_shot_initial_conditions`がExtended | KG専用initializerとは区別 |
| Mixed one-shot | `optimize_mixed_one_shot_acqf`が行別カテゴリ割当て列挙をImplemented | `q=1`、`max_assignments`による列挙上限 |
| 連続部分問題 | BoTorch `optimize_acqf`にDelegated | 得られた値は一般に大域最適性の証明ではない |

個別テストの証拠は旧[Optimization Mapping](../../optimization.md)の`optim/one_shot/mixed.py`／`tests/optim/one_shot/test_mixed.py`行に記録されている。合成例のカテゴリ横断割当てと`max_assignments`等のassertionは**特定の条件での実装契約**を確認するもので、一般のmixed qKG性能や全制約との互換性を証明しない。

[Batch/Noisy acquisition](../acquisition/04_batch_noisy.md)、[One-shotの理論](../../../../theory/optimization/08_batch_one_shot.md)、[Optimizer検証](../verification/optimization.md)、[旧Mapping](../../optimization.md)を併読し、原文の検収完了まで旧Mappingを保持する。

## 5. 拡張batchに対する組合せ数と連続サブ問題

旧Optimization Mappingの固定区間246–257、および src/robotorchan/optim/one_shot/mixed.py の実装に従う。カテゴリ次元 j が K_j 種類の値を持つとき、1行に対する割当て数は

$$
M=\prod_j K_j,
\qquad N_{\mathrm{assignments}}=M^{q_{\mathrm{aug}}}.
$$

ここで q_aug は acq_function.get_augmented_q_batch_size(q) であり、実験候補数 q と区別する。例えばカテゴリ1列が3択、augmented_q=4であれば81通りを列挙する。max_assignmentsより大きい場合はValueErrorを出すので、カテゴリ数・fantasy数を増やすほど組合せ爆発が生じる。

各割当てで [q_aug,d] を1個の連続候補 [1,q_aug*d] にflattenする。カテゴリ列 j の各行 r に相当するflatten後の列番号は r*d+j なので、この列を固定することでfantasy行ごとのカテゴリ差異を保存する。連続部分をBoTorch optimize_acqfへ渡し、得られた拡張変数を [q_aug,d] に戻し、extract_candidates によって実験するq行だけを返す。

~~~python
from robotorchan.optim.one_shot.mixed import optimize_mixed_one_shot_acqf

# acqf: BoTorch OneShotAcquisitionFunction、bounds: [2, d]を事前に用意
candidate, score = optimize_mixed_one_shot_acqf(
    acqf,
    bounds,
    categorical_features={1: (0.0, 1.0)},
    q=1,
    max_assignments=256,
)
# candidateは実評価用。fantasy補助行は含まない。
~~~

このコード断片は acqf と bounds が既に存在する場合の呼び出し形を示す。成立にはカテゴリ値がbounds内、指定次元が有効、選択肢が空でないことが必要。最適化成功を保証する実行例ではない。q>1はNotImplementedErrorであり、一般のq-batch mixed KGへ拡張済みと解釈してはいけない。

**検証境界：** tests/optim/one_shot/test_mixed.pyの限定的なassertionと実装を参照する。各連続部分の大域最適性、全one-shot獲得関数への適合、全candidate制約の対応は未証明である。

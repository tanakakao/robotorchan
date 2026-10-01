# Batch and One-shot Acquisition Optimization

## 8.1 目的

Bayesian Optimization で複数候補を扱うときは、少なくとも次の概念を区別する必要があります。

- joint q-batch optimization
- sequential greedy batch construction
- asynchronous optimization with pending points
- one-shot augmented optimization
- multi-step adaptive lookahead

これらはすべて複数の candidate を扱いますが、最適化変数と情報構造が異なります。

## 8.2 Joint q-batch optimization

q-batch acquisition は

    alpha(X),  X = (x_1, ..., x_q)

を joint に評価します。

したがって解く問題は

    X* = argmax alpha(x_1, ..., x_q)

であり、各 x_i の pointwise acquisition value を独立に rank する問題ではありません。

qEI、qNEI、qEHVI、qNEHVI などでは candidate 間の posterior correlation や
joint improvement が acquisition value に影響します。

robotorchan の scalar acquisition optimizer backend は q > 1 の場合も
[q, d] を1つの decision object として扱います。
gradient-free backend では内部的に [qd] へ flatten する場合がありますが、
acquisition evaluation 前には [q, d] に戻されます。

## 8.3 Joint batch の長所とコスト

joint optimization は batch 全体の相互作用を直接考慮できます。

一方、decision dimension は d から qd へ増えます。
q が大きいほど optimization landscape は高次元になり、
initialization、restart、population size、acquisition evaluation cost の影響が大きくなります。

特に Monte Carlo acquisition では

    optimizer candidates
      x q-batch
      x MC samples
      x posterior computation

という計算が重なるため、black-box evaluation が高価でなくても
acquisition optimization 自体が支配的になる場合があります。

## 8.4 Sequential greedy batch construction

joint q-batch を一度に解く代わりに、1点ずつ選択して batch を構成する方法があります。

概念的には

    x_1 = argmax alpha(x)
    x_2 = argmax alpha(x | x_1 pending)
    ...
    x_q = argmax alpha(x | x_1, ..., x_(q-1) pending)

です。

後続 candidate の選択時には、すでに選んだ candidate を pending として acquisition に知らせます。

これは最初から q 点を joint optimize することとは異なります。
greedy approximation なので、一般に joint optimum と一致する保証はありません。

## 8.5 robotorchan の sequential helper

robotorchan の optimize_acqf_sequential は generic backend wrapper です。

各 step で指定 optimizer を q=1 として呼び出し、得られた candidate を X_pending に追加します。

重要な runtime contract は次です。

- acquisition が set_X_pending を持つ必要がある
- 呼び出し前に存在した X_pending を保持する
- 新規 candidate は既存 pending の後ろへ追加する
- q 点の生成終了後、元の X_pending を復元する
- 各 step の acquisition value を個別に返す

したがって helper 自体が fantasy model を構築するわけではありません。
pending point を acquisition へ伝える責務を持つ orchestration layer です。

## 8.6 Pending points と asynchronous BO

asynchronous BO では、すでに評価を開始したが結果が未観測の点があります。

これを X_pending とすると、新しい candidate は

    observed data + pending locations

を考慮して選ぶ必要があります。

pending point は「観測済みデータ」ではありません。
また、単に candidate space から除外することとも同義ではありません。

acquisition によっては pending locations を内部の fantasy / sampling semantics に
組み込んで価値を調整します。

したがって asynchronous support は optimizer だけの性質ではなく、
acquisition が X_pending をどう扱うかにも依存します。

## 8.7 Fantasy と sequential selection の違い

sequential batch construction と fantasy model は関連しますが同一ではありません。

sequential helper:

    candidate を1点選ぶ
      -> X_pending に追加
      -> 同じ acquisition を再最適化

fantasy model:

    未知の observation を sample
      -> その observation で model を condition
      -> future posterior / decision value を評価

前者は candidate selection procedure、後者は未知観測を積分するための
probabilistic model operation です。

## 8.8 One-shot acquisition

Knowledge Gradient や multi-step lookahead では nested optimization が現れます。

例えば概念的な KG は

    max_x E_y [ max_x' terminal_value(x' | x, y) ]

のように、outer candidate と fantasy outcome ごとの inner decision を含みます。

one-shot formulation は、この inner decision variables を auxiliary variables として
outer optimization variable に組み込みます。

その結果、optimizer が見る tensor は実際に返したい q candidates より大きくなります。

## 8.9 Augmented q-batch

OneShotAcquisitionFunction では public q に対して

    q_aug = get_augmented_q_batch_size(q)

で augmented batch size が決まります。

optimizer は

    [q_aug, d]

を最適化しますが、ユーザーが実際に評価する candidate は通常その一部です。

最適化後は

    extract_candidates(X_aug)

により実 candidate を取り出します。

したがって one-shot acquisition に対して、通常の q-batch と同じ shape の
initial condition を渡すことは一般に正しくありません。

## 8.10 Knowledge Gradient の one-shot optimization

qKnowledgeGradient は代表的な one-shot acquisition です。

fantasy points は実験候補ではなく、future terminal decision を近似するための
auxiliary optimization variables です。

概念的には augmented tensor は

    actual candidates
    + fantasy / auxiliary decision points

を含みます。

BoTorch は qKG に専用の initialization path を持つため、
robotorchan はこれを独自 initializer で置き換えません。

qKG では native BoTorch optimization path を優先します。

## 8.11 qMultiStepLookahead

qMultiStepLookahead は複数段階の adaptive decision tree を one-shot 形式で表現します。

future stage ごとに fantasy branching があるため、
augmented q-batch は horizon と fantasy count に応じて急速に大きくなります。

重要なのは

    batch size q != lookahead depth

という点です。

batch q は現在同時に選ぶ candidate 数です。
lookahead depth は将来観測後に何段階の adaptive decision を評価するかを表します。

## 8.12 Multi-step の augmented initialization

qMultiStepLookahead のように専用 initializer を持たない one-shot acquisition では、
通常の q ではなく q_aug に対する initial conditions が必要です。

robotorchan の gen_augmented_one_shot_initial_conditions は

    q_aug = acq_function.get_augmented_q_batch_size(q)

を求め、BoTorch の gen_batch_initial_conditions を augmented batch 全体へ適用します。

この helper は新しい sampling heuristic を導入するものではありません。
BoTorch standard initializer を正しい augmented shape で呼ぶための bridge です。

## 8.13 Mixed one-shot が難しい理由

通常の mixed optimization では categorical assignment を固定して
continuous coordinates を最適化する方法があります。

しかし one-shot acquisition では augmented batch の各 row が異なる意味を持ちます。

1つの categorical assignment を augmented batch 全体へ一律に固定すると、

    actual candidate
    fantasy decision 1
    fantasy decision 2
    ...

がすべて同じ category に拘束され、one-shot acquisition の内部 decision structure を
壊す可能性があります。

したがって mixed one-shot では row-specific categorical assignment が必要です。

## 8.14 robotorchan の Mixed one-shot

robotorchan の optimize_mixed_one_shot_acqf は correctness-first の専用 path です。

現在の contract は次の通りです。

- q=1 のみ
- OneShotAcquisitionFunction を要求
- augmented q の各 row ごとに categorical assignment を持つ
- categorical combination を全列挙する
- 各 assignment では augmented tensor を flatten して BoTorch optimize_acqf を利用
- 最適化後に extract_candidates で実 candidate を取得
- assignment explosion を max_assignments で制限

categorical feature の値は bounds 内でなければなりません。

## 8.15 Mixed one-shot の組合せ爆発

1 row あたりの categorical assignment 数を C、
augmented row 数を q_aug とすると、単純列挙数は

    C^(q_aug)

です。

複数 categorical dimension がある場合、C 自体が各 dimension の category 数の積になります。

したがって q=1 であっても fantasy count が増えると列挙数は急増します。

現在の max_assignments は単なる性能 tuning ではなく、
correctness-first enumeration を現実的な範囲に制限する safety boundary です。

## 8.16 One-shot と ordinary q-batch の比較

| 項目 | ordinary q-batch | one-shot |
| --- | --- | --- |
| optimizer variable | q candidates | q candidates + auxiliary variables |
| optimizer batch size | q | q_aug |
| returned candidates | optimized q rows | extract_candidates の結果 |
| fantasy variables | 通常はoptimizer変数ではない | optimizer変数に含み得る |
| initialization | q rows | q_aug rows |
| mixed handling |通常のmixed semantics | augmented row semanticsが必要 |

この違いを無視すると、shape は通っても acquisition の意味を壊す可能性があります。

## 8.17 Sequential と one-shot の比較

sequential optimization は candidate を1点ずつ確定していく outer procedure です。

one-shot optimization は nested decision problem の auxiliary variables を
1回の拡張最適化問題へ埋め込む formulation です。

したがって

    sequential = repeated candidate selection
    one-shot = augmented optimization representation

であり、対立概念ではありません。

## 8.18 Constraint との関係

ordinary joint q-batch では inter-point constraint が batch 内 candidate 間の関係を
表現できます。

one-shot では augmented rows のすべてが実験 candidate とは限りません。
そのため「実 candidate に課す制約」を augmented auxiliary rows に機械的に適用すると、
本来の decision problem と異なる可能性があります。

one-shot acquisition に candidate constraint を組み合わせる場合は、

- constraint が actual candidate のみを対象とするか
- auxiliary fantasy decision にも必要か
- initializer と optimizer の両方が同じ semantics を保つか

を確認する必要があります。

## 8.19 最適化方式の選択

代表的な判断は次のように整理できます。

| 目的 | Optimization structure |
| --- | --- |
| q点を同時決定 | joint q-batch |
| q点を貪欲に構築 | sequential + X_pending |
| 実行中評価を考慮 | asynchronous + X_pending-aware acquisition |
| KG | one-shot + specialized KG initialization |
| multi-step lookahead | one-shot augmented tree |
| mixed one-shot | row-specific categorical handling |

backend の capability だけでなく acquisition の tensor contract を確認することが重要です。

## 8.20 実装上の境界

robotorchan は batch / one-shot のすべてを独自 abstraction に置き換える方針ではありません。

BoTorch が正しい native semantics を持つ部分はそのまま利用し、
robotorchan は次の gap を補います。

- backend-agnostic sequential orchestration
- augmented q を意識した generic one-shot initialization
- correctness-first mixed one-shot categorical handling
- optimizer / acquisition compatibility validation

この境界により、BoTorch-first を維持しながら mixed backend や独自 optimizer を
組み合わせられるようにします。

## 参考文献

1. Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
   *Kriging Is Well-Suited to Parallelize Optimization*.
   In Computational Intelligence in Expensive Optimization Problems.
2. Frazier, P. I., Powell, W. B., & Dayanik, S. (2008).
   *A Knowledge-Gradient Policy for Sequential Information Collection*.
   SIAM Journal on Control and Optimization, 47(5), 2410-2439.
3. Wu, J., & Frazier, P. I. (2016).
   *The Parallel Knowledge Gradient Method for Batch Bayesian Optimization*.
   NeurIPS 29.
4. Jiang, S., Malkomes, G., Converse, G., Shofner, A., Moseley, B., & Garnett, R. (2020).
   *Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step Trees*.
   NeurIPS 33.
5. Wilson, J. T., Hutter, F., & Deisenroth, M. P. (2018).
   *Maximizing Acquisition Functions for Bayesian Optimization*.
   NeurIPS 31.
6. BoTorch documentation. *Optimizing acquisition functions* and
   *One-shot acquisition functions*.

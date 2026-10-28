# Batch and One-shot Acquisition Optimization

[獲得関数の先読み章](../acquisition/06_lookahead.md)と[初期化章](03_initialization.md)を前提とします。
本章では、実験する候補、将来観測の標本、補助的な判断変数を区別して数値問題を構成します。

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

joint q-batch は [q, d] を1つの decision object として扱います。
内部的に [qd] へ flatten しても、acquisition evaluation は [q, d] 上で行います。

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

集合の効用 $\alpha(S)$ に対する貪欲な構成なら、既選択集合Sを固定して
$\alpha(S\cup\{x\})-\alpha(S)$ の大きい点を順に選びます。
未知の結果があることを反映する方法と、結果を観測したことにしてモデルを更新する方法は同じではありません。
特に予測平均を確定した観測結果として扱う操作は、未知結果の分布を積分する操作とは異なります。

## 8.5 Sequential selection と pending state

各ステップで選んだ候補を pending set に追加し、後続の獲得関数に反映します。
既存の pending state と新しく選んだ候補を区別する必要があります。
この操作は fantasy model の構築と同義ではありません。

## 8.6 Pending points と asynchronous BO

asynchronous BO では、すでに評価を開始したが結果が未観測の点があります。

これを 未完了の入力集合 とすると、新しい candidate は

    observed data + pending locations

を考慮して選ぶ必要があります。

pending point は「観測済みデータ」ではありません。
また、単に candidate space から除外することとも同義ではありません。

acquisition によっては pending locations を内部の fantasy / sampling semantics に
組み込んで価値を調整します。

したがって asynchronous support は optimizer だけの性質ではなく、
acquisition が 未完了の入力集合 をどう扱うかにも依存します。

## 8.7 Fantasy と sequential selection の違い

sequential batch construction と fantasy model は関連しますが同一ではありません。

sequential helper:

    candidate を1点選ぶ
      -> 未完了の入力集合 に追加
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

one-shot formulation は実際に評価する q 個の候補に加え、
将来の意思決定を表す補助変数を最適化変数に含めます。
補助変数を含む augmented dimension は q より大きくなる場合があります。
最適化後は augmented solution から実際の候補を抽出します。
したがって初期条件も補助変数を含む空間で定義する必要があります。

一つの終端判断を各仮想観測で選ぶ単純なKGでは、現在のq点に加え、S個の標本にそれぞれ一つの判断を置きます。
入力次元がdなら、拡張した決定変数の数は $d(q+S)$ です。
Sを増やすと期待値の近似は精密になり得ますが、最適化の次元も増えます。

## 8.10 Knowledge Gradient の one-shot optimization

バッチKnowledge Gradientは、一括した拡張最適化で扱える代表例です。

fantasy points は実験候補ではなく、future terminal decision を近似するための
auxiliary optimization variables です。

概念的には augmented tensor は

    actual candidates
    + fantasy / auxiliary decision points

を含みます。

fantasy decision の初期値は実際の候補とは異なる役割を持つため、
その構造を考慮した初期化が重要です。

## 8.11 多段階先読み

多段階先読みでは、複数段階の適応的な判断木を一括した拡張問題で表現できます。

future stage ごとに fantasy branching があるため、
augmented q-batch は horizon と fantasy count に応じて急速に大きくなります。

重要なのは

    batch size q != lookahead depth

という点です。

batch q は現在同時に選ぶ candidate 数です。
lookahead depth は将来観測後に何段階の adaptive decision を評価するかを表します。

例えば現在一つの候補を選び、次の観測を2枝、その後を各3枝に近似し、各段階に一つの判断を置くと、
拡張変数には $1+2+2\times3=9$ 点分が現れます。
これは9点を今から並列実験するという意味ではなく、実際の観測履歴に応じて通る枝は一つです。
同じ履歴を共有する判断は同じ変数で表し、未来の結果を先に知って判断する構造を許しません。

## 8.12 Multi-step の augmented initialization

multi-step lookahead の初期条件は、現在の候補だけでなく
将来の fantasy decision を表す補助変数についても構成します。
branching factor が増えると augmented optimization dimension は急速に増加します。

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

## 8.14 Mixed one-shot の探索原理

augmented row ごとに categorical assignment を区別する必要があります。
すべての row で同一 category を固定することは、一般には正しい定式化ではありません。
小規模な離散集合なら row-specific assignment を列挙し、
連続部分問題を条件付きで最適化できます。

例えば仮想観測が等確率の二通りで、終端判断のカテゴリA・Bの価値がそれぞれ $(1,0)$ と $(0,1)$ なら、
結果ごとにカテゴリを選ぶ期待価値は1です。二つの補助判断を同じカテゴリへ固定すると、最良でも0.5になります。
異なる観測結果への適応を数値変数の割当で消してしまうことが、単なる探索精度の差以上の問題です。

## 8.15 Mixed one-shot の組合せ爆発

1 row あたりの categorical assignment 数を C、
augmented row 数を q_aug とすると、単純列挙数は

    C^(q_aug)

です。

複数 categorical dimension がある場合、C 自体が各 dimension の category 数の積になります。

したがって q=1 であっても fantasy count が増えると列挙数は急増します。

この指数的増加は列挙法の計算可能性を制限するため、
大規模な場合には近似探索や構造利用が必要です。

## 8.16 One-shot と ordinary q-batch の比較

| 項目 | ordinary q-batch | one-shot |
| --- | --- | --- |
| optimizer variable | q candidates | q candidates + auxiliary variables |
| optimizer batch size | q | q_aug |
| 返す実評価候補 | 最適化したq点 | 拡張解から実評価に対応するq点を抽出 |
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
| q点を貪欲に構築 | sequential + 未完了の入力集合 |
| 実行中評価を考慮 | asynchronous + 未完了の入力集合-aware acquisition |
| KG | one-shot + specialized KG initialization |
| multi-step lookahead | one-shot augmented tree |
| mixed one-shot | row-specific categorical handling |

数値手法の適用条件だけでなく、各決定変数がどの観測と判断を表すかを確認することが重要です。

## この章で覚えておくこと

同時候補、逐次選択、先読みの補助判断はそれぞれ異なる最適化変数を使います。
補助変数を追加すると将来の適応的な判断を表せますが、次元と離散割当の数も増えます。
現在の実験候補だけを返すまで、各変数の意味と制約を保つ必要があります。

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

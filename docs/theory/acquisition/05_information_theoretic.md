# 5. Information-theoretic Acquisition

## 5.1 情報量を観測価値として使う

Information-theoretic acquisition は、候補点の目的値そのものではなく、**特定の未知量について観測がどれだけ情報を与えるか**を候補価値として扱います。

重要なのは「情報量」という語だけでは不十分で、何についての情報かを明示することです。

例えば Bayesian Optimization では、

- optimizer location \(x^*\)
- optimum value \(f^*\)
- Pareto set / Pareto front

などが情報獲得対象になり得ます。

Active Learning の predictive information gain は別の対象を持つため、同じ mutual information を使っていても目的は異なります。

## 5.2 Entropy

離散確率変数 \(Z\) の entropy は

\[
H(Z)
=
-\sum_z p(z)\log p(z)
\]

です。

連続変数では differential entropy

\[
H(Z)
=
-\int p(z)\log p(z)\,dz
\]

を考えます。

Entropy は uncertainty の尺度ですが、acquisition で重要なのは単なる現在の entropy ではなく、**候補を観測することで対象の uncertainty がどれだけ減るか**です。

## 5.3 Mutual Information

未知量 \(Z\) と候補 \(x\) で得られる将来観測 \(Y_x\) の mutual information は

\[
I(Z;Y_x\mid\mathcal D_n)
=
H(Z\mid\mathcal D_n)
-
\mathbb E_{Y_x}
[
H(Z\mid\mathcal D_n,Y_x)
]
\]

と書けます。

これは、

```text
観測前の uncertainty
    -
観測後に残る expected uncertainty
```

です。

対称性から

\[
I(Z;Y_x\mid\mathcal D_n)
=
H(Y_x\mid\mathcal D_n)
-
\mathbb E_Z[
H(Y_x\mid\mathcal D_n,Z)
]
\]

とも書けます。実際のアルゴリズムでは、どちらの表現が計算しやすいかが重要になります。

## 5.4 Entropy Search の考え方

Entropy Search 系は、最適化問題を「未知の optimum について効率よく学習する問題」として捉えます。

optimizer location を

\[
x^*
=
\arg\max_{x\in\mathcal X} f(x)
\]

とすれば、Predictive Entropy Search（PES）などは \(x^*\) に関する情報獲得を考えます。

この考え方は直接 improvement を最大化する EI と異なり、現在の objective value が高くなくても optimum identification に有益な候補を評価し得ます。

## 5.5 Max-value Entropy Search

Max-value Entropy Search（MES）は optimizer location \(x^*\) ではなく、最大値

\[
f^*
=
\max_{x\in\mathcal X} f(x)
\]

について得られる情報を最大化します。

概念的には

\[
\alpha_{\mathrm{MES}}(x)
=
I(Y_x;f^*\mid\mathcal D_n)
\]

です。

optimizer の位置分布より scalar な optimum value の分布を扱うことで、情報理論的 BO をより計算しやすくすることが MES の重要な発想です。

## 5.6 MES の計算構造

MES では一般に、

1. posterior から optimum value \(f^*\) の samples を得る。
2. 各 sampled optimum value の条件下で候補観測の entropy reduction を評価する。
3. samples について平均する。

という構造を持ちます。

実際の \(f^*\) sampling には近似が必要です。探索空間の離散化、posterior function sampling、極値分布近似など、実装により計算方法は異なります。

したがって「MES の理論的 criterion」と「optimum-value samples の生成方法」は区別する必要があります。

## 5.7 GIBBON

GIBBON は batch Bayesian Optimization を意識した information-theoretic acquisition で、Gaussian mutual information と MES 型の optimum-value information を組み合わせた tractable な lower-bound formulation を利用します。

batch setting では、候補間の redundancy を無視すると似た候補を複数選びやすくなります。GIBBON は候補集合の correlation structure を利用し、情報の重複を考慮します。

したがって GIBBON を「MES を q 点へ単純拡張しただけ」と理解するのは不十分です。

## 5.8 Information gain と exploration

Information-theoretic acquisition は exploration-oriented に見えますが、単なる posterior variance 最大化とは異なります。

posterior variance は

```text
その点の関数値がどれだけ不確実か
```

を評価します。

MES は

```text
その観測が optimum value についてどれだけ教えるか
```

を評価します。

不確実でも optimum と無関係な領域は、MES では価値が低くなり得ます。

## 5.9 Observation noise

noise がある場合、将来観測 \(Y_x\) と latent function value \(f(x)\) は区別されます。

\[
Y_x=f(x)+\epsilon
\]

information gain は「noise を含む観測から latent optimum について何を学べるか」を評価する必要があります。noise が大きい観測は predictive entropy が高くても、その多くが irreducible observation noise なら optimum に関する情報価値は高くありません。

## 5.10 Batch information gain

候補集合 \(X\) に対しては

\[
I(Y_X;Z\mid\mathcal D_n)
\]

を考えます。

候補間に強い correlation がある場合、各候補の pointwise information gain の和は joint information gain を過大評価し得ます。

このため information-theoretic acquisition でも [Batch / Noisy](04_batch_noisy.md) で扱った joint posterior の考え方が重要です。

## 5.11 MES と Knowledge Gradient の違い

MES と Knowledge Gradient（KG）はどちらも myopic improvement より先を意識しますが、最適化対象は異なります。

```text
MES
    optimum value について得られる information

KG
    観測後の最終 decision value の expected improvement
```

MES は entropy / mutual information、KG は value of information の考え方です。KG は次章 [Lookahead](06_lookahead.md) で扱います。

## 5.12 Active Learning の information gain との違い

Active Learning でも mutual information を利用できますが、情報対象が異なります。

例えば predictive information gain では、target distribution 上の未知 prediction について候補観測が与える情報を評価します。

したがって、

```text
MES
    optimum value が情報対象

EPIG
    target prediction が情報対象
```

です。

EPIG は [Active Learning](09_active_learning.md) で扱います。

## 5.13 BoTorch / robotorchan との対応

BoTorch は MES と GIBBON を含む information-theoretic acquisition を提供しています。robotorchan は BoTorch native で利用可能な標準手法を再実装しません。

利用上の位置付けは [Information-theoretic optimization guide](../../optimization/information_theoretic.md)、現在の統合状況は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 5.14 まとめ

Information-theoretic acquisition を理解するときは、必ず

\[
\text{information about what?}
\]

を確認します。

MES は optimum value、PES は optimizer location、predictive AL は prediction についての情報を扱います。

「uncertainty が大きい点を選ぶ」ことと「意思決定対象について情報を最大化する」ことは同じではありません。


## References

- Wang, Z. and Jegelka, S. (2017), *Max-value Entropy Search for Efficient Bayesian Optimization*. ICML.
- Moss, H. B. et al. (2021), *GIBBON: General-purpose Information-Based Bayesian Optimisation*. JMLR.
- BoTorch documentation, *Acquisition Functions*, for the current native MES and lower-bound max-value entropy interfaces.

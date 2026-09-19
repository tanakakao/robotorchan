# High-dimensional Search Strategies

## 直感

surrogate の入力次元を下げても、acquisition optimizer が元の D 次元 box を探索するなら
候補探索の難しさは残ります。高次元 BO では surrogate modeling と search-space optimization を
分離して考えます。

## Original-space search

通常の `optimize_acqf` は original bounds 内で acquisition を最適化します。D が大きいと
multi-start optimization 自体が難しくなります。

## REMBO / HeSBO

REMBO は低次元の dense random embedding を固定し、その空間で探索します。HeSBO は sparse
hash embedding を使い、projection cost を抑えます。どちらも data-driven reducer ではなく
search embedding です。

## ALEBO

ALEBO は固定線形 embedding に加え、original box に対応する feasible polytope と
projection-conditioned Mahalanobis geometry を扱います。

robotorchan では `ALEBOStrategy` が search geometry を管理し、`ALEBOGP` が embedded
coordinates 上の surrogate を担当します。単なる `ReducedGP` の reducer ではありません。

## TuRBO

TuRBO は global box 全体を直接探索せず、成功・失敗に応じて伸縮する local trust region を使います。
局所探索を複数回適応させることで高次元 acquisition optimization を扱います。

## BAxUS

BAxUS は sparse embedding の target dimension を探索中に拡張します。intrinsic dimension が
未知でも、初期から大きな latent dimension を固定せず段階的に search subspace を広げられます。

## Surrogate reduction との組合せ

```text
surrogate-side:
X -> reducer -> GP

search-side:
target coordinate -> candidate in original X -> acquisition
```

役割が違うため、`PCAGP` と REMBO/ALEBO を同じ「次元削減モデル」として数えません。
二重 embedding を導入する場合は bounds、pending points、gradient、candidate feasibility が
どの空間に属するかを明示する必要があります。

## Stateful strategy

TuRBO / BAxUS は観測結果を state に戻す必要があります。surrogate model は目的関数を評価せず、
strategy も surrogate training を暗黙に所有しないという責務分離を維持します。

詳細は [high-dimensional search strategies](../high_dimensional_search_strategies.md) を参照してください。

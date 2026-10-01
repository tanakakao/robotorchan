# Mixed and Discrete Acquisition Optimization

## 4.1 Search space

continuous、integer、categorical variable が同じ candidate space に存在できます。

## 4.2 Categorical variable

category code を連続値として最適化して最後に丸めるだけでは、
一般に category semantics を保存しません。
enumeration、specialized mutation、sampling、native mixed optimization などを使います。

## 4.3 Integer variable

integer は順序を持つ点で categorical と異なります。
rounding、integrality-aware operator、enumeration などが候補になります。

## 4.4 現在の実装

BoTorch mixed、Mixed GA、sampling、および一部 population backend が
mixed / integer capability を持ちます。
`mixed=True` だけで categorical support を意味しないため個別に確認します。

## 4.5 Model と optimizer

Mixed surrogate が category を model 化できることと、
optimizer が category を正しく探索できることは別条件です。

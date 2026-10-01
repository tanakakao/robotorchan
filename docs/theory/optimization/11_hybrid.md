# Hybrid Acquisition Optimization

## 11.1 Motivation

単一 optimizer の弱点を補うため、global exploration と local refinement を
異なる探索法で組み合わせます。

## 11.2 Global-to-local

sampling や population method で有望領域を探し、
gradient-based local optimization で refine する構成が代表例です。

## 11.3 Contract

phase ごとの capability、candidate representation、constraint semantics、
hand-off initialization、gradient availability を一貫させる必要があります。

現在の hybrid backend が存在することは、
任意の backend を自由に合成できることを意味しません。

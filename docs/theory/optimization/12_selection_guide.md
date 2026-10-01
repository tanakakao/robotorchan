# Acquisition Optimizer Selection Guide

## 12.1 Problem structure

optimizer 名より先に、continuous / integer / categorical、
gradient、constraints、q-batch / sequential / one-shot、高次元性を確認します。

## 12.2 基本

smooth continuous acquisition では multi-start gradient-based optimization が基本候補です。
gradient-free なら sampling、DE、CMA-ES、GA、PSO などを検討します。
Mixed space では変数型を保存できる optimizer を選びます。

## 12.3 Constraint

support の有無だけでなく native / penalty / feasibility-first など
enforcement semantics を確認します。

## 12.4 Batch

q-batch support と sequential support は別です。
one-shot では augmented optimization への対応も必要です。

## 12.5 Runtime

最終判断では [Optimization guides](../../optimization/README.md) と
optimizer capability metadata を確認します。
理論的に適用可能でも、現在の API がその組合せを公開しているとは限りません。

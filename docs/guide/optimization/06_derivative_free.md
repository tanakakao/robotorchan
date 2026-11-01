# Derivative-free / Sampling：獲得関数最適化の実装ガイド

## この章の問い

gradientなしでacquisitionを評価し、Random/Sobolやblack-box searchで最大値を探す。joint qはq×dを一つの候補として比較する。

数理的な正本は[理論の対応章](../../theory/optimization/06_derivative_free.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

finite samplesはglobal optimumの保証にならない。mixed variable、fixed features、constraintの意味を保存する。

## Sobol候補poolから獲得関数を最大化する

`sample_candidate_batches`は候補点を生成し、`select_best_sampled_batch`は与えられた有限poolの獲得値を比較する。
最良のサンプルが連続領域全体の大域最適解とは限らない。

## 有限poolのargmaxを実装する

[sampling.py](../../../src/robotorchan/optim/backends/sampling.py)の
`sample_candidate_batches()`は`_draw_random()`または`_draw_sobol()`へ分岐します。Sobolではqとdの積の次元で点を生成し、
`[N,q,d]`へreshapeしてboundsへ写像します。この形が、q点を一つの意思決定として比較するTheoryの変数に対応します。

`select_best_sampled_batch()`は勾配を無効にして獲得関数を評価し、N個のbatchに対してN個のscalarが返ることを確認してargmaxを選びます。
固定特徴やmixed domainの変換は、その評価より前に`optimize_acqf_sampling()`が処理します。有限集合上の最大化はローカル実装であり、
BoTorchへ連続最適化を依頼する経路ではありません。

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/31_acquisition_optimization.ipynb)を参照してください。

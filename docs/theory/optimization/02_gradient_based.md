# Gradient-based Acquisition Optimization

## 2.1 基本原理

微分可能な acquisition \(\alpha(X)\) に対して
\[
\nabla_X\alpha(X)
\]
を利用して局所最適化します。

## 2.2 Multi-start

acquisition は非凸なので、複数 initial conditions から最適化し、
最良解を選ぶ方法が基本です。

## 2.3 現在の実装

BoTorch backend に加え、Adam、AdamW、SGD を candidate-input optimizer として利用できます。
model training optimizer と同名でも、最適化対象は model parameter ではなく candidate です。

## 2.4 適用条件

candidate input に対する acquisition gradient が利用可能であることが重要です。
離散変数や tree ensemble では gradient-free method が自然な場合があります。

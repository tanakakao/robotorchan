# Derivative-free Acquisition Optimization

## 6.1 目的

gradient が利用できない、信頼できない、または離散構造を含む場合、
acquisition value の評価だけで探索します。

## 6.2 Sampling

Random / Sobol search は候補集合を生成し、その中から高い acquisition value を選びます。
Sobol sequence は低 discrepancy な点集合による空間充填を狙います。

## 6.3 Population methods

DE、CMA-ES、GA、PSO などは population を反復更新します。
詳細は [Evolutionary and population methods](07_evolutionary.md) を参照してください。

## 6.4 Tree surrogate

tree ensemble のように candidate gradient を前提にしない posterior では
gradient-free search が自然です。
`TreeEnsembleSearchStrategy` はこの責務に位置付けられます。

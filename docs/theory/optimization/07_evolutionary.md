# Evolutionary and Population-based Optimization

## 7.1 共通構造

複数候補を保持し、selection、mutation、recombination、
distribution update、velocity update などで探索します。

## 7.2 Differential Evolution

DE は個体間差分から mutation vector を作り、
crossover と selection で population を更新します。

## 7.3 CMA-ES

CMA-ES は探索分布の平均と covariance を適応させます。

## 7.4 Genetic Algorithm

GA は selection、crossover、mutation を基本要素とします。
Mixed GA は変数型に応じた operator で mixed structure を保持します。

## 7.5 Particle Swarm Optimization

PSO は particle の位置・速度と individual / global best を用いて更新します。

## 7.6 NSGA-II

NSGA-II は non-dominated sorting と crowding distance を利用する
multi-objective evolutionary optimization です。
`optimize_vector_nsga2` は scalar acquisition dispatcher とは別の primitive です。

## 7.7 BO での位置付け

expensive objective を大量評価するのではなく、
比較的安価な acquisition landscape を探索する backend として利用できます。

# Acquisition Optimization Initialization

## 3.1 役割

multi-start optimization では initial conditions が探索する basin を決めます。
したがって initialization は非凸探索の品質に関係します。

## 3.2 Raw samples と restart

典型的には search space から raw samples を生成し、
acquisition value に基づいて restart 候補を選びます。

## 3.3 One-shot

Knowledge Gradient などでは auxiliary / fantasy variables を含む
augmented input を最適化するため、通常の q-candidate initialization と異なります。

## 3.4 実装

詳細は [Acquisition initialization](../../optimization/initialization.md) を参照してください。

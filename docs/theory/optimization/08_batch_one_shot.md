# Batch and One-shot Acquisition Optimization

## 8.1 Joint q-batch

q-batch acquisition は
\[
\alpha(X_1,\ldots,X_q)
\]
を joint に評価します。
各点を独立に rank する問題とは異なります。

## 8.2 Sequential

joint optimization とは別に、1点ずつ選びながら pending / fantasy 情報を更新する
sequential greedy strategy があります。

## 8.3 One-shot

Knowledge Gradient 系では auxiliary fantasy variables を candidate と同時に最適化する
one-shot formulation があります。

## 8.4 Mixed

one-shot augmented batch に単一 categorical assignment を固定すると
内部構造を壊す場合があります。
acquisition と mixed optimizer の compatibility を別途確認します。

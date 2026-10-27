# Derivative-free / Sampling：獲得関数最適化の実装ガイド

## 1. この章の問い

gradientなしでacquisitionを評価し、Random/Sobolやblack-box searchで最大値を探す。joint qはq×dを一つの候補として比較する。

数理的な正本は[理論の対応章](../../../../theory/optimization/06_derivative_free.md)とする。本章は、そこで定義される問題の**候補の座標系、実装経路、契約、検証上の限界**を学ぶための日本語案内である。

## 2. 理論から実装まで

```text
surrogate.posterior(X)
      ↓
acquisition α(X) ← objective / constraints
      ↓
optimizer: bounds + fixed features + candidate feasibility
      ↓
candidate X*（joint q × d）
      ↓
実際のblack-box評価 → 次の観測を追加
```

robotorchanのrandom/Sobol探索と各gradient-free backendの責務を整理。

| 役割 | 実装の追跡先 | 検証すべき点 |
| --- | --- | --- |
| 数学的目的 | `docs/theory/optimization/06_derivative_free.md` | αを最大化する変数・制約・近似 |
| 公開API / dispatcher | `src/robotorchan/optim/` | 対応するoptimizer名、能力宣言、fallback |
| 具体的な探索backend | `src/robotorchan/optim/`とBoTorch optimizer | 実際の探索アルゴリズム、依存ライブラリ |
| 数値実行証拠 | [旧Optimization Mapping](../../optimization.md) | test assertions、実装境界、未対応条件 |

実装方法は**Implemented / Extended / Delegated**を分けて読む。BoTorchへ処理を委譲する名前と、robotorchanが実際に更新則を実装する名前を同等の独自アルゴリズムとして数えない。実際のbackend、candidate制約とposteriorの相互依存は旧Mappingを参照する。

## 3. 適用条件と失敗の解釈

finite samplesはglobal optimumの保証にならない。mixed variable、fixed features、constraintの意味を保存する。

最適化器が候補を返すことと、その候補が厳密な大域最適解であることは違う。候補座標の型、bounds、fixed features、constraint tolerance、q軸、one-shot追加変数、random seed等を適用先に合わせる。モデルの学習後posteriorが有限でも、candidate探索の収束は別途確認する。

## 4. 検証と使い方の入口

最低限の確認順序は、**(1) 問題の型→(2) 数理契約→(3) optimizer dispatch→(4) backend→(5) 制約と出力shape→(6) 再現実験**である。CI成功だけで全backend・全constraint・全qに対して正しいとは主張しない。

[Optimization利用ガイド](../../../../optimization/README.md)、[理論Optimization索引](../../../../theory/optimization/README.md)、[旧Optimization Mapping](../../optimization.md)、[Verification](../../verification.md)を参照。原文は章別照合完了まで削除しない。

## 5. Sobol候補poolから獲得関数を最大化する

`sample_candidate_batches`は候補点を生成し、
`select_best_sampled_batch`は与えられた有限poolの獲得値を比較する。
最良のサンプルが連続領域全体の大域最適解とは限らない。

```python
import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends.sampling import (
    sample_candidate_batches,
    select_best_sampled_batch,
)

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = UpperConfidenceBound(model=model, beta=0.5)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
samples = sample_candidate_batches(bounds, num_samples=32, q=1, seed=7)
candidate, score = select_best_sampled_batch(acqf, samples)
assert samples.shape == (32, 1, 1)
assert candidate.shape == (1, 1)
assert torch.isfinite(score).all()
assert torch.all(candidate >= bounds[0])
assert torch.all(candidate <= bounds[1])
```

この例の対象はcontinuous・q=1の有限poolであり、
mixed変数や候補制約を含む探索結果の可行性を保証しない。

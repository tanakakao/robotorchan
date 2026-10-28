# Hybrid optimization：獲得関数最適化の実装ガイド

## 1. この章の問い

global gradient-free探索の候補をgradient-based local refinementの初期点として渡す段階的最適化。

数理的な正本は[理論の対応章](../../../../theory/optimization/11_hybrid.md)とする。本章は、そこで定義される問題の**候補の座標系、実装経路、契約、検証上の限界**を学ぶための日本語案内である。

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

robotorchan hybrid backend→BoTorch local refinementのhandoffで`batch_initial_conditions`を明示する。

| 役割 | 実装の追跡先 | 検証すべき点 |
| --- | --- | --- |
| 数学的目的 | `docs/theory/optimization/11_hybrid.md` | αを最大化する変数・制約・近似 |
| 公開API / dispatcher | `src/robotorchan/optim/` | 対応するoptimizer名、能力宣言、fallback |
| 具体的な探索backend | `src/robotorchan/optim/`とBoTorch optimizer | 実際の探索アルゴリズム、依存ライブラリ |
| 数値実行証拠 | [旧Optimization Mapping](../../optimization.md) | test assertions、実装境界、未対応条件 |

実装方法は**Implemented / Extended / Delegated**を分けて読む。BoTorchへ処理を委譲する名前と、robotorchanが実際に更新則を実装する名前を同等の独自アルゴリズムとして数えない。実際のbackend、candidate制約とposteriorの相互依存は旧Mappingを参照する。

## 3. 適用条件と失敗の解釈

joint q×dを保持し、global候補の独立top-qへの置換をしない。非微分posteriorならlocal refinementは不適切。

最適化器が候補を返すことと、その候補が厳密な大域最適解であることは違う。候補座標の型、bounds、fixed features、constraint tolerance、q軸、one-shot追加変数、random seed等を適用先に合わせる。モデルの学習後posteriorが有限でも、candidate探索の収束は別途確認する。

## 4. 検証と使い方の入口

最低限の確認順序は、**(1) 問題の型→(2) 数理契約→(3) optimizer dispatch→(4) backend→(5) 制約と出力shape→(6) 再現実験**である。CI成功だけで全backend・全constraint・全qに対して正しいとは主張しない。

[Optimization利用ガイド](../../../../optimization/README.md)、[理論Optimization索引](../../../../theory/optimization/README.md)、[旧Optimization Mapping](../../optimization.md)、[Verification](../../verification.md)を参照。原文は章別照合完了まで削除しない。

## 5. Differential EvolutionからBoTorch局所探索へ接続する

以下は公開dispatcherの`optimizer="hybrid"`を使用し、
Differential Evolution（DE）で得たjoint候補を
BoTorchの局所最適化の`batch_initial_conditions`に渡す実行例である。
内部handoffでは`[q,d]`から`[1,q,d]`へ次元が追加され、
局所探索後には実験候補`[q,d]`を返す。

```python
import torch
from botorch.acquisition.analytic import PosteriorMean
from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = PosteriorMean(model=model)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
candidate, score = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    optimizer="hybrid",
    seed=7,
    optimizer_options={
        "global_optimizer": "de",
        "global_options": {"options": {"maxiter": 2, "popsize": 4}},
        "num_restarts": 1,
    },
)
assert candidate.shape == (1, 1)
assert torch.isfinite(candidate).all()
assert torch.isfinite(score).all()
assert torch.all(candidate >= bounds[0])
assert torch.all(candidate <= bounds[1])
```

この例はDEから局所探索への実行経路と候補のboundsを確認する。
DE自体の大域最適性、複数restart、mixed変数、非線形制約の可行性は検証していない。

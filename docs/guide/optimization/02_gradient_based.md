# Gradient-based：獲得関数最適化の実装ガイド

## 1. この章の問い

必要なのはcandidate Xに対する∇α(X)であり、GPパラメータ学習の勾配とは違う。reparameterized MCサンプルで候補勾配を評価できる。

数理的な正本は[理論の対応章](../../theory/optimization/02_gradient_based.md)とする。本章は、そこで定義される問題の**候補の座標系、実装経路、契約、検証上の限界**を学ぶための日本語案内である。

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

BoTorch `optimize_acqf`のmulti-start local searchと、robotorchan内のfirst-order/gradient backendを対応付ける。

| 役割 | 実装の追跡先 | 検証すべき点 |
| --- | --- | --- |
| 数学的目的 | `docs/theory/optimization/02_gradient_based.md` | αを最大化する変数・制約・近似 |
| 公開API / dispatcher | `src/robotorchan/optim/` | 対応するoptimizer名、能力宣言、fallback |
| 具体的な探索backend | `src/robotorchan/optim/`とBoTorch optimizer | 実際の探索アルゴリズム、依存ライブラリ |
| 数値実行証拠 | [旧Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md) | test assertions、実装境界、未対応条件 |

実装方法は**Implemented / Extended / Delegated**を分けて読む。BoTorchへ処理を委譲する名前と、robotorchanが実際に更新則を実装する名前を同等の独自アルゴリズムとして数えない。実際のbackend、candidate制約とposteriorの相互依存は旧Mappingを参照する。

## 3. 適用条件と失敗の解釈

入力gradientが存在することだけではoptimizerや非線形制約への適合性は証明されない。

この章のBoTorch backendは複数の初期点から局所探索する。`num_restarts`と`raw_samples`は学習データ数ではなく探索初期化の設定である。返却候補のshape・bounds・有限性は第5節で確認するが、非凸獲得関数の大域最大値であることまでは保証しない。非線形制約の可行性は[Candidate constraints](05_constraints.md)で別に検証する。

## 4. 検証と使い方の入口

まず第5節の`optimize_acqf`例で、(1)モデルのposterior、(2)UCBの候補スコア、(3)BoTorch backendによる最大化、(4)候補`[q,d]`の返却を順に追う。その後、初期点の指定は[Initialization](03_initialization.md)、カテゴリ列の固定列挙は[Mixed / Discrete](04_mixed_discrete.md)で確認する。

[Optimization利用ガイド](../../optimization/README.md)、[理論Optimization索引](../../theory/optimization/README.md)、[旧Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md)、[Verification](../../../.github/project-records/theory-guide/mapping/verification.md)を参照。原文は章別照合完了まで削除しない。

## 5. 公開dispatcherによる候補生成の最小例

以下はモデルの学習とは別に、候補`X`に対してanalytic UCBを最大化する例。
`robotorchan.optim.optimize_acqf`を使い、BoTorch backendへ処理を委譲する。
返り値は獲得関数値と候補点であり、真の目的関数を観測した値ではない。

```python
import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = UpperConfidenceBound(model=model, beta=0.5)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
candidate, value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    optimizer="botorch",
    num_restarts=2,
    raw_samples=16,
)
assert candidate.shape == (1, 1)
assert torch.isfinite(candidate).all()
assert torch.isfinite(value).all()
assert torch.all(candidate >= bounds[0])
assert torch.all(candidate <= bounds[1])
```

ここではmulti-startの局所最適化を利用しており、厳密な大域最適性は示さない。
`q=1`・連続boundsという条件での使用例である。
混合変数、非線形制約、one-shotの追加変数は各対応章で検証する。

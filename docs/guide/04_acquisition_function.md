# Acquisition Function：理論から評価・候補生成まで

## 1. この章で解決する問題

ベイズ最適化において、代理モデルが返す**確率的な予測**を、次に評価する候補の**意思決定スコア**へ変換する方法を理解する。獲得関数の数学的定義と個別の式は[理論第4章](../theory/04_acquisition_function.md)および[Acquisition理論体系](../theory/acquisition/README.md)を正本とする。本章はその計算がrobotorchan・BoTorchのどのコンポーネントで行われるかを説明する。

観測からモデルを更新した後、候補集合 `X` のposteriorを使って `α(X)` を評価する。ただし**獲得関数を評価することと、その関数を最大化して実行可能な候補を生成することは別の処理**である。

```text
観測 (train_X, train_Y) → モデルの学習
                              ↓
候補 X → model.posterior(X) → objective / utility
                              ↓
                      acquisition α(X)
                              ↓
               acquisition optimizer (bounds, constraints)
                              ↓
                         次の候補 X*
```

## 2. Posteriorからutilityを構築する

GP posteriorの単一点周辺分布は平均 `μ(x)` と標準偏差 `σ(x)` を持つ。一方、`q>1` の候補をまとめて扱うMC獲得関数では、複数候補間の相関を含むjoint posteriorからサンプルを取る。独立した各点の平均・分散だけで、一般のq-batch utilityを正しく復元できるとは限らない。

- **Analytic（Delegated）:** 対応する正規分布の仮定のもとで閉形式評価が可能なとき、BoTorchの `LogExpectedImprovement`、`UpperConfidenceBound` などを利用する。これらをrobotorchan独自実装と呼ばない。
- **MC（Delegated）:** BoTorch `qLogExpectedImprovement`、`qLogNoisyExpectedImprovement` などがposterior samplesをutilityに写して平均する。sample shape・`q`・objectiveとposterior output dimensionsの互換性が必要になる。
- **Local（Implemented/Extended）:** robotorchan固有の回帰／分類Active Learningスコア、情報量・境界探索などは別途ソースと数式を照合する。すべてをBoTorch-nativeやGaussian analytic acquisitionへ還元しない。

Improvementで使うincumbent `best_f` は目的の向きと同じスケールで定義する。最大化・最小化を混ぜたまま利用しない。標準EIより数値的安定性を考慮したLogEI系をデフォルト候補として扱うが、各手法の適用条件は[標準獲得関数ガイド](../optimization/standard_acquisition.md)に従う。

## 3. 実装の責務・公開API

| 論理コンポーネント | 実装・公開入口 | 区分と限界 |
| --- | --- | --- |
| Posterior | `robotorchan.models.SingleTaskGP.posterior` | **Delegated**：基礎GP posteriorはBoTorchに委譲。獲得関数ではない |
| 標準Analytic/MC BO | `botorch.acquisition` | **Delegated**：robotorchanは標準クラスを複製・一括再exportしない |
| 目的値や制約のbridge | `src/robotorchan/acquisition/composition.py`、`robotorchan.acquisition.make_*_acquisition` | **Extended**：入力モデル、objective、constraintの意味を上流クラスへ接続。独立した新しい獲得アルゴリズムではない |
| Local Active Learning | `src/robotorchan/acquisition/active_learning/` | **Implemented**：PosteriorVariance、Straddle、BALD等。確率の種類に応じた契約が異なる |
| Registry | `src/robotorchan/acquisition/registry.py`、`get_acquisition_registry_entry` | **Implemented（metadata）**：存在・compatibility metadataは数式の独立検証を意味しない |
| Optimizer接続 | `acquisition/optimizer_compatibility.py` と `optim/` | **Extended/Delegated**：経路とbackendの責務を分離。登録済みというだけで実行可能性は保証されない |

詳細な静的調査記録は[既存Acquisition Mapping](../../.github/project-records/theory-guide/mapping/acquisition.md)に保全している。旧ファイルは移行の照合検証が完了するまで削除しない。

## 4. 最小の実装例：GP → analytic acquisition → optimizer

```python
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(2 * torch.pi * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
fit_gpytorch_mll(model.make_mll())

# 最大化問題。best_fの定義とモデル出力のスケールを揃える。
acqf = LogExpectedImprovement(model=model, best_f=train_Y.max())
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
candidate, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=32,
)
print(candidate, acq_value)
```

この例は**連続入力・単一出力・q=1・制約なし**の最大化を扱う。候補点の獲得関数値は、候補点で実際に高価な目的関数を測った値ではない。使用APIは公開ソースと既存利用ガイドに照らして選定したが、その記載だけで任意条件の動作を保証するものではない。

## 5. q-batch、Objective、Constraintの契約

### q-batch

`q` は同時に評価する候補点数であり、modelのposterior batch dimensionsとは別。analytic EIの代表例は`q=1`、`qLogExpectedImprovement`はjoint MCサンプルからq候補のimprovementを評価する。`q=1`専用のLocal Active Learning関数を、単純に`q>1`へ拡張したと考えてはいけない。

### ObjectiveとConstraint

- **Objective:** 複数出力posteriorや分類確率を、利用目的に応じたutilityへ変換する。単なる任意の出力列選択と同一ではない。
- **Outcome constraints:** posteriorサンプルの目的値・可行性を獲得関数のutility側で扱う。
- **Candidate constraints:** 候補点 `X` 自体の許容集合をoptimizer側で扱う。両者を同じAPI契約とみなさない。

`src/robotorchan/acquisition/composition.py`のbridgeが扱えるmodel/objective/constraintの組合せは限定される。例えば単一無制約regression objective向け`make_qei_acquisition`と、constrained / multiobjective / classification用のhelperを交換可能と推測しない。

### One-shot / mixed

Knowledge Gradientなどはfantasyを含むone-shot最適化が必要になる。既存[Acquisition Mapping](../../.github/project-records/theory-guide/mapping/acquisition.md)では、通常mixed optimizerと`MIXED_ONE_SHOT`経路を区別している。一般の`q`制約だけで実行可能と判定しない。

## 6. 検証範囲と未検証事項

[既存Verification Mapping](../../.github/project-records/theory-guide/mapping/verification.md)のacquisition検証には、独立参照値、サンプル単位の数値照合、特定のassertionが記録されている。一方、registry登録、型・shape検査、有限値の確認、CI成功は、それだけでEI等の数式同値性やoptimizerの大域最適性を証明しない。**property、入力条件、dtype、seed、許容誤差、実際に比較した値**が一致する範囲のみ検証済みとする。

- Analytic/MCの数値同値性は対象モデル、sampling、目的関数とノイズ条件に限定される。
- `q`、pending points、fantasization、mixed variables、制約などの全組合せを検証済みとは扱わない。
- Classification ALのclass probabilityとlatent GP posteriorを同一視しない。
- Acquisition値の検証と、候補探索が収束・実行可能になることの検証を分ける。

## 7. 関連章・参照先

- [理論第4章](../theory/04_acquisition_function.md)、[Acquisition全12章](../theory/acquisition/README.md)
- [BOの実装ガイド](01_bayesian_optimization.md)、[GP](02_gaussian_process.md)
- [獲得関数の利用ガイド](../optimization/README.md)、[標準獲得関数](../optimization/standard_acquisition.md)
- [旧Acquisition Mapping](../../.github/project-records/theory-guide/mapping/acquisition.md)、[Verification](../../.github/project-records/theory-guide/mapping/verification.md)
- [統合Notebook一覧](../../examples/README.md)

改善、信頼上界、情報獲得、Lookahead、多目的、Active Learning の詳細は、この Guide の獲得関数各章を参照してください。本章がそれらの数式検証と移行の完了を宣言するものではない。

## 理論章から分離した実装上の条件

### 4.15 Theory、利用方法、実装状況を分けて読む

robotorchan では獲得関数関連の情報を次のように分離します。

| 知りたいこと | 参照先 |
| --- | --- |
| 獲得関数全体の概要 | 本章 |
| 数式、理論、仮定、手法間の関係 | [Acquisition Function Theory](../theory/acquisition/README.md) |
| API、コード例、実際の利用方法 | [Optimization guide](../optimization/README.md) |
| 現在の対応範囲と制約 | [Acquisition integration status](../../docs-internal/contracts/optimization/acquisition-integration.md) |

robotorchan は BoTorch-first を基本とします。Theory に掲載されている手法が
robotorchan 固有実装であるとは限らず、BoTorch native の手法を直接利用する場合も
あります。

#### 4.5 Analytic と Monte Carlo：実装上の補足

$`S`$ を増やすと一般に Monte Carlo 近似誤差は小さくなりますが、計算量は増えます。
実際のBoTorchではQMC samplingなどを利用し、単純な独立乱数より効率的に
期待値を近似する場合があります。

#### 4.5 Analytic と Monte Carlo：実装上の補足

MC は batch、multi-output、nonlinear objective、constraints などへ拡張しやすく、
BoTorch の獲得関数設計でも重要な役割を持ちます。

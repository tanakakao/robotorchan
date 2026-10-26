# Optimizer selection：獲得関数最適化の実装ガイド

## 1. この章の問い

variable types、constraints、gradients、q、search geometry、evaluation costから最適化方法を選ぶ。

数理的な正本は[理論の対応章](../../../../theory/optimization/12_selection_guide.md)とする。本章は、そこで定義される問題の**候補の座標系、実装経路、契約、検証上の限界**を学ぶための日本語案内である。

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

robotorchan dispatcher、直接BoTorch呼出し、named strategy、optimizer compatibility metadataを区別する。

| 役割 | 実装の追跡先 | 検証すべき点 |
| --- | --- | --- |
| 数学的目的 | `docs/theory/optimization/12_selection_guide.md` | αを最大化する変数・制約・近似 |
| 公開API / dispatcher | `src/robotorchan/optim/` | 対応するoptimizer名、能力宣言、fallback |
| 具体的な探索backend | `src/robotorchan/optim/`とBoTorch optimizer | 実際の探索アルゴリズム、依存ライブラリ |
| 数値実行証拠 | [旧Optimization Mapping](../../optimization.md) | test assertions、実装境界、未対応条件 |

実装方法は**Implemented / Extended / Delegated**を分けて読む。BoTorchへ処理を委譲する名前と、robotorchanが実際に更新則を実装する名前を同等の独自アルゴリズムとして数えない。実際のbackend、candidate制約とposteriorの相互依存は旧Mappingを参照する。

## 3. 適用条件と失敗の解釈

登録されていても全入力形状・制約・one-shot・backendで実行できるわけではない。

最適化器が候補を返すことと、その候補が厳密な大域最適解であることは違う。候補座標の型、bounds、fixed features、constraint tolerance、q軸、one-shot追加変数、random seed等を適用先に合わせる。モデルの学習後posteriorが有限でも、candidate探索の収束は別途確認する。

## 4. 検証と使い方の入口

最低限の確認順序は、**(1) 問題の型→(2) 数理契約→(3) optimizer dispatch→(4) backend→(5) 制約と出力shape→(6) 再現実験**である。CI成功だけで全backend・全constraint・全qに対して正しいとは主張しない。

[Optimization利用ガイド](../../../../optimization/README.md)、[理論Optimization索引](../../../../theory/optimization/README.md)、[旧Optimization Mapping](../../optimization.md)、[Verification](../../verification.md)を参照。原文は章別照合完了まで削除しない。


## 5. 11種類のnamed optimizerと実際のdispatch

[旧Optimization Mapping](../../optimization.md)の固定区間182–205を、Phase 3 routingで指定された[基礎ガイド](01_foundations.md)と本章のselection guideへ分担して移す。公開入口は src/robotorchan/optim/dispatch.py の optimize_acqf(acq_function, bounds, q, *, optimizer=...) である。すべての名前が **acquisition A(X)の最大化** を目的とするが、異なる候補生成則と近似精度を持つ。optimizerを変えることはacquisition数式を変えることではない。

| optimizer名 | 数学的な探索方法 | dispatch先と所有権 |
| --- | --- | --- |
| botorch | 複数初期値から連続候補を局所最大化 | backends/botorch.py の optimize_acqf_botorch → BoTorchへDelegated。named routeはmixed variable_spaceを直接受け付けない |
| torch_adam | 勾配によるAdam更新 | backends/torch.py の optimize_acqf_torch → BoTorch optimize_acqf + gen_candidates_torch + torch.optim.Adam。Extended/Delegated |
| torch_adamw | 分離weight decayを伴うAdamW更新 | 同上 + torch.optim.AdamW。weight decayをGP priorと解釈しない |
| torch_sgd | 勾配によるSGD更新 | 同上 + torch.optim.SGD。収束保証を意味しない |
| random | 独立な一様乱数候補batchを多数評価 | backends/sampling.py、torch.randとargmaxでImplemented |
| sobol | q×d次元のscrambled Sobol点による候補batch評価 | backends/sampling.py、SobolEngineとargmaxでImplemented |
| de | Differential Evolutionによるpopulation探索 | backends/differential_evolution.py、optimize_acqf_de |
| cmaes | 共分散行列適応進化戦略 | backends/cmaes.py、optimize_acqf_cmaes |
| ga | 選択・交叉・変異を使う遺伝的探索 | 通常はbackends/genetic_algorithm.py、mixed variable_spaceならbackends/mixed_genetic_algorithm.py |
| pso | 粒子群の位置・速度更新による探索 | backends/pso.py、optimize_acqf_pso |
| hybrid | 複数探索手順の組合せ | backends/hybrid.py、optimize_acqf_hybrid。詳細なstageはそのbackend実装で確認 |

### 5.1 samplingの選択式とqの意味

randomおよびsobolは $N$ 個の候補集合 $X^{(k)}\in\mathbb R^{q\times d}$ をサンプリングして

$$
k^\star=\operatorname*{arg\,max}_{1\le k\le N} A(X^{(k)}),
\qquad X^\star=X^{(k^\star)}
$$

を返す。つまり $q$ 点を個別に採点して上位を束ねるのではなく、**q点のjoint acquisition score**を比較する。sample_candidate_batchesの出力shapeは[N,q,d]。select_best_sampled_batchは各batchにつき1スカラーを要求し、そうでなければValueErrorを送出する。mixed入力ではrandomは合法な整数・カテゴリ集合から抽出し、sobolは連続一様座標を等幅inverse-CDF binへ写像する。いずれもglobal acquisition最適性を証明しない。

### 5.2 パラメータをどのbackendへ渡すか

| 引数・機能 | botorch | torch_adam / torch_adamw / torch_sgd | random / sobol / DE / CMA-ES / GA / PSO / hybrid |
| --- | --- | --- | --- |
| num_restarts、raw_samples、options | BoTorchへ渡す | BoTorch restart・初期化へ渡す | BoTorch固有optionsとbatch_initial_conditionsは拒否。samplingはoptimizer_optionsのnum_samplesを用い、省略時はraw_samplesか512 |
| optimizer_options | 非空なら拒否 | 非空なら拒否 | backend固有パラメータとして解釈 |
| seed | dispatcherからの直接指定は受けない（native初期化を利用） | 直接指定を拒否 | samplingとevolutionary系へ渡す |
| sequential=True | native経路で可 | 拒否 | joint q方式のため拒否。明示的なsequential APIを別途選ぶ |
| ic_generator / ic_gen_kwargs | BoTorch nativeへ渡す | 拒否 | 拒否 |
| fixed_features、candidate constraints、mixed variables | backend capabilitiesに依存 | fixed_featuresは処理、candidate constraintsは明示拒否、mixed非対応 | 各backendのcapabilityに依存。mixed時のGAは別backendへ分岐 |

上表はsrc/robotorchan/optim/dispatch.pyによる**dispatch契約**である。capabilitiesに対応があることと、各backendで特定の等式・非線形制約の実行が成功することは同一ではない。とくにPyTorch backendはCandidateConstraints.has_constraintsが真ならValueErrorを送出し、制約違反を黙って許容するわけではない。direct seed指定もtorch_*では拒否される。

### 5.3 最小例：SobolとRandomの比較

この例は既存acquisitionの入力を前提とせず、BoTorch AcquisitionFunctionの最小サブクラスでdispatch引数を示す。実問題ではこの関数の代わりに学習済みモデルに基づくqEI等を渡す。

~~~python
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from robotorchan.optim.dispatch import optimize_acqf


class QuadraticAcquisition(AcquisitionFunction):
    def __init__(self):
        super().__init__(model=None)

    def forward(self, X):
        return -((X - 0.4) ** 2).sum(dim=(-2, -1))


bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
acqf = QuadraticAcquisition()
candidate, score = optimize_acqf(
    acqf,
    bounds,
    q=2,
    optimizer="sobol",
    optimizer_options={"num_samples": 128},
    seed=29,
)
# candidateは[q=2,d=1]、scoreは最大acquisition値のスカラー
~~~

q=2のsampled *batch*を最適化する例であり、局所解や大域解の最適性を主張しない。sampled pointsとargmaxの性質は tests/optim/backends/test_phase9_sampling_optimizer_reference.py の既存assertionで限定的に検証されている。

### 5.4 named dispatcherに含まれない経路

NSGA-II（多目的探索）、BoTorch native mixed、mixed one-shotは上記OptimizerNameの11候補に含まれず、個別のAPIで利用する。特にone-shotのfantasy補助変数を通常のmixed最適化へそのまま渡さない。詳細は[Mixed/Discrete](04_mixed_discrete.md)、[Batch/one-shot](08_batch_one_shot.md)、[Evolutionary](07_evolutionary.md)で確認する。

**証拠区分：** src/robotorchan/optim/dispatch.py、backends/torch.py、backends/sampling.py、旧Optimization Mappingの静的照合。全11backendの実行・数値精度を今回検証したわけではない。174見出しの全文移行は未検収である。

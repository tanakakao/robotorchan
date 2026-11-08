# Optimizer selection：獲得関数最適化の実装ガイド

## この章の問い

variable types、constraints、gradients、q、search geometry、evaluation costから最適化方法を選ぶ。

数理的な正本は[理論の対応章](../../theory/optimization/12_selection_guide.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

登録されていても全入力形状・制約・one-shot・backendで実行できるわけではない。

## 11種類のnamed optimizerと実際のdispatch

[最適化の基礎](01_foundations.md)と併せて、公開の dispatcher と backend を確認します。公開入口は
src/robotorchan/optim/dispatch.py の optimize_acqf(acq_function, bounds, q, *, optimizer=...) である。
すべての名前が **acquisition A(X)の最大化** を目的とするが、異なる候補生成則と近似精度を持つ。optimizerを変えることはacquisition数式を変えることではない。

- **botorch**
  数学的な探索方法：複数初期値から連続候補を局所最大化
  dispatch先と所有権：backends/botorch.py の optimize_acqf_botorch → BoTorchへDelegated。
  named routeはmixed variable_spaceを直接受け付けない

- **torch_adam**
  数学的な探索方法：勾配によるAdam更新
  dispatch先と所有権：backends/torch.py の optimize_acqf_torch → BoTorch optimize_acqf +
  gen_candidates_torch + torch.optim.Adam。Extended/Delegated

- **torch_adamw**
  数学的な探索方法：分離weight decayを伴うAdamW更新
  dispatch先と所有権：同上 + torch.optim.AdamW。weight decayをGP priorと解釈しない

- **torch_sgd**
  数学的な探索方法：勾配によるSGD更新
  dispatch先と所有権：同上 + torch.optim.SGD。収束保証を意味しない

- **random**
  数学的な探索方法：独立な一様乱数候補batchを多数評価
  dispatch先と所有権：backends/sampling.py、torch.randとargmaxでImplemented

- **sobol**
  数学的な探索方法：q×d次元のscrambled Sobol点による候補batch評価
  dispatch先と所有権：backends/sampling.py、SobolEngineとargmaxでImplemented

- **de**
  数学的な探索方法：Differential Evolutionによるpopulation探索
  dispatch先と所有権：backends/differential_evolution.py、optimize_acqf_de

- **cmaes**
  数学的な探索方法：共分散行列適応進化戦略
  dispatch先と所有権：backends/cmaes.py、optimize_acqf_cmaes

- **ga**
  数学的な探索方法：選択・交叉・変異を使う遺伝的探索
  dispatch先と所有権：通常はbackends/genetic_algorithm.py、
  mixed variable_spaceならbackends/mixed_genetic_algorithm.py

- **pso**
  数学的な探索方法：粒子群の位置・速度更新による探索
  dispatch先と所有権：backends/pso.py、optimize_acqf_pso

- **hybrid**
  数学的な探索方法：複数探索手順の組合せ
  dispatch先と所有権：backends/hybrid.py、optimize_acqf_hybrid。詳細なstageはそのbackend実装で確認

### samplingの選択式とqの意味

randomおよびsobolは $N$ 個の候補集合 $X^{(k)}\in\mathbb R^{q\times d}$ をサンプリングして

$$
k^\star=\operatorname*{arg\,max}_{1\le k\le N} A(X^{(k)}),
\qquad X^\star=X^{(k^\star)}
$$

を返す。つまり $q$ 点を個別に採点して上位を束ねるのではなく、**q点のjoint acquisition score**を比較する。
sample_candidate_batchesの出力shapeは[N,q,d]。select_best_sampled_batchは各batchにつき1スカラーを要求し、
そうでなければValueErrorを送出する。mixed入力ではrandomは合法な整数・カテゴリ集合から抽出し、sobolは連続一様座標を等幅inverse-CDF binへ写像する。
いずれもglobal acquisition最適性を証明しない。

### パラメータをどのbackendへ渡すか

- **num_restarts、raw_samples、options**
  botorch：BoTorchへ渡す
  torch_adam / torch_adamw / torch_sgd：BoTorch restart・初期化へ渡す
  random / sobol / DE / CMA-ES / GA / PSO / hybrid：BoTorch固有optionsとbatch_initial_conditionsは拒否。
  samplingはoptimizer_optionsのnum_samplesを用い、省略時はraw_samplesか512

- **optimizer_options**
  botorch：非空なら拒否
  torch_adam / torch_adamw / torch_sgd：非空なら拒否
  random / sobol / DE / CMA-ES / GA / PSO / hybrid：backend固有パラメータとして解釈

- **seed**
  botorch：dispatcherからの直接指定は受けない（native初期化を利用）
  torch_adam / torch_adamw / torch_sgd：直接指定を拒否
  random / sobol / DE / CMA-ES / GA / PSO / hybrid：samplingとevolutionary系へ渡す

- **sequential=True**
  botorch：native経路で可
  torch_adam / torch_adamw / torch_sgd：拒否
  random / sobol / DE / CMA-ES / GA / PSO / hybrid：joint q方式のため拒否。明示的なsequential APIを別途選ぶ

- **ic_generator / ic_gen_kwargs**
  botorch：BoTorch nativeへ渡す
  torch_adam / torch_adamw / torch_sgd：拒否
  random / sobol / DE / CMA-ES / GA / PSO / hybrid：拒否

- **fixed_features、candidate constraints、mixed variables**
  botorch：backend capabilitiesに依存
  torch_adam / torch_adamw / torch_sgd：fixed_featuresとcandidate constraintsを処理。制約はペナルティ探索＋最終可行性検査（既定strict）であり、mixed非対応
  random / sobol / DE / CMA-ES / GA / PSO / hybrid：各backendのcapabilityに依存。mixed時のGAは別backendへ分岐

上表はsrc/robotorchan/optim/dispatch.pyによる**dispatch契約**である。capabilitiesに対応があることと、
各backendで特定の等式・非線形制約の実行が成功することは同一ではない。
PyTorch backendは制約を受け付け、既定のstrictモードでは違反した最終候補をValueErrorで拒否する。penaltyモードでは違反候補が返る場合がある。
direct seed指定もtorch_*では拒否される。

### named dispatcherに含まれない経路

NSGA-II（多目的探索）、BoTorch native mixed、mixed one-shotは上記OptimizerNameの11候補に含まれず、個別のAPIで利用する。
特にone-shotのfantasy補助変数を通常のmixed最適化へそのまま渡さない。詳細は[Mixed/Discrete](04_mixed_discrete.md)、
[Batch/one-shot](08_batch_one_shot.md)、[Evolutionary](07_evolutionary.md)で確認する。

### 理論と実装の対応

数学的に適用可能な手法が、特定の実装ですべて利用可能とは限りません。実際の対応状況は [Optimization guides](../../optimization/README.md)
を参照してください。

#### High-dimensional search：実装上の補足

REMBO、HeSBO、ALEBO、TuRBO、BAxUSによる探索空間の構成は、「DEかBoTorchか」という数値backendの選択とは別の軸である。

#### Mixed-variable selection：実装上の補足

現在の主な経路は explicit BoTorch mixed、Mixed GA、および対応範囲内のDE / PSOです。

#### Search strategy と backend を分ける：実装上の補足

REMBO、BAxUS、TuRBOは search strategy 側、BoTorch、DE、GA、PSOなどは numerical backend 側です。

#### 関連ドキュメント：実装上の補足

- [Optimization guides](../../optimization/README.md)
- [制約処理](../../../docs-internal/contracts/optimization/constraint-handling-metadata.md) -
  [初期化の対応](../../../docs-internal/verification/optimization/initialization-compatibility.md)
- [Optimizer runtime contract](../../../docs-internal/contracts/optimization/runtime-contract.md)

## 関連する実装資料

variable semantics q semantics constraint semantics initialization semantics coordinate system
dtype / device API surface

- [Optimization guides](../../optimization/README.md)
- [初期化の対応](../../../docs-internal/verification/optimization/initialization-compatibility.md)
- [Optimizer runtime contract](../../../docs-internal/contracts/optimization/runtime-contract.md)

## metadataとdispatch分岐の対応

[capabilities.py](../../../src/robotorchan/optim/capabilities.py)は
制約の扱い、mixed変数、初期化などをbackendごとに宣言します。
[dispatch.py](../../../src/robotorchan/optim/dispatch.py)の
`_validate_requested_capabilities()`が要求と照合し、対応しない引数は`_reject_*` helperで明示的に拒否します。

この条件分岐はローカルの接続実装です。数値探索の更新則は各backend、BoTorch、SciPy、PyTorchなどの担当箇所で別に実現されます。
search strategyによる座標や状態の管理も、backend選択の内部には取り込みません。

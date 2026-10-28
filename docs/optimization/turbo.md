# TuRBO search strategy

`TuRBOStrategy` は surrogate model の内部表現ではなく、公開された元入力空間で trust region を管理します。

```python
center = train_X[train_Y.squeeze(-1).argmax()]
strategy = TuRBOStrategy(bounds, center=center)
result = strategy.optimize(acq_function)

new_Y = objective(result.candidates)
strategy.update_state(new_Y, candidates=result.candidates)
```

`center` は明示的に渡します。これにより `PCAGP` など surrogate 内部で入力を低次元表現へ変換するモデルでも、TuRBO は常に元入力空間の座標で動作します。

目的関数を評価する責務は strategy にはありません。候補を評価した後、観測値と候補点を `update_state` に渡すと、改善した候補が新しい incumbent になり、success / failure counter と trust-region length が更新されます。

初回観測では `best_value=-inf` を特別扱いし、有限な最初の観測を success として初期化します。


## Trust-region geometry

デフォルトでは各入力次元を等方的に扱います。trust-region length は各次元の global bounds の幅に対する相対値として解釈されるため、入力範囲が異なる場合でも同じ相対スケールで局所領域を構成します。

ARD lengthscale などから得た正の重みを `dimension_weights` に渡すと、各次元の幅を異方的にできます。重みは内部で幾何平均が1になるよう正規化されるため、全重みを同じ定数倍しても trust region は変化しません。

```python
strategy = TuRBOStrategy(
    bounds,
    center=center,
    dimension_weights=lengthscales,
)
```

geometry自体は `turbo_trust_region_bounds` としてモデル非依存です。モデル固有のlengthscale抽出はこの関数の責務に含めません。これにより通常のARD GP、SAAS、reduced-space modelなどで異なる「どのlengthscaleを使うか」という規則を、trust-region geometryから分離できます。

生成されたlocal boundsは常にglobal boundsでclipされます。現段階ではpublic input space上のbox geometryを定義しており、normalized `[0, 1]^d` 専用にはしていません。モデル側のinput transformを再適用しないため、二重normalizationも行いません。


## Baseline TuRBO-1 loop

Baseline TuRBO-1では、surrogate modelとacquisition functionの構築は呼び出し側が担当し、`TuRBOStrategy` は現在のtrust region内でcandidateを生成します。評価後の観測値を `update_state` に戻すことで、incumbentとtrust-region stateを次iterationへ引き継ぎます。

```python
for _ in range(n_iterations):
    model = SingleTaskGP(train_X, train_Y)
    acquisition = ExpectedImprovement(model, best_f=train_Y.max())
    result = strategy.optimize(acquisition, q=1)

    new_X = result.candidates
    new_Y = objective(new_X)
    strategy.update_state(new_Y, candidates=new_X)

    train_X = torch.cat([train_X, new_X])
    train_Y = torch.cat([train_Y, new_Y])
```

candidate optimizationはrobotorchanの `optimize_acqf` dispatchを経由します。既定値は `optimizer="botorch"` で、BoTorchのgradient-based acquisition optimizationを利用します。これによりTuRBO固有コードはlocal search regionの管理に集中し、optimizer実装を重複させません。

初期のbaseline実装はcontinuous single-objective TuRBO-1を対象としていました。現在はrestart、Thompson sampling、q-batch state、candidate constraints、async pending-point連携も追加されています。各機能の対応範囲と制限は以下の節を参照してください。


## Thompson candidate generation

TuRBOではacquisition optimizationとは別に、local candidate pool上のThompson samplingを利用できます。

```python
result = strategy.thompson_sample(
    model,
    q=1,
    n_candidates=2000,
)
```

candidate poolは現在のtrust region内にSobol点を生成し、各候補では既定で `min(20 / d, 1)` の確率で次元をperturbします。全次元が未変更になった候補には少なくとも1次元のperturbationを強制します。この高次元maskはTuRBOの標準的なThompson sampling構成に対応します。

posterior samplingとsample maximizerの選択は既存の `select_thompson_candidates` に委譲し、TuRBO側では再実装しません。`seed` を指定したstrategyではSobol poolとperturbation maskを再現可能に生成します。

この経路はtrust regionから有限candidate poolを構築する処理です。現在のmixed/discrete、candidate constraints、async pending pointsの対応範囲と制限は以下の専用節を参照してください。


## Batch TuRBO

Batch TuRBOでは `TuRBOState.batch_size` が1回の候補生成・評価で完了するbatch sizeを表します。候補生成時の `q` はこの値と一致する必要があります。

```python
state = TuRBOState(dim=d, batch_size=4, best_value=current_best)
strategy = TuRBOStrategy(bounds, center=incumbent, state=state)

result = strategy.optimize(acquisition, q=4)
new_Y = objective(result.candidates)
strategy.update_state(new_Y, candidates=result.candidates)
```

state transitionはcandidateごとではなく、完了したbatchごとに1回です。batch内の最大objective値を以前のbest valueと比較し、batch全体をsuccessまたはfailureとして1回だけcounterへ反映します。このため `failure_tolerance` のdimension/batch-size依存定義と実際の候補生成qが一致します。

Thompson経路でも同じbatch contractを使用します。`replacement=False` でlocal candidate poolからq点を選ぶため、同一batch内で同じpool rowを重複選択しません。

同期batchに加えて、非同期評価の完了順と`X_pending`の扱いも後述の専用節で説明します。pending pointsはstate更新に含めず、完了した観測のみで更新します。


## Restart strategy

trust-region lengthが `length_min` を下回ると `restart_triggered=True` になり、通常のcandidate generationは停止します。restartは暗黙には実行されず、明示的に `strategy.restart()` を呼びます。

```python
if strategy.state.restart_triggered:
    strategy.restart()
```

既定restartはglobal bounds上のscrambled Sobol点を新しいcenterとして選びます。`strategy.seed` またはrestart時の `seed` で再現可能です。明示的な `center` を指定して外部のglobal-search policyから再開することもできます。

restartではlocalな `length`、success/failure counter、restart flagをリセットします。一方、既観測データに対するglobal bestである `best_value`、`dim`、`batch_size`、tolerance設定は保持します。新しいrestart centerは未評価点なので、それ自体をbest incumbentとして扱いません。

`restart_count` はstateに保存され、benchmarkやresume時にrestart履歴を追跡できます。Phase 7の既定policyは単純なglobal Sobol re-entryです。model-based global restartや複数trust-regionの管理はこのbaseline restartとは区別します。


## High-dimensional model integration

TuRBOのtrust-region geometryと高次元surrogateは独立した責務です。モデルがpublic inputの各次元に対応する単一ARD lengthscale vectorを公開する場合、モデルからgeometry weightを更新できます。

```python
strategy.update_dimension_weights_from_model(model)
```

raw-space lengthscaleはまずglobal bounds幅で割ってbound-relative scaleへ変換し、その後geometric meanが1になるよう正規化されます。これにより非正規化boundsでもrangeを二重反映せず、ARDの相対的なgeometryだけをtrust regionへ反映します。

この自動連携は、`model.covar_module.lengthscale` の最終次元がpublic input dimensionと1対1に対応すると確認できるモデルを対象にします。SAASのMCMC sampleやensembleなどleading batch dimensionを持つ場合は、各public dimensionについてmedian lengthscaleを代表値に使います。Additive kernelのように単一lengthscale tensorを公開しない構造、Mixedモデルのencoded dimension、reduced/latent-space modelを暗黙に逆写像することはしません。

一方、TuRBOのcandidate generation自体はsurrogate-specificではありません。高次元の `EnsembleMapSaasSingleTaskGP` について、20次元public spaceからlocal Thompson candidateを生成しposterior samplingへ渡すruntime regressionを持ちます。ensemble posteriorのbatch dimensionは共通Thompson sampling層で処理し、TuRBOには常に `q x d` のpublic-space candidateを返します。Fully Bayesian SAASはNUTS fittingを伴うため、通常CIとは分離されたモデル検証を前提とします。

Reduced-spaceモデル固有のraw/latent trust-region semanticsはPhase 9で扱います。

## Reduced-space surrogate integration

TuRBOの既定search spaceは、surrogate内部のlatent spaceではなくpublic/original input spaceです。PCA、PLS、Random Projectionなどのfrozen input reducerを持つモデルでも、candidate、incumbent、bounds、trust-region stateはraw input dimensionで保持します。surrogateはposterior評価時にcandidateを内部latent representationへ変換します。

この分離により、inverse transformを持たないreducerでもTuRBOを利用できます。また、latent GPのARD lengthscaleをraw input dimensionのtrust-region weightとして暗黙に利用することはありません。`original_input_dim != reduced_input_dim` のモデルに対して `update_dimension_weights_from_model()` を呼ぶと明示的にエラーにします。raw-spaceでanisotropic geometryが必要な場合は、public input dimensionに対応する `dimension_weights` を明示してください。

latent-space TuRBOは、latent candidateを実行可能なraw candidateへ戻す写像と、global bounds / candidate constraintsを保存するfeasibility contractが必要です。現在のInputReducer共通契約はforward `transform()` のみであるため、PCAだけを特別扱いして疑似逆変換する実装は採用しません。decoder / inverse mappingの共通契約が整うまで、latent-space candidate optimizationはunsupportedです。

Mixed reduced modelsはcategorical passthroughを含むため、このcontinuous reduced-space方針とは分離し、Mixed / Discrete対応のPhase 13で扱います。jointly trained encoderのrepresentation driftもfrozen reducerとは異なるため、latent trust-regionを自動追従させません。

## Candidate constraints

TuRBOのtrust regionはcandidate/input-space constraintを置き換えません。acquisition optimizationでは、local trust-region boundsと `CandidateConstraints` の両方を同時に満たすcandidateを探索します。

連続最適化経路でのfeasible setは概念的に次の積集合です。

```text
global input bounds
    ∩
current TuRBO trust-region bounds
    ∩
linear / nonlinear CandidateConstraints
```

trust regionを作る際にnonlinear constraintをboundsへ近似変換するのではなく、global
boundsから切り出したlocal boundsをBoTorch optimizerへ渡し、そのlocal optimization内で
candidate constraintを同時に課します。このためnon-convexなnonlinear feasible setでも
trust-region geometryとは独立したまま扱えます。joint q最適化ではinter-point nonlinear
constraintも通常のBoTorch contract `g(X) >= 0` のまま利用できます。

```python
constraints = CandidateConstraints(
    inequality_constraints=((indices, coefficients, rhs),),
)
result = strategy.optimize(acquisition, constraints=constraints)
```

線形不等式・線形等式は既存のrobotorchan optimizer contractをそのまま利用します。BoTorch backendではBoTorchの `optimize_acqf` へ制約を渡します。非線形candidate constraintも同じ契約を利用し、BoTorch backendではfeasibleな `batch_initial_conditions` または対応するinitial-condition generatorが必要です。TuRBOが不正な初期点を暗黙生成して制約を回避することはありません。

Thompson samplingではlocal candidate poolを `CandidateConstraints` でfilterした後にposterior samplingします。feasible点が `q` 未満なら明示的に失敗し、unconstrained candidateへfallbackしません。現在この経路で保証するのは各candidateを独立に判定できるintra-point constraintです。q-batch全体を結ぶinter-point linear/nonlinear constraintは個別pool filteringでは数学的に保証できないためunsupportedです。

candidate constraintは入力座標に対する既知制約です。constrained EIなどのoutcome constraintは未知の出力・制約モデルをacquisition側で扱う別概念であり、TuRBOのtrust-region geometryへ変換しません。Mixed / discrete constraint semanticsの現行の対応範囲は、後述のMixed and discrete search spaces節を参照してください。

## Asynchronous evaluations and fantasization

TuRBO は独自の非同期 scheduler を追加しません。未完了評価は獲得関数の `X_pending` とし、`TuRBOState` は完了した目的値だけで更新します。

`TuRBOStrategy.optimize()` は設定済みの獲得関数をそのまま使います。pending の管理は上流の責務で、呼び出し前にコンストラクタや `set_X_pending()` で設定してください。incremental noisy improvement などは pending を固有の baseline 状態へ取り込むため、TuRBO が一律に snapshot・復元しません。

非同期 worker は batch_size=1 のまま、1 評価の完了で状態と surrogate を更新し、他の未完了点を `X_pending` に残して補充候補を生成できます。pending 自体は成功・失敗 counter を増やしません。

Thompson sampling では有限な局所プールから `X_pending` と一致する行を除外し、未完了点の再投入を防ぎます。fantasy-aware sampling を提供する意味ではありません。未完了評価への条件付けが必要ならモデルの `fantasize()` 契約で構築したモデルを `thompson_sample()` に渡します。

Fantasization therefore remains a model/acquisition responsibility:

- model `fantasize()` owns fantasy-model construction where the model advertises support;
- acquisition `X_pending` owns pending-point semantics for compatible acquisitions;
- TuRBO owns local trust-region geometry and completed-result state transitions.

fantasy batch 軸と TuRBO の q-batch 軸は異なり、`TuRBOState.batch_size` を変更しません。Kronecker モデルの既存の明示的な fantasization 制限も維持します。



## Noisy observations and robust objectives

TuRBO の trust-region state は、raw observation と state decision utility を分離できます。
`update_state(values, state_values=...)` の `values` は完了した生観測であり、
`observed_best_value` を更新します。`state_values` は同じ nominal candidate に対応する
posterior mean、denoised utility、または robust risk utility を渡すための任意引数です。
これを指定した場合、success / failure、`best_value`、incumbent center の選択は
`state_values` に基づきます。省略時は従来どおり `values` がそのまま使われます。

この分離により、単発の noisy observation の上振れをそのまま trust region の拡張や
center 移動として扱う必要がありません。TuRBO 自身は posterior mean や risk measure の
計算方法を規定せず、model / objective 側で計算済みの utility を受け取ります。
したがって qNEI / qLogNEI など acquisition 固有のノイズ処理を state logic に埋め込みません。

Input perturbation を使う robust BO でも trust region は nominal public input space にあります。
`InputPerturbation` が posterior 評価時に scenario axis を生成し、`Expectation`、CVaR などの
risk objective がその軸を集約します。scenario 数 `n_w` は TuRBO の入力次元、q、center、
trust-region bounds には追加されません。robust utility で state を更新する場合も、
`state_values` は nominal candidate ごとに1値、つまり `state.batch_size` 個を渡します。

```python
result = strategy.optimize(qlognei)
observed_y = evaluate(result.candidates)
robust_utility = posterior_risk_values  # one utility per completed nominal candidate
strategy.update_state(
    observed_y,
    candidates=result.candidates,
    state_values=robust_utility,
)
```

`best_value` は trust-region decision utility のbest、`observed_best_value` はraw observationの
bestです。この2つはノイズ・robust workflowでは一致しないことがあります。


## Mixed and discrete search spaces

Phase 13では、Mixed / discrete TuRBOを「カテゴリコードへ連続距離を導入する手法」には
しません。`MixedVariableSpace`でcontinuous、integer、categoricalの所有権を明示し、
trust regionのlengthによる局所化はcontinuous dimensionsだけに適用します。integerと
categorical dimensionsはglobalな合法値集合を維持します。

`TuRBOStrategy.optimize_mixed()`は、integerの全合法値とcategoricalの明示値集合の直積を
`fixed_features_list`として列挙し、既存のBoTorch `optimize_acqf_mixed` backendへ渡します。
各discrete assignmentの下でcontinuous coordinatesだけが現在のtrust region内で最適化されます。
このためcategory 0と1がcategory 1と2より近い、といった人工的な順序や距離を導入しません。

この実装はdiscrete assignment数が列挙可能な場合を対象にします。非常に大きな組合せ空間に
対しては全列挙の計算量が支配的になるため、Phase 13では専用の離散trust-region、Hamming
ball、COMBO系探索などを導入しません。その領域は別strategyとして設計すべきです。

```python
variable_space = MixedVariableSpace(
    bounds,
    integer_dims=(1,),
    categorical_values={2: [0.0, 1.0, 2.0]},
)
result = strategy.optimize_mixed(acquisition, variable_space)
```

centerとcandidateは常にpublic/raw input coordinatesです。state updateも通常の
`update_state(values, candidates=...)`を使い、改善したcandidateのdiscrete assignmentを含む
完全なraw candidateが次のcenterになります。ただし次回もdiscrete dimensions自体は数値的に
縮小せず、合法値集合全体を比較します。


## Multi-objective acquisition functions

Phase 14では、TuRBOをMORBOのようなmulti-objective専用trust-region algorithmへ変更しません。
robotorchanですでに利用できるBoTorch-nativeのqLogEHVI、qLogNEHVI、qLogNParEGOなどを、
通常のacquisition functionとして現在のsingle trust region内で最適化します。Pareto partitioning、
reference point、scalarizationはacquisition側の責務のままです。

この構成ではcandidate generationはmulti-objectiveでも既存のstrategy.optimize(acquisition)を
そのまま利用できます。qLogEHVIとqLogNEHVIについてlocal bounds内でcandidateを返すE2E regressionを
持ちます。multi-output modelやMCMultiOutputObjectiveの出力選択もTuRBO側へ複製しません。

一方、TuRBO-1のstate transitionには「改善したか」というscalar decisionが必要です。
Pareto outcome vectorをupdate_state()へ直接渡すと、どの点がbest centerかを一意に決められません。
そのためPhase 14では、valuesとstate_valuesはcandidateごとに1個のscalar utilityであることを
明示的に要求します。multi-objective workflowでは呼び出し側がhypervolume contribution、固定した
scalarization、または別の明示的なstate utilityを定義してからstateを更新します。

```python
result = strategy.optimize(qlogehvi)
new_Y = evaluate(result.candidates)  # [q, m]
state_utility = compute_state_utility(new_Y)  # [q]
strategy.update_state(
    state_utility,
    candidates=result.candidates,
)
```

重要なのは、acquisition utilityとstate utilityを同一概念として暗黙に扱わないことです。
qLogEHVIの値そのものを観測objectiveとしてstateへ保存する必要はありません。Pareto archiveの管理、
複数trust region、regionごとのlocal Pareto set、hypervolume-based region selectionなどを備えた
MORBO相当の手法は、single-region TuRBO-1とは異なるalgorithmです。Phase 14ではその名称や保証を
robotorchanのTuRBOStrategyへ持ち込みません。


## Multi-fidelity search

fidelity は構造化探索変数です。`optimize_multifidelity(...)` は incumbent 周辺の設計座標だけを縮め、全 fidelity 座標を大域 bounds に戻します。どの fidelity を観測するかは獲得関数が決めます。

`qMultiFidelityKnowledgeGradient` などと接続でき、目標 fidelity 射影とコスト効用は獲得関数が担います。TuRBO が実験候補を目標 fidelity に固定したり上流の多忠実度処理を重複したりしません。

幾何を直接使う `turbo_multifidelity_trust_region_bounds(...)` は正・負の fidelity index を受け取り、重複を拒否します。fidelity 以外の設計次元が少なくとも 1 個必要です。

### State semantics across fidelities

fidelity 間の生観測値は incumbent 効用として自動比較できません。低 fidelity の値が大きくても目標 fidelity で優れているとは限らないため、`update_state(..., state_values=...)` を使います。`values` は観測 scalar、`state_values` は成功・失敗と中心移動に使う呼び出し側の効用です。

たとえば目標 fidelity の posterior utility を `state_values` に渡すか、目標 fidelity の完了評価だけで中心を更新します。fidelity をまたぐ効用を自動推定しません。

```python
result = strategy.optimize_multifidelity(
    mfkg,
    fidelity_dims=[fidelity_dim],
)
new_y = evaluate(result.candidates)
state_utility = target_fidelity_utility(result.candidates, new_y)
strategy.update_state(
    new_y,
    state_values=state_utility,
    candidates=result.candidates,
)
```

単一 region の TuRBO と標準多忠実度獲得関数の組合せです。別の多忠実度 trust-region アルゴリズム、MF-MES、暗黙の fidelity scheduling を追加するものではありません。


## Strategy API contract

`TuRBOStrategy` は共通の `SearchStrategy` 契約に従います。`optimize(acq_function, q=...)` が返す `SearchResult` の候補は常に公開入力座標です。専用構造は単一の万能引数ではなく明示的なメソッドで表します。

- `optimize(...)`: continuous acquisition optimization inside the current trust region.
- `optimize_mixed(...)`: exact discrete assignments plus a local continuous region.
- `optimize_multifidelity(...)`: local design coordinates with global fidelity coordinates.
- `thompson_sample(...)`: posterior-sampling candidate generation from a local candidate pool.

4 つの候補生成経路は `trust_region_center`、`trust_region_bounds`、`trust_region_length`、`success_counter`、`failure_counter`、`restart_count`、`batch_size`、`candidate_generation`、`candidate_constraints` を共通 metadata とします。Mixed の離散次元、多忠実度の `fidelity_dims` などの診断は追加情報です。

`optimize*`・`thompson_sample` は状態を更新しません。完了観測を `update_state(...)` で明示適用し、必要な restart を `restart(...)` で解決します。非同期完了順や robust・多忠実度の効用を候補生成の副作用から分離します。


## Benchmark and validation

`benchmarks/turbo.py` は Ackley・Rosenbrock・Levy の Global BO と TuRBO を、既定 20・50・100 次元と複数 seed で比較します。関数・次元・seed ごとに同じ scrambled Sobol 初期化を使い、最良観測値、simple regret、trust-region length を記録します。

実測比較と回帰調査のための benchmark です。CI は目的契約と短い反復実行を確認し、固定少数 sample で TuRBO が必ず勝つことを要求しません。性能の主張には反復実験と不確実性の報告が必要です。


## Final support boundary

実装は単一 region の状態付き探索層です。連続最適化、posterior sampling、同期 q-batch、Thompson の明示的 pending 除外、候補制約、呼び出し側の robust・noisy 効用、厳密離散割当の Mixed、多忠実度の設計座標のみの trust region は専用テストで確認します。

MORBO 型の複数 region 多目的最適化、MF-MES、大規模離散空間の Hamming・graph region、worker の自動 scheduler、次元削減モデルの潜在 trust region は含意しません。別戦略が潜在幾何を所有しない限り、reduced surrogate でも元入力空間の region を公開します。

整数・カテゴリ・fidelity は大域領域を保つ構造化座標で、ARD 重みの正規化から除外します。通常の設計次元だけで、局所幅用の幾何平均正規化を計算します。

最小ループは `examples/turbo.py`、実測比較は `benchmarks/turbo.py` です。複数 seed の結果として解釈し、CI が強制する優越性とみなさないでください。
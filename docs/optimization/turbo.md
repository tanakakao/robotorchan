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

Phase 4のbaselineはcontinuous single-objective TuRBO-1を対象とします。restartの実行、Thompson sampling、q-batch固有state semantics、constraints、async/fantasizationは後続Phaseで追加します。


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

この経路はcontinuous trust regionから有限candidate poolを構築する処理です。mixed/discrete search space、constraints、async pending pointsはそれぞれ後続Phaseで扱います。


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

Phase 6では同期batchを対象とします。評価完了順が異なるasynchronous batch、`X_pending`、fantasizationはPhase 11で扱います。


## Restart strategy

trust-region lengthが `length_min` を下回ると `restart_triggered=True` になり、通常のcandidate generationは停止します。restartは暗黙には実行されず、明示的に `strategy.restart()` を呼びます。

```python
if strategy.state.restart_triggered:
    strategy.restart()
```

既定restartはglobal bounds上のscrambled Sobol点を新しいcenterとして選びます。`strategy.seed` またはrestart時の `seed` で再現可能です。明示的な `center` を指定して外部のglobal-search policyから再開することもできます。

restartではlocalな `length`、success/failure counter、restart flagをリセットします。一方、既観測データに対するglobal bestである `best_value`、`dim`、`batch_size`、tolerance設定は保持します。新しいrestart centerは未評価点なので、それ自体をbest incumbentとして扱いません。

`restart_count` はstateに保存され、benchmarkやresume時にrestart履歴を追跡できます。Phase 7の既定policyは単純なglobal Sobol re-entryです。model-based global restartや複数trust-regionの管理はこのbaseline restartとは区別します。

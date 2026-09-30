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

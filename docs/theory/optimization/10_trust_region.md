# Trust-region Acquisition Optimization

## 10.1 Local geometry

trust-region method は有望領域の周囲に局所探索領域を置き、
成功・失敗に応じて領域サイズを更新します。

## 10.2 TuRBO

TuRBO は BO に trust-region state を導入し、
global domain 全体ではなく局所領域内で candidate を生成します。

## 10.3 Stateful strategy

center、length、success / failure counter、restart condition は
surrogate parameter ではなく search state です。

詳細は [High-dimensional Search Strategies](../20_high_dimensional_search.md) と
[TuRBO implementation guide](../../optimization/turbo.md) を参照してください。

# BoundaryVariance

`BoundaryVariance` は robotorchan 固有の level-set 推定用ヒューリスティックです。BoTorch の標準獲得関数や文献上の標準アルゴリズムとして引用しないでください。

事後平均 `mu(x)`、分散 `v(x)`、目標水準 `t` に対して、スコアは次の通りです。

`v(x) * exp(-0.5 * ((mu(x) - t) / sqrt(v(x))) ** 2)`

分散の項は不確実な点を優先し、指数の項は不確実性に比べて平均が境界から遠い点を抑えます。

`Straddle` と異なり `beta` は持ちません。以前の未使用引数は削除され、互換引数としても受け取りません。

現在は q=1 に対応し、集約の意味が未定義の ensemble posterior は拒否します。

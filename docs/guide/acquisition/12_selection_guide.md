# Acquisition Selection Guide：理論から実装へ

## この章で解決する問い

手法名より最終decision purpose、noise、batch q、objective数、制約、fidelity、target distributionを先に確定する。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/12_selection_guide.md)が正本であり、
この章は**数式の計算対象 → クラス構造 →内部処理 → 近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

BoTorch-native改善/信頼上界/多目的/情報理論と、ローカルAL/レベルセット、semantic bridgeを分けて選択する。registryがあることのみで適用可能とは判定しない。

## 採用時の確認と注意事項

model posterior type・sampleability・q・output count・one-shot・candidate constraintsの全条件を検査。

### robotorchanのドキュメント層

手法を選んだ後は、目的に応じて参照先を分けます。

```text
docs/theory/acquisition/
    WHY
    理論・数式・関係性

docs/optimization/
    HOW
    API・使用例・現在の適用範囲

docs-internal/contracts/optimization/acquisition-integration.md
    STATUS
    現在の integration / limitation

src/robotorchan/acquisition/
    IMPLEMENTATION
```

Theory に記載されていることは、robotorchan がすべて独自実装していることを意味しません。

BoTorch native で十分な標準 acquisition は再実装せず、robotorchan-specific acquisition が必要な領域だけを補完します。

## 補足となる実装上の設計

#### Multi-objective BO：実装上の表現

```text
hypervolume improvement を直接評価
    → qLogEHVI

noise を考慮
    → qLogNEHVI

scalarization で Pareto front を探索
    → qLogNParEGO

将来の multi-objective decision value
    → HVKG（理論上の選択肢。現在の robotorchan runtime-validated integration 外）
```

#### Threshold boundary を学びたい：実装上の補足

qBoundaryVariance は robotorchan-specific heuristic であり、literature named method として扱いません。

#### Model capability と acquisition capability を混同しない：実装上の補足

robotorchan には Mixed、MultiTask、Multi-Fidelity、高次元、ロバスト、表現力の高い surrogate models があります。

#### Step 8: implementation / optimizer contract：実装上の補足

最後に現在のrobotorchan / BoTorch integration、optimizer compatibility、runtime validationを確認します。
Theoryで定義できることと、現在の実行可能workflowは同義ではありません。

#### この章で覚えておくこと：実装上の補足

- acquisition名ではなくdecision purposeから選ぶ
- BO、function learning、level-set learningを最初に分ける
- multi-output modelとmulti-objective decisionを混同しない
- noise、q-batch、pending、fantasyは別のaxis
- unknown output constraintとknown candidate constraintを分ける
- cost-awareとmulti-fidelityを分ける
- information-theoretic methodでは「何についてのinformationか」を確認する
- model capabilityとacquisition capabilityとoptimizer capabilityを別々に確認する
- posterior requirementをmarginal moments / joint Gaussian / posterior samplesで確認する
- registryはcompatibility surfaceであり、catalogでもruntime validationの代替でもない
- Theory coverageはrobotorchan-owned implementationやruntime supportを意味しない

#### まとめ：実装上の補足

この分離を保つことで、robotorchan の多様な surrogate models と BoTorch の acquisition ecosystemを組み合わせても、
問題設定の意味を失わずに設計できます。

## 関連する実装資料

現在のregistry名、対応するposterior、q-batch、mixed one-shot、ensemble、multi-fidelity、制約などの実装条件は
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
と各optimization guideを参照してください。理論上の適用可能性は、現在のruntime supportを保証しません。

## 数理モデルとクラス構造

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

```text
candidate 自身の uncertainty
    → qPosteriorVariance / qPosteriorStd

target region 全体の uncertainty
    → qNIPV / integrated variance reduction
```

qPosteriorVariance は単純で解釈しやすい一方、candidate を観測した結果ほかの場所の uncertainty がどう減るかは直接評価しません。

```text
established boundary criterion
    → qStraddle

randomized exploration coefficient
    → qRandomizedStraddle

variance × target proximity heuristic
    → qBoundaryVariance
```

MultiTaskGP を使っていることだけで multi-task experimental design が自動的に定義されるわけではありません。

| Problem setting | Acquisition family to inspect first | Main question |
| --- | --- | --- |
| Single-objective, low-noise BO | LogEI / UCB | improvement or confidence bound |
| Noisy BO | qLogNEI | latent improvement under noise |
| Parallel BO | q-acquisition | joint batch utility |
| Optimum information | MES / GIBBON | information about optimum |
| Terminal decision value | KG | value of information |
| Multi-step planning | qMultiStepLookahead | adaptive future decisions |
| Multi-objective | qLogEHVI / qLogNEHVI / qLogNParEGO | Pareto utility |
| Unknown constraints | constrained acquisition | feasible utility |
| Function learning | qPosteriorVariance / qNIPV | predictive uncertainty |
| Target-aware AL | EPIG | information about target predictions |
| Boundary learning | qStraddle family | uncertainty near threshold |
| Multi-fidelity | MF-KG + cost-aware utility | target-fidelity value per cost |

## 要求能力から互換性の理由を組み立てる

[registry.py](../../../src/robotorchan/acquisition/registry.py)は獲得関数の実装と
posterior requirement、q、出力数、fantasize、one-shotなどのmetadataを結び付けます。
[compatibility.py](../../../src/robotorchan/acquisition/compatibility.py)はモデル側の能力と比較し、
[optimizer_compatibility.py][opt-compat]は候補最適化側の能力を比較します。
数学的なdecision purposeを選ぶことと、実装可能な計算経路を選ぶことを分離しています。

この判定はrobotorchanの接続処理です。EIやKGの値を再計算したり、選択した獲得関数の実測性能をランキングしたりする処理ではありません。
標準手法の推定器へ到達する経路は各対応章に分けて説明しています。

[opt-compat]: ../../../src/robotorchan/acquisition/optimizer_compatibility.py

# Acquisition Selection Guide：理論から実装へ

## 1. この章で解決する問い

手法名より最終decision purpose、noise、batch q、objective数、制約、fidelity、target distributionを先に確定する。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/12_selection_guide.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

## 2. posteriorからcandidateへの処理の流れ

```text
train observations → surrogate.fit / posterior(X)
                                     ↓
                          objective / utility
                                     ↓
                  acquisition function α(X)
                                     ↓
                    candidate optimizer
                                     ↓
             selected X → costly evaluation
```

候補点で観測する未知の応答値と、観測前に計算する獲得関数値は異なる。獲得関数が最大化するutilityを、posterior meanの予測結果と読み替えない。analytic計算、MC sampling、fantasization、classification probability等、用いる数値表現は手法とモデルによって異なる。

## 3. 実装の対応と委譲

BoTorch-native改善/信頼上界/多目的/情報理論と、ローカルAL/レベルセット、semantic bridgeを分けて選択する。registryがあることのみで適用可能とは判定しない。

参照する代表的な公開APIは ``robotorchan.acquisition.get_acquisition_registry_entry``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

model posterior type・sampleability・q・output count・one-shot・candidate constraintsの全条件を検査。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../../.github/project-records/theory-guide/mapping/acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../optimization/README.md)、[理論Acquisition索引](../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。

## 実装補足：理論本文から分離した対応情報

### 12.31 robotorchanのドキュメント層

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

## 理論章から分離した実装上の条件

#### 12.10 Multi-objective BO：実装上の表現

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

#### 12.16 Threshold boundary を学びたい：実装上の補足


BoundaryVariance は robotorchan-specific heuristic であり、literature named method として扱いません。

#### 12.23 Model capability と acquisition capability を混同しない：実装上の補足

robotorchan には Mixed、MultiTask、Multi-Fidelity、高次元、ロバスト、表現力の高い surrogate models があります。

#### Step 8: implementation / optimizer contract：実装上の補足

最後に現在のrobotorchan / BoTorch integration、optimizer compatibility、runtime validationを確認します。
Theoryで定義できることと、現在の実行可能workflowは同義ではありません。

#### 12.33 この章で覚えておくこと：実装上の補足

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

#### 12.34 まとめ：実装上の補足

この分離を保つことで、robotorchan の多様な surrogate models と BoTorch の acquisition ecosystem
を組み合わせても、問題設定の意味を失わずに設計できます。

## 数値 API と利用上の補足


現在のregistry名、対応するposterior、q-batch、mixed one-shot、ensemble、
multi-fidelity、制約などの実装条件は
[Acquisition integration status](../../../docs-internal/contracts/optimization/acquisition-integration.md) と
各optimization guideを参照してください。理論上の適用可能性は、現在のruntime
supportを保証しません。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

```text
candidate 自身の uncertainty
    → PosteriorVariance / PosteriorStd

target region 全体の uncertainty
    → qNIPV / integrated variance reduction
```



PosteriorVariance は単純で解釈しやすい一方、candidate を観測した結果ほかの場所の uncertainty がどう減るかは直接評価しません。

```text
established boundary criterion
    → Straddle

randomized exploration coefficient
    → RandomizedStraddle

variance × target proximity heuristic
    → BoundaryVariance
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
| Function learning | PosteriorVariance / qNIPV | predictive uncertainty |
| Target-aware AL | EPIG | information about target predictions |
| Boundary learning | Straddle family | uncertainty near threshold |
| Multi-fidelity | MF-KG + cost-aware utility | target-fidelity value per cost |

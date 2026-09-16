# 高次元入力モデル選択ガイド

robotorchan の高次元入力モデルは、**次元削減の目的**、**データ数**、**BOでの実性能**で選ぶのが基本です。モデル名だけで優劣を決めず、同じデータ分割・初期設計・候補集合・seedで比較してください。

## まず試すモデル

| 状況 | 主な候補 | 特徴 |
| --- | --- | --- |
| 線形な低次元構造がありそう | `PCAGP` | Yを使わず安定。最初の基準モデルに向く |
| Yに関係する方向を抽出したい | `PLSGP` | 教師あり線形削減。小標本では成分数を小さくする |
| 非線形構造を圧縮したい | `AutoEncoderGP` / `VAEGP` | 表現学習はY非依存。事前学習後freeze |
| 非線形かつYに関連する表現が欲しい | `SupervisedAutoEncoderGP` / `SupervisedVAEGP` | Yを使って潜在表現を事前学習 |
| GP予測性能に合わせて表現を学習したい | `JointEncoderGP` / `JointVAEGP` | GP objectiveをencoderまで伝播 |
| 高次元だが少数の元変数だけが効く | SAAS / MAP-SAAS系 | 明示的な次元削減をせず関連次元を疎に選択 |

`RandomProjectionGP` は非常に安価な比較基準として有用です。複雑な表現学習モデルが Random Projection を明確に上回らない場合、表現学習コストに見合っているか再検討できます。

## 推奨比較順

小標本のBOでは、まず `SingleTaskGP`、`PCAGP`、`PLSGP`、`RandomProjectionGP` を比較します。その後、非線形性が必要なら AE/VAE 系を追加します。Joint 系は表現自由度が高い分、データが少ない場合の過学習と学習不安定性を確認してください。

モデル選択では、次の3段階で絞り込むと比較しやすくなります。

1. **予測品質**: RMSE / Gaussian NLL で明らかに不適切なモデルを除外する。
2. **計算コスト**: 学習時間、posterior時間、acquisition評価時間を確認する。
3. **BO性能**: 複数seedの simple regret / best observed trajectory で最終比較する。

予測RMSEが低いモデルが必ずしもBOで最も早く良い点を発見するとは限らないため、BO用途では3段階目を重視します。

## Phase 11 ベンチマーク構成

### 1. 予測性能と acquisition 評価

`benchmarks/high_dimensional_inputs.py` は、低次元active subspaceを埋め込んだ高次元synthetic regressionでモデルを比較します。

```bash
python benchmarks/high_dimensional_inputs.py \
  --n-train 64 \
  --n-test 128 \
  --input-dim 40 \
  --latent-dim 5 \
  --neural-epochs 50
```

主な評価値は以下です。

- RMSE: posterior mean の予測誤差
- Gaussian NLL: posterior uncertainty を含めた予測品質
- training time: reducer学習とGP fittingを含む時間
- posterior time: test setに対するposterior計算時間
- acquisition time: 共通候補集合に対する qLogEI 評価時間
- acquisition value: 候補集合内で得られた最大 acquisition value

結果はデフォルトで `benchmark_results/high_dimensional_inputs.csv` に保存されます。

### 2. SAAS 系

`benchmarks/high_dimensional_saas.py` では MAP-SAAS 系を比較できます。Fully Bayesian SAAS は MCMC と追加依存関係が必要で計算量も大きいため、明示的に有効化する設計です。

SAAS は「低次元へ写像する」モデルではなく、元の入力空間で少数の有効次元を疎に選択する考え方です。PCA/PLS/AE系とは異なる仮定を置くため、高次元で有効変数が少ない問題では比較対象に含める価値があります。

### 3. Sequential BO

`benchmarks/high_dimensional_bo.py` は、同一の初期設計と候補プールを使って sequential BO を実行します。候補選択には qLogEI を使い、各反復で surrogate を再学習します。

軽量モデルで複数seedを比較する例です。

```bash
python benchmarks/high_dimensional_bo.py \
  --n-initial 12 \
  --n-candidates 256 \
  --input-dim 40 \
  --latent-dim 5 \
  --n-iterations 10 \
  --mc-samples 64 \
  --seeds 0,1,2,3,4
```

AE/VAE、Joint系、MAP-SAASまで含める場合は `--include-extended` を追加します。Joint系は通常のGP fittingと学習目的が異なり、`JointEncoderGP` はGP MLL、`HybridAutoEncoderGP` は `hybrid_loss()`、`JointVAEGP` は `joint_loss()` を使います。

出力は2種類です。

- `benchmark_results/high_dimensional_bo.csv`: seed別の生trajectory
- `benchmark_results/high_dimensional_bo_summary.csv`: model × iteration の集計結果

集計CSVには simple regret の mean / population std / median / 25%点 / 75%点と、best observed の mean / std が保存されます。単一seedの結果だけでモデルを選ばず、平均性能とばらつきの両方を確認してください。

## ベンチマーク結果の読み方

### simple regret

simple regret が小さいほど、その時点までに良い候補を発見できています。最終iterationだけでなく、trajectory全体を確認すると「序盤に強いモデル」と「データ追加後に伸びるモデル」を区別できます。

### seed間ばらつき

平均simple regretが同程度なら、標準偏差や四分位範囲も確認します。特定seedだけ非常に良い結果になっていないか、初期設計への依存が大きくないかを判断できます。

### 計算時間とのトレードオフ

Joint / VAE / SAAS系がわずかなregret改善しか示さず学習時間が大幅に増える場合、実運用ではより単純なモデルが合理的なことがあります。一方、評価実験自体が非常に高価な材料・製造BOでは、surrogate学習時間よりサンプル効率を優先する判断もあります。

## 潜在次元の扱い

`latent_dim` / `n_components` は固定値だけで評価しないでください。例えば `2, 5, 10` のように複数候補を比較します。PCAの説明分散だけで決めると、目的変数に重要だが入力分散が小さい方向を落とす可能性があります。PLSやSupervised系ではYを利用するため、この挙動は異なります。

latent dimension 自体を大量にチューニングするとBO benchmarkへの過適合が起きるため、少数の事前候補を定め、同一seed集合で比較する運用が扱いやすいです。

## BOでの注意点

現在のReducedGP系は、GP内部の入力次元を削減しますが、獲得関数の最適化自体は元のD次元空間で行います。そのため、高次元化によるGP modeling負荷は軽減できても、`optimize_acqf` の探索難易度そのものは残ります。

Phase 11 の sequential benchmark は、この影響を surrogate 比較から分離するため、まず有限の共通候補プール上で acquisition を評価しています。したがって、このbenchmarkで良い結果でも、連続高次元空間での `optimize_acqf` が容易であることまでは意味しません。

非常に高いDでacquisition optimizationがボトルネックになる場合は、REMBO / BAxUS / TuRBOのような**探索戦略側の高次元対応**を別レイヤーとして検討します。これはPCA-GPやVAE-GPなどのsurrogate reductionとは別問題です。

## 実データでの推奨手順

最初から全モデルを本番BOで比較する必要はありません。まず同一train/test splitで予測benchmarkを行い、明らかに弱いモデルを除外します。残ったモデルについて複数seedのBO replayまたは候補プールbenchmarkを行い、simple regretと計算コストを比較します。

材料・製造データでは入力変数間の強い相関、制約、組成比、カテゴリ変数が存在することがあります。その場合、synthetic benchmarkの順位をそのまま実データへ適用せず、問題構造に合う変換・モデル・探索戦略を選んでください。

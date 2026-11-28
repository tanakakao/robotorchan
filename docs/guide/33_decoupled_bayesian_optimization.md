# 33. Decoupled Bayesian Optimization：理論と実装の対応

[Theory第33章](../theory/33_decoupled_bayesian_optimization.md) と並行して読む章です。
実行例は [Notebook 65](../../examples/notebooks/65_decoupled_bo_basics.ipynb) を参照してください。

## 1. 部分観測からモデルへ

出力ごとに異なる観測点を保持し、BoTorchの SingleTaskGP を ModelListGP で束ねます。
各モデルの学習入力は [n_j,d]、出力は [n_j,1] です。
n_j が異なっても、共通の予測入力に対する posterior.mean は [...,q,m] になります。
未測定値を埋める必要はありません。独立GPの尤度の和は SumMarginalLogLikelihood で扱い、
fit_gpytorch_mll に学習を委譲します。相関を持つモデルへの置換は獲得関数の対応条件を確認して行います。

## 2. 獲得関数の選択と内部処理

公開入口は create_acquisition です。
[Factory](../../src/robotorchan/acquisition/factory.py) が登録名からクラスを解決します。

| 獲得関数 | 価値の定義 | 処理 |
| --- | --- | --- |
| qDecoupledKnowledgeGradient | 有限意思決定集合の期待効用改善 | 独立GPのmasked fantasy |
| qDecoupledExpectedPredictiveInformationGain | 予測情報量 | Gaussian条件付き共分散 |
| qHypervolumeKnowledgeGradient | Pareto集合の期待HV改善 | BoTorchのone-shot |
| qMultiFidelityHypervolumeKnowledgeGradient | 目標忠実度での期待HV改善 | BoTorchのone-shot |
| qDecoupledParticleKnowledgeGradient | 異種出力の期待効用改善 | 粒子の尤度再重み付け |
| qDecoupledActiveLearning | 出力別ALスコアの集計 | マスクと重みで選択 |

KGでは decision_X が最終推奨点の集合、候補Xは次に測定する点です。
両者は同じである必要はありません。ListSampler は独立GPごとのfantasyを生成します。
weights は目的の価値を表し、制約出力の重みはゼロにします。
EPIGでは target_X が予測関心点、observation_noise は観測雑音の分散です。
標準偏差ではなく元の出力単位の分散を指定します。
HVKGの補助点は実際の測定候補ではありません。BoTorchのone-shot契約を利用します。

## 3. 候補とマスクを同時に選ぶ

[optimize_acqf_decoupled](../../src/robotorchan/optim/decoupled.py) は
許容マスク [k,m] と bounds [2,d] を受け取り、候補 [q,d]、選択マスク [q,m]、
ランキング値を返します。マスクはbool型で、各行に少なくとも一つのTrueが必要です。
出力順序はモデルと一致させます。

jointでは順序付きマスク割当を列挙し、各割当に対する連続候補の最適化を既存optimizerへ委譲します。
割当数は k^q なので上限を設けます。sequentialとheuristicは計算量を抑える近似です。
sequentialにはone-shotや候補制約などとの組み合わせに制限があります。
元の獲得関数のマスクは最適化終了時に復元されます。
MixedVariableSpaceは入力の離散値を定義し、評価マスクとは別の選択軸を管理します。

## 4. 評価費用と予算

output_costs は出力別加算費用、mask_costs は許容マスクごとの費用、
cost_function は入力依存の予測費用です。三者は排他的です。
非負の獲得関数値は費用で割り、負の近似値は費用を掛けてランキングします。
返却値は費用指定時にはランキング値であり、生の獲得関数値ではありません。
one-shotでnative cost-aware utilityを使う場合、二重の費用補正はできません。

[CostBudget](../../src/robotorchan/workflow/cost_budget.py) は残予算、予約、精算を管理します。
optimizerは予算で候補を絞りますが、予算を消費しません。
実験側で実測費用を記録するか、Studyで予約・精算します。
入力依存のhard budgetはBoTorch backendと実行可能な初期条件を必要とし、
他の候補制約やmixed離散列挙との組み合わせには制限があります。

## 5. Studyと非同期処理

[create_decoupled_proposer](../../src/robotorchan/study/decoupled.py) は
観測履歴を出力名ごとの(X,Y)へ変換し、利用者のmodel_factoryに渡します。
モデル学習はmodel_factoryの責務です。獲得関数の生成と候補・マスク選択を接続します。
部分tellでは観測済み出力だけが次の学習に入り、未完了マスクはpendingとして扱います。
pending対応はnative HVKG系とGaussian EPIGなどの明示的に対応した獲得関数に限られます。
未対応の獲得関数へpendingを無理に渡すことはできません。

## 6. 比較結果を読む

[ベンチマーク実行手順](../../benchmarks/README.md) では初期費用を含む共通予算で比較します。
真値で測る推奨Hypervolumeと、実際に学習へ追加する部分観測は区別します。
コスト格子で直前の結果を保持し、seed軌跡単位でBootstrapを行います。
信頼区間は点ごとの区間であり、性能優位性を自動的に保証しません。

## 7. 費用・Mixed・忠実度の実行例

[Notebook 66](../../examples/notebooks/66_decoupled_cost_mixed_multifidelity.ipynb) では、固定費用、工程共有費用、入力依存費用、joint q-batch、候補制約、Mixed変数、忠実度別費用を比較します。忠実度ごとのランキング値を比較し、選択されたマスクの実測費用だけを記録します。

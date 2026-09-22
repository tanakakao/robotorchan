# Capability-aware workflow

robotorchan は、モデル数を増やすだけでなく、問題要件と実装能力を明示的に対応付けるための capability-aware layer を提供します。

## 流れ

ProblemSpec → Model Registry → Selector → Model × Acquisition Compatibility → Recommendation → Capability Benchmark

ProblemSpec は問題の構造を宣言します。モデル名を直接指定する設定ではありません。

    from robotorchan.models.capabilities import InputType
    from robotorchan.problem import ProblemPurpose, ProblemSpec
    from robotorchan.recommendation import recommend_compatible_workflows

    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        input_type=InputType.MIXED,
        high_dimensional=True,
    )
    recommendations = recommend_compatible_workflows(spec)

## ProblemSpec の意味

- input_type: continuous / mixed
- task_type: single-task / multi-task
- output_type: single-output / multi-output
- objective_type: BO における single-objective / multi-objective
- multi_fidelity: fidelity 構造の有無
- structured_output: structured posterior を必要とするか
- high_dimensional: 高次元向け戦略を必要とするか
- robust: robustness strategy を必要とするか
- preference: pairwise preference observation を使うか

output_type, objective_type, task_type は別概念です。Active Learning の multi-output を multi-objective BO として扱いません。

## Selector

select_compatible_models() は registry metadata に基づく構造的なフィルタです。モデル名の文字列解析によって capability を推測しません。evaluate_models() を使うと、不適合モデルについて理由も取得できます。

## Recommendation

recommend_compatible_workflows() は互換候補を返しますが、モデル性能のランキングや「最良モデル」の判定は行いません。

Active Learning では robotorchan の acquisition registry と model-acquisition compatibility を組み合わせます。multi-output 問題では single-output 限定 acquisition を除外します。

Bayesian Optimization では現在、BoTorch 標準 acquisition を robotorchan 側で再実装・複製していないため、Recommendation は互換モデルまでを返します。

## Benchmark

robotorchan.benchmarks.capability は予測精度 benchmark ではなく、capability engine の整合性を検証するための構造 benchmark です。registry 全体に対して ProblemSpec のフィルタと Recommendation が一貫していることを確認します。

モデル自体の予測性能や BO performance benchmark とは目的を分離しています。

## Source of truth

モデル capability の source of truth は robotorchan.models.registry.MODEL_REGISTRY です。docs/model_coverage.json はそこから生成されます。Acquisition extension の metadata は robotorchan.acquisition.registry.ACQUISITION_REGISTRY が保持します。

新しいモデルや acquisition を追加するときは、実装だけでなく対応する capability metadata と contract test も更新してください。

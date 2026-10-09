# Sequential, batch and asynchronous benchmark comparison

`run_parallel_comparison(config, batch_size=3, max_concurrency=2)`
runs three random-search baselines using the same problem, seeds,
initial designs and candidate evaluation budget:

- **Sequential:** q=1, with each observation available before the next proposal.
- **Synchronous batch:** q=batch_size, with observations updated after each batch.
- **Asynchronous:** q=batch_size and bounded concurrency; completed observations
  are immediately available while unfinished proposals remain pending.

The asynchronous baseline uses fixed positive durations of one **simulated**
time unit per candidate. Its `simulated_makespan` is not measured wall time.
The `completion_order` property records submission indices in completion
order. The runner tracks pending candidates and records observed/truth
values separately.

This phase compares execution semantics, not optimizer quality. The
baseline is random search, and the three arms consume the same number
of candidate evaluations, not necessarily the same number of proposal
updates or the same simulated elapsed time. All arms use matching initial
designs. The async runner now recreates problem factories per seed to
prevent stateful observation RNG leakage across independent runs.

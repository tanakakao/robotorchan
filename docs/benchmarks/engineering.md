# Engineering-inspired benchmark problems

These deterministic design benchmarks exercise physical design tradeoffs
and outcome constraints using the standard `g(X) >= 0` convention.
They are **nondimensionalized illustrative engineering models**, not
validated high-fidelity structural or thermal simulations.

| Problem | Inputs | Minimized objective | Feasibility conditions |
| --- | --- | --- | --- |
| cantilever_beam | beam width b and height h, both 0.2–2 | cross-sectional area b*h | b*h^2 >= 0.2 and b*h^3 >= 0.1 |
| thermal_management | insulation i (0–2), cooling c (0–1) | i + 2*c^2 | i+c >= 1 and c <= 0.8 |

The cantilever constraints model normalized bending stress and
deflection limits. The thermal benchmark represents temperature
reduction versus cooling capacity and operating cost.

Both problems expose two continuous inputs, one minimization objective
and two deterministic outcome constraints. Known global optimum values
are intentionally omitted until analytically certified; feasible and
infeasible reference designs are tested.

Register with `register_engineering_problems(registry)` and run through
`run_benchmark` using any compatible candidate generator.

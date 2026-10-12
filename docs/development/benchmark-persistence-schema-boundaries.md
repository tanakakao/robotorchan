# Benchmark persistence schema boundaries

The benchmark persistence modules use distinct JSON contracts. Similar helper
names do not make the serialized artifacts interchangeable.

| Module | Public persistence contract | Schema |
| --- | --- | --- |
| `persistence.py` | Individual `BenchmarkTrajectory` records | Nested `tensors` with `dtype` and `data`; version 1/2 |
| `storage.py` | Configured experiment results | `config` and `trajectories`; tensor `dtype`, `shape`, `values`; version 1/2 |
| `trajectory_persistence.py` | `TrajectoryArtifact` | Configured experiment envelope with strict trajectory history; version 1 |
| `report_persistence.py` | `ReportArtifact` | Report, provenance, and schema version 1 |

## Cleanup decision

Do not consolidate these public formats or redirect their loaders to one
another. The version numbers are local to each format, not a global benchmark
schema version. In particular, the configured experiment formats may share
field names but impose different version and validation contracts.

Phase 9 adds cross-format rejection regression tests to guard this boundary.
Any future internal helper extraction must preserve serialized field names,
validation rules, dtype, tensor shapes, timing histories, and compatibility
with older supported versions. Validate with existing round-trip tests before
changing shared codecs. No public persistence API is removed in this phase.

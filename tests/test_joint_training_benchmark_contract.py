import importlib.util
import sys
from pathlib import Path

import torch

BENCHMARK_PATH = Path(__file__).parents[1] / "benchmarks" / "high_dimensional_bo.py"
SPEC = importlib.util.spec_from_file_location("joint_training_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_extended_joint_specs_share_training_policy():
    specs = BENCHMARK.model_specs(2, neural_epochs=1, include_extended=True)

    assert specs["JointEncoderGP"].fit_policy == "joint"
    assert specs["HybridAutoEncoderGP"].fit_policy == "joint"
    assert specs["JointVAEGP"].fit_policy == "joint"


def test_fit_model_calls_common_training_loss():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        def training_loss(self):
            self.calls += 1
            return self.weight.square()

    model = Model()
    BENCHMARK.fit_model(
        model,
        fit_policy="joint",
        joint_steps=2,
        joint_learning_rate=1e-2,
    )

    assert model.calls == 2

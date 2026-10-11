"""Isolate global PyTorch RNG used by benchmark strategy callbacks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from robotorchan.benchmarks.random_streams import derive_stream_seed


@contextmanager
def strategy_rng_scope(seed: int, batch_index: int, device: torch.device) -> Iterator[None]:
    """Seed a strategy batch while restoring external CPU/CUDA RNG states.

    This scope controls PyTorch global RNG, not NumPy, Python random,
    custom generators, or nondeterministic GPU kernels.
    """
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer.")
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    batch_seed = derive_stream_seed(seed, f"strategy_batch:{batch_index}")
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(batch_seed)
        yield

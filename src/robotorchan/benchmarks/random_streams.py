"""Independent, deterministic random streams for empirical benchmark runs."""

from __future__ import annotations

import hashlib

import torch


def derive_stream_seed(seed: int, stream: str) -> int:
    """Derive a stable PyTorch seed from a run seed and named stream.

    The mapping is independent of Python's randomized hash seed.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string.")
    payload = f"robotorchan:empirical:v1:{seed}:{stream}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63)


def make_stream_generator(seed: int, stream: str, device: torch.device) -> torch.Generator:
    """Create a fresh run-local generator without modifying global RNG state."""
    return torch.Generator(device=device).manual_seed(derive_stream_seed(seed, stream))

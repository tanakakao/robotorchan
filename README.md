# robotorchan

`robotorchan` is an extension library for [BoTorch](https://botorch.org/) focused on Bayesian optimization, active learning, and experimental design methods that are useful beyond BoTorch's core API.

## Design goals

- **BoTorch-native**: accept and return standard BoTorch / PyTorch objects whenever possible.
- **Thin extensions**: do not reimplement functionality already provided well by BoTorch.
- **Research-friendly**: make experimental acquisition functions and optimization utilities easy to test.
- **Composable**: keep models, acquisition functions, objectives, transforms, and optimizers separable.
- **Tested**: numerical behavior and tensor shapes are covered by automated tests.
- **Clean-room implementation**: robotorchan is implemented from scratch from public algorithms and public APIs. It does not copy source code from prior private/internal projects.

## Initial scope

Planned extension areas include:

- custom acquisition functions for Bayesian optimization and active learning;
- multi-objective, constrained, robust, and risk-aware extensions;
- classification / boundary-search acquisition functions;
- ordinal and preference-oriented optimization;
- lookahead and information-theoretic methods;
- utilities for mixed, discrete, and structured search spaces;
- optional surrogate-model extensions that remain compatible with BoTorch APIs.

## Requirements

- Python >= 3.11
- BoTorch >= 0.18.1

## Status

Early development. The public API is not yet stable.

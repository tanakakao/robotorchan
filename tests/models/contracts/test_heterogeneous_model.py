"""Tests for the heterogeneous model container baseline."""

import pytest
import torch
from torch import nn

from robotorchan.models import (
    HeterogeneousModel,
    KroneckerMultiTaskGP,
    ModelListGP,
    MultiTaskGP,
    SingleTaskGP,
)
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.classification.binary.non_gp.sklearn import RandomForestBinaryClassifier

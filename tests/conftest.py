from __future__ import annotations

import matplotlib
import pytest

from pendulum_lab.design import LinearModel, linear_model
from pendulum_lab.params import LabConfig

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def config() -> LabConfig:
    return LabConfig()


@pytest.fixture(scope="session")
def lin(config: LabConfig) -> LinearModel:
    return linear_model(config)

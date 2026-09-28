from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pendulum_lab.controllers import Controller, Observation, StateFeedback
from pendulum_lab.design import balance_strategies, lqr_design
from pendulum_lab.metrics import (
    control_energy,
    overshoot_percent,
    settling_time,
    step_metrics,
    success,
    undershoot_percent,
    wilson_interval,
)
from pendulum_lab.params import LabConfig, SensorConfig
from pendulum_lab.simulation import measure, pulse, simulate, step_reference
from pendulum_lab.types import FloatArray


class ConstantForce(Controller):
    name = "const"

    def __init__(self, value: float) -> None:
        self.value = value

    def reset(self, batch: int) -> None:
        self.batch = batch

    def compute(self, obs: Observation) -> FloatArray:
        return np.full(obs.x_hat.shape[0], self.value)


class Recorder(Controller):
    """Commands 1, 2, 3, ... and records the pending buffer it observes."""

    name = "rec"

    def reset(self, batch: int) -> None:
        self.pendings: list[FloatArray] = []

    def compute(self, obs: Observation) -> FloatArray:
        self.pendings.append(obs.pending.copy())
        return np.full(obs.x_hat.shape[0], float(obs.k + 1))


def test_saturation_is_enforced(config: LabConfig) -> None:
    res = simulate(ConstantForce(100.0), np.zeros(4), config, 0.2)
    assert np.all(np.abs(res.u) <= config.actuator.u_max + 1e-12)
    assert np.all(res.u_cmd == 100.0)


@pytest.mark.parametrize("d", [0, 1, 3])
def test_delay_line(config: LabConfig, d: int) -> None:
    cfg = replace(config.with_delay(d), force_disturbance_std=0.0)
    ctrl = Recorder()
    res = simulate(ctrl, np.zeros(4), cfg, 0.1)
    applied = res.u[:, 0]
    expected = np.array([0.0] * d + list(range(1, 11 - d)), dtype=float)
    np.testing.assert_allclose(applied, expected)
    if d > 0:
        np.testing.assert_allclose(ctrl.pendings[5][0], np.arange(5 - d + 1, 6, dtype=float))


def test_simulation_is_reproducible(config: LabConfig) -> None:
    s = balance_strategies(config)[3]
    a = simulate(
        s.make_controller(),
        np.array([0, 0, 0.1, 0]),
        config,
        1.0,
        estimator=s.make_estimator() if s.make_estimator else None,
        seed=42,
    )
    b = simulate(
        s.make_controller(),
        np.array([0, 0, 0.1, 0]),
        config,
        1.0,
        estimator=s.make_estimator() if s.make_estimator else None,
        seed=42,
    )
    np.testing.assert_array_equal(a.x, b.x)
    c = simulate(
        s.make_controller(),
        np.array([0, 0, 0.1, 0]),
        config,
        1.0,
        estimator=s.make_estimator() if s.make_estimator else None,
        seed=43,
    )
    assert not np.array_equal(a.x, c.x)


def test_batched_plants_simulate_independently(config: LabConfig) -> None:
    K = lqr_design(config).K
    p = config.params
    batch = replace(p, l=np.array([0.25, 0.3, 0.35]))
    cfg = replace(
        config, force_disturbance_std=0.0, sensor=SensorConfig(sigma_x=0.0, sigma_theta=0.0)
    )
    res = simulate(StateFeedback(K, "k"), np.array([0, 0, 0.1, 0]), cfg, 1.0, plant=batch)
    single = simulate(
        StateFeedback(K, "k"), np.array([0, 0, 0.1, 0]), cfg, 1.0, plant=replace(p, l=0.35)
    )
    np.testing.assert_allclose(res.x[:, 2], single.x[:, 0], atol=1e-12)


def test_batch_size_mismatch_raises(config: LabConfig) -> None:
    batch = replace(config.params, l=np.array([0.25, 0.3]))
    with pytest.raises(ValueError, match="batch"):
        simulate(ConstantForce(0.0), np.zeros((3, 4)), config, 0.1, plant=batch)


def test_measurement_noise_statistics() -> None:
    rng = np.random.default_rng(0)
    s = SensorConfig(sigma_x=0.01, sigma_theta=0.02)
    y = measure(np.zeros((20000, 4)), s, rng)
    np.testing.assert_allclose(y.std(axis=0), [0.01, 0.02], rtol=0.03)


def test_quantisation() -> None:
    rng = np.random.default_rng(0)
    s = SensorConfig(sigma_x=0.0, sigma_theta=0.0, quant_x=0.01, quant_theta=0.1)
    y = measure(np.array([[0.123, 0, 0.26, 0]]), s, rng)
    np.testing.assert_allclose(y, [[0.12, 0.3]])


def test_reference_and_pulse() -> None:
    r = step_reference(0.5, 1.0)
    assert r(0.99) == 0.0
    assert r(1.0) == 0.5
    p = pulse(3.0, 1.0, 0.1)
    assert p(0.99) == 0.0
    assert p(1.05) == 3.0
    assert p(1.1) == 0.0


def test_settling_time_of_second_order_response() -> None:
    zeta, wn = 0.5, 2.0
    t = np.linspace(0, 20, 20001)
    wd = wn * np.sqrt(1 - zeta**2)
    y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * t))
    ts = settling_time(t, y - 1.0, 0.02)[0]
    # classic 2 % estimate 4 / (zeta wn) = 4 s; exact value is a bit lower
    assert 3.0 < ts < 4.2
    os_expected = 100 * np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    assert overshoot_percent(y, 0.0, 1.0)[0] == pytest.approx(os_expected, rel=1e-3)
    assert undershoot_percent(y, 0.0, 1.0)[0] == 0.0


def test_settling_time_never_settles() -> None:
    t = np.arange(10.0)
    assert np.isnan(settling_time(t, np.ones(10), 0.5)[0])
    assert settling_time(t, np.zeros(10), 0.5)[0] == 0.0


def test_control_energy() -> None:
    assert control_energy(np.full((100, 1), 2.0), 0.01)[0] == pytest.approx(4.0)


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0)
    assert lo > 0.95
    assert np.isnan(wilson_interval(0, 0)[0])


def test_lqg_step_metrics_and_success(config: LabConfig) -> None:
    s = next(s for s in balance_strategies(config) if s.key == "lqg")
    res = simulate(
        s.make_controller(),
        np.zeros(4),
        config,
        8.0,
        estimator=s.make_estimator() if s.make_estimator else None,
        reference=step_reference(0.5, 0.5),
        seed=1,
    )
    m = step_metrics(res, 0.5, 0.5, u_max=config.actuator.u_max)
    assert m.settling_time[0] < 5.0
    assert m.overshoot[0] < 10.0
    assert m.undershoot[0] > 0.0  # non-minimum-phase: the cart first moves backwards
    assert success(res)[0]


def test_falling_pole_is_failure(config: LabConfig) -> None:
    res = simulate(ConstantForce(0.0), np.array([0, 0, 0.3, 0]), config, 2.0)
    assert not success(res)[0]

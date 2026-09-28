from __future__ import annotations

import numpy as np
import pytest

from pendulum_lab.analysis import augment_input_delay
from pendulum_lab.controllers import Observation
from pendulum_lab.design import (
    LinearModel,
    balance_strategies,
    kalman_filter,
    kalman_gain,
    lqr_delay_design,
    lqr_design,
)
from pendulum_lab.estimation import steady_state_kalman_gain
from pendulum_lab.lti import (
    closed_loop_matrix,
    compute_margins,
    gain_margin_interval,
    loop_transfer,
    observer_controller_lti,
    plant_lti,
    static_gain_lti,
)
from pendulum_lab.model import C_MEAS
from pendulum_lab.params import LabConfig


def test_lqg_separation_principle(config: LabConfig, lin: LinearModel) -> None:
    """Closed-loop poles = regulator poles U estimator poles U buffer poles."""
    d = config.actuator.delay_steps
    K = lqr_delay_design(config).K
    L = kalman_gain(config)
    plant = plant_lti(lin.Ad, lin.Bd, C_MEAS, d, config.Ts)
    ctrl = observer_controller_lti(lin.Ad, lin.Bd, C_MEAS, L, K, d, config.Ts)
    got = np.linalg.eigvals(closed_loop_matrix(plant, ctrl))
    Az, Bz = augment_input_delay(lin.Ad, lin.Bd, d)
    expected = np.concatenate(
        [
            np.linalg.eigvals(Az - Bz @ K),
            np.linalg.eigvals(lin.Ad - L @ C_MEAS @ lin.Ad),
            np.zeros(d),
        ]
    )
    np.testing.assert_allclose(np.sort(np.abs(got)), np.sort(np.abs(expected)), atol=1e-6)


def test_observer_lti_matches_filter_and_gain(config: LabConfig, lin: LinearModel) -> None:
    """The analytical LQG model reproduces the simulated KF + LQR at steady state."""
    d = config.actuator.delay_steps
    K = lqr_design(config).K
    L = kalman_gain(config)
    ctrl = observer_controller_lti(lin.Ad, lin.Bd, C_MEAS, L, K, d, config.Ts)
    kf = kalman_filter(config)
    _, P_prior = steady_state_kalman_gain(lin.Ad, C_MEAS, kf.Qd, kf.R)
    kf.initialize(np.zeros((1, 4)), P_prior)  # start at the steady-state covariance
    rng = np.random.default_rng(0)
    q = np.zeros(ctrl.n_states)
    pending: list[float] = [0.0] * d
    u_applied = 0.0
    for k in range(40):
        y = 1e-3 * rng.normal(size=2)
        if k > 0:
            kf.predict(np.array([u_applied]))
        xh = kf.update(y[None])
        u_sim = -(K @ xh[0]).item()
        u_lti = (ctrl.C @ q + ctrl.D @ y).item()
        assert u_sim == pytest.approx(u_lti, rel=1e-6, abs=1e-9)
        q = ctrl.A @ q + ctrl.B @ y
        pending.append(u_sim)
        u_applied = pending.pop(0)


def test_static_gain_margins_have_expected_structure(config: LabConfig, lin: LinearModel) -> None:
    K = lqr_design(config).K
    plant = plant_lti(lin.Ad, lin.Bd, np.eye(4), 0, config.Ts)
    m = compute_margins(plant, static_gain_lti(K, config.Ts))
    assert m.stable
    assert 0 < m.gain_low < 1 < m.gain_high
    assert m.phase_margin_deg > 30
    assert m.delay_margin_samples >= 1


def test_gain_margin_boundaries_are_exact(config: LabConfig, lin: LinearModel) -> None:
    K = lqr_design(config).K
    plant = plant_lti(lin.Ad, lin.Bd, np.eye(4), 1, config.Ts)
    ctrl = static_gain_lti(K, config.Ts)
    lo, hi = gain_margin_interval(plant, ctrl)

    def rho(k: float) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(plant, ctrl, k)))))

    assert rho(lo * 1.001) < 1 < rho(lo * 0.999)
    assert rho(hi * 0.999) < 1 < rho(hi * 1.001)


def test_delay_margin_consistent_with_phase_margin(config: LabConfig) -> None:
    for s in balance_strategies(config):
        plant, ctrl = s.make_lti()
        m = compute_margins(plant, ctrl)
        # the continuous-delay estimate brackets the integer-sample result
        samples = m.delay_margin_s / config.Ts
        assert m.delay_margin_samples <= samples + 1e-9
        assert samples < m.delay_margin_samples + 1 + 1e-9, s.key


def test_loop_transfer_unit_crossing_phase(config: LabConfig, lin: LinearModel) -> None:
    K = lqr_design(config).K
    plant = plant_lti(lin.Ad, lin.Bd, np.eye(4), 0, config.Ts)
    ctrl = static_gain_lti(K, config.Ts)
    m = compute_margins(plant, ctrl)
    L = loop_transfer(plant, ctrl, np.array([m.crossover_rad_s]))[0]
    assert abs(L) == pytest.approx(1.0, abs=1e-6)


def test_unstable_loop_reports_nan(config: LabConfig, lin: LinearModel) -> None:
    plant = plant_lti(lin.Ad, lin.Bd, np.eye(4), 0, config.Ts)
    m = compute_margins(plant, static_gain_lti(np.zeros((1, 4)), config.Ts))
    assert not m.stable
    assert np.isnan(m.phase_margin_deg)


def test_every_strategy_lti_is_stable(config: LabConfig) -> None:
    for s in balance_strategies(config):
        plant, ctrl = s.make_lti()
        rho = np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(plant, ctrl))))
        assert rho < 1.0, s.key


def test_observation_is_frozen() -> None:
    obs = Observation(np.zeros((1, 4)), np.zeros((1, 2)), np.zeros(1), np.zeros((1, 0)), 0, 0.0)
    with pytest.raises(AttributeError):
        obs.k = 1  # type: ignore[misc]

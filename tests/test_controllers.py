from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from pendulum_lab.analysis import augment_input_delay, continuous_to_discrete_poles
from pendulum_lab.controllers import (
    CascadePID,
    EnergySwingUp,
    Observation,
    PIDGains,
    StateFeedback,
    bryson_weights,
    dlqr,
    dlqr_delay_compensated,
    pole_placement_gain,
)
from pendulum_lab.design import (
    DesignChoices,
    LinearModel,
    lqr_design,
    pole_placement_design,
    swingup_controller,
)
from pendulum_lab.model import pendulum_energy
from pendulum_lab.params import LabConfig


def _obs(x_hat: np.ndarray, pending: np.ndarray | None = None) -> Observation:
    B = x_hat.shape[0]
    return Observation(
        x_hat=x_hat,
        y=x_hat[:, [0, 2]],
        ref=np.zeros(B),
        pending=np.zeros((B, 0)) if pending is None else pending,
        k=0,
        t=0.0,
    )


def test_bryson_weights() -> None:
    Q, R = bryson_weights(0.5, 1.0, 0.1, 2.0, 4.0)
    np.testing.assert_allclose(np.diag(Q), [4.0, 1.0, 100.0, 0.25])
    assert R[0, 0] == pytest.approx(1 / 16)


def test_dlqr_stabilises_and_satisfies_riccati(lin: LinearModel) -> None:
    Q, R = bryson_weights(*DesignChoices().lqr_limits)
    des = dlqr(lin.Ad, lin.Bd, Q, R)
    assert np.max(np.abs(des.closed_loop_poles)) < 1.0
    P = solve_discrete_are(lin.Ad, lin.Bd, Q, R)
    np.testing.assert_allclose(des.P, P)
    # Optimal cost-to-go decreases along closed-loop trajectories.
    Acl = lin.Ad - lin.Bd @ des.K
    x = np.array([0.1, 0.0, 0.1, 0.0])
    assert (Acl @ x) @ P @ (Acl @ x) < x @ P @ x


def test_delay_compensated_lqr_places_design_poles_plus_zeros(lin: LinearModel) -> None:
    Q, R = bryson_weights(*DesignChoices().lqr_limits)
    d = 2
    des = dlqr_delay_compensated(lin.Ad, lin.Bd, Q, R, d)
    base = dlqr(lin.Ad, lin.Bd, Q, R)
    Az, Bz = augment_input_delay(lin.Ad, lin.Bd, d)
    poles = np.linalg.eigvals(Az - Bz @ des.K)
    assert des.K.shape == (1, 4 + d)
    # Finite-spectrum assignment: the nominal LQR poles plus d poles at the origin.
    nonzero = np.sort_complex(poles[np.abs(poles) > 1e-6])
    np.testing.assert_allclose(nonzero, np.sort_complex(base.closed_loop_poles), atol=1e-6)


def test_pole_placement_places_requested_poles(lin: LinearModel) -> None:
    s_poles = [-2 + 1.5j, -2 - 1.5j, -5.5, -8.0]
    K, z = pole_placement_gain(lin.Ad, lin.Bd, s_poles, lin.Ts)
    got = np.sort_complex(np.linalg.eigvals(lin.Ad - lin.Bd @ K))
    np.testing.assert_allclose(got, np.sort_complex(z), atol=1e-8)
    np.testing.assert_allclose(z, continuous_to_discrete_poles(s_poles, lin.Ts))


def test_state_feedback_sign_and_wrapping(config: LabConfig) -> None:
    K = lqr_design(config).K
    ctrl = StateFeedback(K, "lqr")
    x = np.array([[0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.1 + 2 * np.pi, 0.0]])
    u = ctrl.compute(_obs(x))
    # Pole leaning forward -> push the cart forward; angle wrapping is transparent.
    assert u[0] > 0
    assert u[0] == pytest.approx(u[1])


def test_state_feedback_checks_delay_buffer(config: LabConfig) -> None:
    ctrl = StateFeedback(np.ones((1, 6)), "comp")
    with pytest.raises(ValueError, match="delay"):
        ctrl.compute(_obs(np.zeros((1, 4)), np.zeros((1, 1))))
    u = ctrl.compute(_obs(np.zeros((1, 4)), np.array([[1.0, 2.0]])))
    assert u[0] == pytest.approx(-3.0)


def test_pole_placement_design_is_stable(config: LabConfig, lin: LinearModel) -> None:
    K = pole_placement_design(config)
    assert np.max(np.abs(np.linalg.eigvals(lin.Ad - lin.Bd @ K))) < 1.0


def _pid(config: LabConfig) -> CascadePID:
    ch = DesignChoices()
    return CascadePID(ch.pid_outer, ch.pid_inner, config.Ts, config.actuator.u_max)


def test_pid_lti_matches_implementation(config: LabConfig) -> None:
    pid = _pid(config)
    lti = pid.to_lti()
    rng = np.random.default_rng(3)
    ys = 1e-3 * rng.normal(size=(50, 2))
    ys[0] = 0.0  # the implementation primes its derivative memory with the first sample
    pid.reset(1)
    q = np.zeros(lti.n_states)
    for k, y in enumerate(ys):
        obs = Observation(
            x_hat=np.zeros((1, 4)),
            y=y[None],
            ref=np.zeros(1),
            pending=np.zeros((1, 0)),
            k=k,
            t=k * config.Ts,
        )
        u = pid.compute(obs)[0]
        u_lti = (lti.C @ q + lti.D @ y).item()
        assert u == pytest.approx(u_lti, abs=1e-12)
        q = lti.A @ q + lti.B @ y


def test_pid_antiwindup_freezes_integrator(config: LabConfig) -> None:
    pid = CascadePID(
        PIDGains(0.1, 0.1, 0.1, 0.05), PIDGains(30.0, 10.0, 4.0, 0.02), config.Ts, u_max=1.0
    )
    pid.reset(1)
    obs = Observation(
        x_hat=np.zeros((1, 4)),
        y=np.array([[0.0, 0.5]]),
        ref=np.zeros(1),
        pending=np.zeros((1, 0)),
        k=0,
        t=0.0,
    )
    for _ in range(50):
        pid.compute(obs)
    assert pid.q[0, 3] == pytest.approx(0.0)  # inner integrator frozen at saturation


def test_pid_pushes_towards_leaning_side(config: LabConfig) -> None:
    pid = _pid(config)
    pid.reset(1)
    obs = Observation(
        x_hat=np.zeros((1, 4)),
        y=np.array([[0.0, 0.05]]),
        ref=np.zeros(1),
        pending=np.zeros((1, 0)),
        k=0,
        t=0.0,
    )
    assert pid.compute(obs)[0] > 0


def test_swingup_modes_and_energy_pumping(config: LabConfig) -> None:
    ctrl = swingup_controller(config)
    ctrl.reset(2)
    x = np.array([[0.0, 0.0, np.pi - 0.2, 1.0], [0.0, 0.0, 0.1, 0.0]])
    obs = Observation(
        x_hat=x,
        y=x[:, [0, 2]],
        ref=np.zeros(2),
        pending=np.zeros((2, config.actuator.delay_steps)),
        k=0,
        t=0.0,
    )
    ctrl.compute(obs)
    assert ctrl.mode.tolist() == [0.0, 1.0]
    # The pumping acceleration increases the energy of the hanging pole.
    a = ctrl.swing_acceleration(x[:1], np.zeros(1))
    ml = float(config.params.m * config.params.l)
    th, om = x[0, 2], x[0, 3]
    E_dot = -ml * np.cos(th) * om * (a[0] + 0.0)
    assert pendulum_energy(x[:1], config.params)[0] < 0
    assert E_dot > 0


def test_swingup_hysteresis(config: LabConfig) -> None:
    ctrl = swingup_controller(config)
    ctrl.reset(1)
    pend = np.zeros((1, config.actuator.delay_steps))

    def step(theta: float) -> float:
        x = np.array([[0.0, 0.0, theta, 0.0]])
        ctrl.compute(
            Observation(x_hat=x, y=x[:, [0, 2]], ref=np.zeros(1), pending=pend, k=0, t=0.0)
        )
        return float(ctrl.mode[0])

    assert step(0.2) == 1.0  # caught
    assert step(0.6) == 1.0  # inside hysteresis band: stay in balance mode
    assert step(1.2) == 0.0  # released


def test_swingup_rejects_invalid_hysteresis(config: LabConfig) -> None:
    ctrl = swingup_controller(config)
    with pytest.raises(ValueError, match="hysteresis"):
        EnergySwingUp(config.params, ctrl.balance, theta_catch=1.0, theta_release=0.5)

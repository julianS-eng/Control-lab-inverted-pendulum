from __future__ import annotations

import numpy as np
import pytest

from pendulum_lab.design import LinearModel, extended_kalman_filter, kalman_filter, kalman_gain
from pendulum_lab.estimation import (
    ExtendedKalmanFilter,
    disturbance_input_matrix,
    initial_covariance,
)
from pendulum_lab.model import C_MEAS, integrate, linearize
from pendulum_lab.params import LabConfig


def test_disturbance_matrix_force_column_equals_b(config: LabConfig) -> None:
    _, B = linearize(config.params)
    np.testing.assert_allclose(disturbance_input_matrix(config.params)[:, 0], B[:, 0])


def test_time_varying_filter_converges_to_steady_state_gain(config: LabConfig) -> None:
    kf = kalman_filter(config)
    kf.initialize(np.zeros((1, 4)), initial_covariance(config.sensor.R))
    for _ in range(500):
        kf.predict(np.zeros(1))
        kf.update(np.zeros((1, 2)))
    kf.predict(np.zeros(1))
    P = kf.P
    L = P @ C_MEAS.T @ np.linalg.inv(C_MEAS @ P @ C_MEAS.T + kf.R)
    np.testing.assert_allclose(L, kalman_gain(config), rtol=1e-6, atol=1e-9)


def test_observer_is_stable(config: LabConfig, lin: LinearModel) -> None:
    L = kalman_gain(config)
    assert np.max(np.abs(np.linalg.eigvals(lin.Ad - L @ C_MEAS @ lin.Ad))) < 1.0


def _open_loop_run(
    config: LabConfig, est: object, steps: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Nominal plant near upright with a stabilising LQR on the true state."""
    from pendulum_lab.design import lqr_design

    K = lqr_design(config).K[0]
    rng = np.random.default_rng(7)
    x = np.array([[0.05, 0.0, 0.08, 0.2]])
    est.initialize(  # type: ignore[attr-defined]
        np.array([[x[0, 0], 0.0, x[0, 2], 0.0]]), initial_covariance(config.sensor.R)
    )
    errs, truths = [], []
    u = np.zeros(1)
    for k in range(steps):
        y = x[:, [0, 2]] + rng.normal(size=(1, 2)) * [
            config.sensor.sigma_x,
            config.sensor.sigma_theta,
        ]
        if k > 0:
            est.predict(u)  # type: ignore[attr-defined]
        xh = est.update(y)  # type: ignore[attr-defined]
        errs.append(xh[0] - x[0])
        truths.append(x[0].copy())
        u = np.array([-K @ x[0]])
        x = integrate(x, u, config.params, config.Ts, config.substeps)
    return np.array(errs), np.array(truths)


@pytest.mark.parametrize("make", [kalman_filter, extended_kalman_filter])
def test_filters_beat_the_sensor_and_estimate_velocities(config: LabConfig, make: object) -> None:
    est = make(config)  # type: ignore[operator]
    errs, _ = _open_loop_run(config, est)
    rms = np.sqrt(np.mean(errs[100:] ** 2, axis=0))
    assert rms[0] < config.sensor.sigma_x
    assert rms[2] < config.sensor.sigma_theta
    assert rms[1] < 0.05  # m/s
    assert rms[3] < 0.15  # rad/s


def _swing_rate_rms(config: LabConfig, est: object) -> float:
    rng = np.random.default_rng(1)
    x = np.array([[0.0, 0.0, np.pi - 1.5, 0.0]])  # large oscillation about the hanging position
    est.initialize(  # type: ignore[attr-defined]
        np.array([[0.0, 0.0, x[0, 2], 0.0]]), initial_covariance(config.sensor.R)
    )
    errs = []
    for k in range(400):
        y = x[:, [0, 2]] + rng.normal(size=(1, 2)) * [1e-3, 2e-3]
        if k > 0:
            est.predict(np.zeros(1))  # type: ignore[attr-defined]
        errs.append(est.update(y)[0] - x[0])  # type: ignore[attr-defined]
        x = integrate(x, 0.0, config.params, config.Ts, config.substeps)
    e = np.array(errs)[100:]
    return float(np.sqrt(np.mean(e[:, 3] ** 2)))


def test_ekf_tracks_large_swings_where_linear_model_is_wrong(config: LabConfig) -> None:
    ekf_err = _swing_rate_rms(config, extended_kalman_filter(config))
    kf_err = _swing_rate_rms(config, kalman_filter(config))
    assert ekf_err < 0.15
    assert ekf_err < 0.5 * kf_err


def test_ekf_batched_covariance_shapes(config: LabConfig) -> None:
    ekf = extended_kalman_filter(config)
    ekf.initialize(np.zeros((3, 4)), np.eye(4))
    ekf.predict(np.zeros(3))
    xh = ekf.update(np.zeros((3, 2)))
    assert xh.shape == (3, 4)
    assert ekf.P.shape == (3, 4, 4)
    np.testing.assert_allclose(ekf.P, np.transpose(ekf.P, (0, 2, 1)), atol=1e-12)


def test_ekf_requires_nominal_params(config: LabConfig) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="nominal"):
        ExtendedKalmanFilter(replace(config.params, M=np.ones(2)), 0.01, np.eye(4), np.eye(2))

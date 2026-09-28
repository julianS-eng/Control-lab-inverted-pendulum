from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.signal import cont2discrete

from pendulum_lab.analysis import (
    augment_input_delay,
    controllability_report,
    finite_horizon_gramian,
    observability_report,
    spectral_radius,
    van_loan_process_noise,
)
from pendulum_lab.design import LinearModel
from pendulum_lab.model import C_MEAS


def test_zoh_matches_scipy(lin: LinearModel) -> None:
    Ad, Bd, *_ = cont2discrete((lin.A, lin.B, np.eye(4), np.zeros((4, 1))), lin.Ts, "zoh")
    np.testing.assert_allclose(lin.Ad, Ad, atol=1e-12)
    np.testing.assert_allclose(lin.Bd, Bd, atol=1e-12)
    np.testing.assert_allclose(lin.Ad, expm(lin.A * lin.Ts), atol=1e-12)


def test_cart_pole_is_controllable(lin: LinearModel) -> None:
    r = controllability_report(lin.A, lin.B, "u")
    assert r.full_rank
    assert r.weak_modes == ()
    assert controllability_report(lin.Ad, lin.Bd, "u").full_rank


def test_observability_depends_on_sensor_set(lin: LinearModel) -> None:
    both = observability_report(lin.A, C_MEAS, "x, theta")
    x_only = observability_report(lin.A, C_MEAS[:1], "x")
    th_only = observability_report(lin.A, C_MEAS[1:], "theta")
    assert both.full_rank
    assert x_only.full_rank
    assert th_only.rank == 3
    # The cart position (eigenvalue 0) is invisible to the angle sensor.
    assert len(th_only.weak_modes) == 1
    assert abs(th_only.weak_modes[0]) < 1e-9


def test_finite_gramian_is_positive_definite(lin: LinearModel) -> None:
    W = finite_horizon_gramian(lin.Ad, lin.Bd, 100)
    assert np.min(np.linalg.eigvalsh(W)) > 0


def test_van_loan_matches_quadrature(lin: LinearModel) -> None:
    G = np.eye(4)[:, [1, 3]]
    Qc = np.diag([2.0, 0.5])
    Ts = 0.05
    Qd = van_loan_process_noise(lin.A, G, Qc, Ts)
    s = np.linspace(0.0, Ts, 2001)
    vals = np.array([expm(lin.A * si) @ G @ Qc @ G.T @ expm(lin.A * si).T for si in s])
    ref = np.trapezoid(vals, s, axis=0)
    np.testing.assert_allclose(Qd, ref, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(Qd, Qd.T)


def test_van_loan_without_dynamics_is_exact() -> None:
    Qd = van_loan_process_noise(np.zeros((2, 2)), np.eye(2), np.diag([1.0, 3.0]), 0.1)
    np.testing.assert_allclose(Qd, np.diag([0.1, 0.3]), atol=1e-15)


@pytest.mark.parametrize("d", [0, 1, 3])
def test_delay_augmentation_reproduces_shifted_input(lin: LinearModel, d: int) -> None:
    Az, Bz = augment_input_delay(lin.Ad, lin.Bd, d)
    rng = np.random.default_rng(0)
    u = rng.normal(size=30)
    z = np.zeros(4 + d)
    x = np.zeros(4)
    for k in range(30):
        z = Az @ z + Bz[:, 0] * u[k]
        u_app = u[k - d] if k - d >= 0 else 0.0
        x = lin.Ad @ x + lin.Bd[:, 0] * u_app
        np.testing.assert_allclose(z[:4], x, atol=1e-12)


def test_delay_augmentation_rejects_negative(lin: LinearModel) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        augment_input_delay(lin.Ad, lin.Bd, -1)


def test_open_loop_discrete_is_unstable(lin: LinearModel) -> None:
    assert spectral_radius(lin.Ad) > 1.0

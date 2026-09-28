from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pendulum_lab.model import (
    cart_force_for_acceleration,
    dynamics,
    integrate,
    linearize,
    numerical_jacobians,
    pendulum_energy,
    total_energy,
    wrap_angle,
)
from pendulum_lab.params import CartPoleParams


@pytest.fixture
def p() -> CartPoleParams:
    return CartPoleParams()


@pytest.mark.parametrize("theta", [0.0, np.pi])
def test_equilibria_have_zero_derivative(p: CartPoleParams, theta: float) -> None:
    s = np.array([0.3, 0.0, theta, 0.0])
    np.testing.assert_allclose(dynamics(s, 0.0, p), 0.0, atol=1e-12)


@pytest.mark.parametrize("upright", [True, False])
def test_analytic_linearization_matches_finite_differences(
    p: CartPoleParams, upright: bool
) -> None:
    A, B = linearize(p, upright=upright)
    s0 = np.array([0.0, 0.0, 0.0 if upright else np.pi, 0.0])
    An, Bn = numerical_jacobians(p, s0)
    np.testing.assert_allclose(A, An, atol=1e-7)
    np.testing.assert_allclose(B, Bn, atol=1e-9)


def test_linearization_rejects_batched_params(p: CartPoleParams) -> None:
    batched = replace(p, M=np.array([0.5, 0.6]))
    with pytest.raises(ValueError, match="scalar"):
        linearize(batched)


def test_upright_is_unstable_and_hanging_is_stable(p: CartPoleParams) -> None:
    A_up, _ = linearize(p, upright=True)
    A_dn, _ = linearize(p, upright=False)
    assert np.max(np.linalg.eigvals(A_up).real) > 1.0
    assert np.max(np.linalg.eigvals(A_dn).real) <= 1e-12


def test_pushing_cart_tilts_pole_backwards(p: CartPoleParams) -> None:
    # A positive force accelerates the cart forward and the pole backwards.
    d = dynamics(np.zeros(4), 1.0, p)
    assert d[1] > 0.0
    assert d[3] < 0.0


def test_frictionless_energy_is_conserved(p: CartPoleParams) -> None:
    p0 = replace(p, b_c=0.0, b_p=0.0)
    s = np.array([0.0, 0.5, 2.0, -1.0])
    E0 = total_energy(s, p0)
    s = integrate(s, 0.0, p0, 5.0, 2000)
    assert abs(float(total_energy(s, p0) - E0)) < 1e-8


def test_friction_dissipates_energy(p: CartPoleParams) -> None:
    s0 = np.array([0.0, 0.5, 2.0, -1.0])
    s1 = integrate(s0, 0.0, p, 2.0, 400)
    assert total_energy(s1, p) < total_energy(s0, p)


def test_batched_dynamics_match_individual_evaluation(p: CartPoleParams) -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(5, 4))
    u = rng.normal(size=5)
    batched = replace(p, M=np.linspace(0.4, 0.6, 5), l=np.linspace(0.2, 0.4, 5))
    out = dynamics(states, u, batched)
    for i in range(5):
        pi = replace(p, M=float(batched.M[i]), l=float(batched.l[i]))  # type: ignore[index]
        np.testing.assert_allclose(out[i], dynamics(states[i], u[i], pi), rtol=1e-12)


def test_partial_feedback_linearization_achieves_requested_acceleration(
    p: CartPoleParams,
) -> None:
    rng = np.random.default_rng(1)
    s = rng.normal(size=(20, 4))
    a = rng.normal(size=20)
    u = cart_force_for_acceleration(s, a, p)
    np.testing.assert_allclose(dynamics(s, u, p)[:, 1], a, atol=1e-10)


def test_pendulum_energy_reference_points(p: CartPoleParams) -> None:
    mgl = float(p.m * p.g * p.l)
    assert pendulum_energy(np.zeros(4), p) == pytest.approx(0.0)
    assert pendulum_energy(np.array([0, 0, np.pi, 0.0]), p) == pytest.approx(-2 * mgl)


def test_wrap_angle_range() -> None:
    th = np.linspace(-20, 20, 1001)
    w = wrap_angle(th)
    assert np.all(w >= -np.pi)
    assert np.all(w < np.pi)
    np.testing.assert_allclose(np.sin(w), np.sin(th), atol=1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(th), atol=1e-12)


def test_uniform_rod_inertia() -> None:
    p = CartPoleParams.uniform_rod(M=1.0, m=0.3, l=0.25, b_c=0.0, b_p=0.0)
    assert pytest.approx(0.3 * 0.5**2 / 12) == p.J
    assert p.is_nominal()

r"""Nonlinear cart-pole model derived with Lagrange's equations, and its linearisation.

Conventions
-----------
State ``s = [x, x_dot, theta, theta_dot]`` where ``x`` is the cart position and
``theta`` is the pole angle measured **from the upright vertical**, positive when
the pole tip moves towards ``+x``. ``theta = 0`` is the (unstable) upright
equilibrium and ``theta = pi`` the (stable) hanging one. The input ``u`` is the
horizontal force on the cart.

Lagrangian
----------
With pole centre of mass at ``(x + l sin(theta), l cos(theta))``:

.. math::

    T &= \tfrac12 (M+m)\dot x^2 + m l \cos\theta\,\dot x\dot\theta
         + \tfrac12 (J + m l^2)\dot\theta^2, \\
    V &= m g l \cos\theta .

Applying :math:`\frac{d}{dt}\partial_{\dot q}L - \partial_q L = Q` with the
generalised forces ``Q_x = u - b_c x_dot`` and ``Q_theta = -b_p theta_dot``:

.. math::

    (M+m)\ddot x + m l\cos\theta\,\ddot\theta - m l \sin\theta\,\dot\theta^2
        &= u - b_c \dot x, \\
    m l \cos\theta\,\ddot x + (J + m l^2)\ddot\theta - m g l \sin\theta
        &= -b_p \dot\theta .

The mass matrix is always invertible because
``(M+m)(J+ml^2) - (m l cos theta)^2 >= M m l^2 + (M+m) J > 0``.
"""

from __future__ import annotations

import numpy as np

from pendulum_lab.params import CartPoleParams
from pendulum_lab.types import FloatArray

#: Number of states.
NX = 4
#: Measured outputs: cart position and pole angle.
C_MEAS: FloatArray = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def dynamics(state: FloatArray, u: FloatArray | float, p: CartPoleParams) -> FloatArray:
    """Evaluate the continuous-time nonlinear dynamics ``s_dot = f(s, u)``.

    Args:
        state: State array of shape ``(..., 4)``.
        u: Cart force, broadcastable to ``state.shape[:-1]`` [N].
        p: Physical parameters (scalars or arrays broadcastable to the batch).

    Returns:
        State derivative with the same shape as ``state``.
    """
    vel = state[..., 1]
    th = state[..., 2]
    om = state[..., 3]
    s = np.sin(th)
    c = np.cos(th)
    mt = p.M + p.m
    ml = p.m * p.l
    inertia = p.inertia_pivot
    f1 = u - p.b_c * vel + ml * s * om**2
    f2 = ml * p.g * s - p.b_p * om
    det = mt * inertia - (ml * c) ** 2
    xdd = (inertia * f1 - ml * c * f2) / det
    thdd = (mt * f2 - ml * c * f1) / det
    return np.stack(np.broadcast_arrays(vel, xdd, om, thdd), axis=-1)


def cart_force_for_acceleration(
    state: FloatArray, accel: FloatArray | float, p: CartPoleParams
) -> FloatArray:
    """Invert the model: force that produces a desired cart acceleration.

    This is collocated partial feedback linearisation (Spong, 1995). Given
    ``x_ddot = a`` the pole equation fixes ``theta_ddot``, and the cart equation
    returns the force.

    Args:
        state: State array of shape ``(..., 4)``.
        accel: Desired cart acceleration [m/s^2].
        p: Model parameters (normally the nominal ones).

    Returns:
        Force [N] with shape ``state.shape[:-1]``.
    """
    vel = state[..., 1]
    th = state[..., 2]
    om = state[..., 3]
    s = np.sin(th)
    c = np.cos(th)
    ml = p.m * p.l
    thdd = (ml * p.g * s - p.b_p * om - ml * c * accel) / p.inertia_pivot
    force = (p.M + p.m) * accel + ml * c * thdd - ml * s * om**2 + p.b_c * vel
    return np.asarray(force, dtype=float)


def rk4_step(state: FloatArray, u: FloatArray | float, p: CartPoleParams, dt: float) -> FloatArray:
    """Advance the nonlinear model one explicit Runge-Kutta-4 step with constant input.

    Args:
        state: State array of shape ``(..., 4)``.
        u: Force held constant over the step (zero-order hold).
        p: Physical parameters.
        dt: Step length [s].

    Returns:
        State after ``dt``.
    """
    k1 = dynamics(state, u, p)
    k2 = dynamics(state + 0.5 * dt * k1, u, p)
    k3 = dynamics(state + 0.5 * dt * k2, u, p)
    k4 = dynamics(state + dt * k3, u, p)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    state: FloatArray, u: FloatArray | float, p: CartPoleParams, Ts: float, substeps: int
) -> FloatArray:
    """Integrate over one sampling period using ``substeps`` RK4 steps (ZOH input).

    Args:
        state: State array of shape ``(..., 4)``.
        u: Force held over the period.
        p: Physical parameters.
        Ts: Sampling period [s].
        substeps: Number of RK4 steps per period.

    Returns:
        State at the end of the period.
    """
    dt = Ts / substeps
    for _ in range(substeps):
        state = rk4_step(state, u, p, dt)
    return state


def total_energy(state: FloatArray, p: CartPoleParams) -> FloatArray:
    """Total mechanical energy of cart and pole (zero potential at pivot height) [J]."""
    vel = state[..., 1]
    th = state[..., 2]
    om = state[..., 3]
    ml = p.m * p.l
    kinetic = (
        0.5 * (p.M + p.m) * vel**2 + ml * np.cos(th) * vel * om + 0.5 * p.inertia_pivot * om**2
    )
    return np.asarray(kinetic + ml * p.g * np.cos(th), dtype=float)


def pendulum_energy(state: FloatArray, p: CartPoleParams) -> FloatArray:
    """Pole energy used by the swing-up law, zero at the upright equilibrium [J].

    ``E = 1/2 (J + m l^2) theta_dot^2 + m g l (cos(theta) - 1)``; the hanging
    rest position has ``E = -2 m g l``.
    """
    th = state[..., 2]
    om = state[..., 3]
    return np.asarray(
        0.5 * p.inertia_pivot * om**2 + p.m * p.g * p.l * (np.cos(th) - 1.0), dtype=float
    )


def wrap_angle(theta: FloatArray | float) -> FloatArray:
    """Wrap an angle to the interval ``[-pi, pi)``."""
    return np.asarray((np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi, dtype=float)


def linearize(p: CartPoleParams, upright: bool = True) -> tuple[FloatArray, FloatArray]:
    r"""Analytic Jacobians of the dynamics at an equilibrium.

    Around ``theta = 0`` (``upright=True``) with ``I = J + m l^2`` and
    ``D = (M+m) I - (m l)^2``:

    .. math::

        A = \begin{bmatrix}
            0 & 1 & 0 & 0 \\
            0 & -I b_c / D & -(m l)^2 g / D & m l b_p / D \\
            0 & 0 & 0 & 1 \\
            0 & m l b_c / D & (M+m) m g l / D & -(M+m) b_p / D
        \end{bmatrix},\quad
        B = \begin{bmatrix} 0 \\ I/D \\ 0 \\ -m l / D \end{bmatrix}.

    The hanging equilibrium (``upright=False``) flips the sign of ``ml`` in the
    coupling terms (``cos(pi) = -1``).

    Args:
        p: Nominal (scalar) parameters.
        upright: Linearise about ``theta = 0`` if ``True``, else ``theta = pi``.

    Returns:
        Tuple ``(A, B)`` with shapes ``(4, 4)`` and ``(4, 1)``.
    """
    if not p.is_nominal():
        raise ValueError("linearize() requires scalar (nominal) parameters")
    M, m, l, g = float(p.M), float(p.m), float(p.l), float(p.g)
    b_c, b_p = float(p.b_c), float(p.b_p)
    inertia = float(p.inertia_pivot)
    sgn = 1.0 if upright else -1.0
    ml = sgn * m * l
    D = (M + m) * inertia - (m * l) ** 2
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -inertia * b_c / D, -ml * ml * g / D, ml * b_p / D],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, ml * b_c / D, (M + m) * ml * g / D, -(M + m) * b_p / D],
        ]
    )
    B = np.array([[0.0], [inertia / D], [0.0], [-ml / D]])
    return A, B


def numerical_jacobians(
    p: CartPoleParams, state: FloatArray, u: float = 0.0, eps: float = 1e-6
) -> tuple[FloatArray, FloatArray]:
    """Central-difference Jacobians of the nonlinear dynamics at an arbitrary point.

    Used to verify :func:`linearize` and by the extended Kalman filter.

    Args:
        p: Scalar parameters.
        state: Linearisation state, shape ``(4,)``.
        u: Linearisation input [N].
        eps: Finite-difference step.

    Returns:
        Tuple ``(A, B)`` with shapes ``(4, 4)`` and ``(4, 1)``.
    """
    state = np.asarray(state, dtype=float)
    A = np.zeros((NX, NX))
    for i in range(NX):
        d = np.zeros(NX)
        d[i] = eps
        A[:, i] = (dynamics(state + d, u, p) - dynamics(state - d, u, p)) / (2 * eps)
    B = ((dynamics(state, u + eps, p) - dynamics(state, u - eps, p)) / (2 * eps)).reshape(NX, 1)
    return A, B

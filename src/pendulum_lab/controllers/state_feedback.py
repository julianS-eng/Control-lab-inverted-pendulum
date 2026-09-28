r"""Linear state-feedback designs: discrete LQR (optionally delay-compensated) and pole placement.

All designs are *direct discrete* designs on the ZOH model ``(Ad, Bd)`` rather
than continuous designs emulated at the sampling rate; see
:func:`pendulum_lab.experiments.sampling_study` for why this matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import place_poles

from pendulum_lab.analysis import augment_input_delay, continuous_to_discrete_poles
from pendulum_lab.controllers.base import Controller, Observation, state_error
from pendulum_lab.types import ComplexArray, FloatArray


@dataclass(frozen=True)
class LQRDesign:
    """Result of a discrete LQR design.

    Attributes:
        K: State-feedback gain ``(1, n)`` for the law ``u = -K z``.
        P: Stabilising solution of the discrete algebraic Riccati equation.
        closed_loop_poles: Eigenvalues of ``Ad - Bd K``.
        Q: State weight used.
        R: Input weight used.
    """

    K: FloatArray
    P: FloatArray
    closed_loop_poles: ComplexArray
    Q: FloatArray
    R: FloatArray


def bryson_weights(
    x_max: float, v_max: float, theta_max: float, omega_max: float, u_max: float
) -> tuple[FloatArray, FloatArray]:
    """Bryson's rule: weight each variable by the inverse square of its tolerable excursion.

    ``Q = diag(1/x_max^2, 1/v_max^2, 1/theta_max^2, 1/omega_max^2)`` and
    ``R = 1/u_max^2``. This normalises the cost so that every term contributes
    about one unit when its variable reaches the acceptable limit, which turns
    the choice of ``Q`` and ``R`` into a choice of physically meaningful limits.

    Args:
        x_max: Acceptable cart-position deviation [m].
        v_max: Acceptable cart speed [m/s].
        theta_max: Acceptable pole-angle deviation [rad].
        omega_max: Acceptable pole angular rate [rad/s].
        u_max: Acceptable force [N].

    Returns:
        ``(Q, R)``.
    """
    Q = np.diag([1 / x_max**2, 1 / v_max**2, 1 / theta_max**2, 1 / omega_max**2])
    R = np.array([[1 / u_max**2]])
    return Q, R


def dlqr(Ad: FloatArray, Bd: FloatArray, Q: FloatArray, R: FloatArray) -> LQRDesign:
    """Infinite-horizon discrete LQR.

    Minimises ``sum_k z_k^T Q z_k + u_k^T R u_k`` subject to
    ``z_{k+1} = Ad z_k + Bd u_k``. The gain is
    ``K = (R + Bd^T P Bd)^{-1} Bd^T P Ad`` where ``P`` solves the DARE.

    Args:
        Ad: Discrete state matrix.
        Bd: Discrete input matrix.
        Q: Positive semi-definite state weight.
        R: Positive definite input weight.

    Returns:
        The :class:`LQRDesign`.
    """
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    poles = np.linalg.eigvals(Ad - Bd @ K).astype(np.complex128)
    return LQRDesign(K=K, P=P, closed_loop_poles=poles, Q=Q, R=R)


def dlqr_delay_compensated(
    Ad: FloatArray, Bd: FloatArray, Q: FloatArray, R: FloatArray, delay_steps: int
) -> LQRDesign:
    """LQR on the delay-augmented model ``z = [x; u_{k-d}; ...; u_{k-1}]``.

    The buffered commands receive zero state weight: they are only included so
    that the optimal gain can *predict* the effect of inputs already in flight.
    This is the discrete-time equivalent of a Smith predictor / finite-spectrum
    assignment.

    Args:
        Ad: Discrete state matrix of the physical plant.
        Bd: Discrete input matrix of the physical plant.
        Q: State weight for the physical states ``(n, n)``.
        R: Input weight.
        delay_steps: Input delay in samples.

    Returns:
        Design whose gain has ``n + delay_steps`` columns.
    """
    Az, Bz = augment_input_delay(Ad, Bd, delay_steps)
    n = Ad.shape[0]
    Qz = np.zeros_like(Az)
    Qz[:n, :n] = Q
    return dlqr(Az, Bz, Qz, R)


def pole_placement_gain(
    Ad: FloatArray, Bd: FloatArray, continuous_poles: list[complex], Ts: float
) -> tuple[FloatArray, ComplexArray]:
    """Discrete pole placement from a continuous-time pole specification.

    The desired s-plane poles are mapped with ``z = exp(s Ts)`` and placed with
    the robust Kautsky-Nichols-Van Dooren algorithm (``scipy.signal.place_poles``).

    Args:
        Ad: Discrete state matrix.
        Bd: Discrete input matrix.
        continuous_poles: Desired closed-loop poles in the s-plane.
        Ts: Sampling period [s].

    Returns:
        ``(K, z_poles)``.
    """
    z_poles = continuous_to_discrete_poles(continuous_poles, Ts)
    res = place_poles(Ad, Bd, z_poles)
    K = np.asarray(res.gain_matrix, dtype=float)
    return K, z_poles


class StateFeedback(Controller):
    """Static state feedback ``u = -K_x (x_hat - x_ref) - K_b u_pending``.

    ``K_b`` is non-empty only for delay-compensated designs, where the gain also
    acts on the commands still travelling through the delay line.
    """

    def __init__(self, K: FloatArray, name: str) -> None:
        """Create the controller.

        Args:
            K: Gain of shape ``(1, 4 + d)``; the extra ``d`` columns act on the
                pending-command buffer.
            name: Display name.
        """
        self.K = np.atleast_2d(np.asarray(K, dtype=float))
        self.name = name

    @property
    def delay_steps(self) -> int:
        """Number of buffered commands the gain expects."""
        return int(self.K.shape[1] - 4)

    def reset(self, batch: int) -> None:
        """Stateless: nothing to reset."""

    def compute(self, obs: Observation) -> FloatArray:
        """Return ``-K [x_hat - x_ref; pending]``."""
        err = state_error(obs.x_hat, obs.ref)
        u = -err @ self.K[0, :4]
        d = self.delay_steps
        if d > 0:
            if obs.pending.shape[1] != d:
                raise ValueError(
                    f"{self.name}: gain designed for delay {d}, actuator has {obs.pending.shape[1]}"
                )
            u = u - obs.pending @ self.K[0, 4:]
        return np.asarray(u, dtype=float)

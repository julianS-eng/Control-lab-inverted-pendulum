"""State estimation from noisy measurements of cart position and pole angle.

* :class:`KalmanFilter` - discrete linear Kalman filter on the upright linear
  model. Combined with an LQR gain it forms the LQG controller.
* :class:`ExtendedKalmanFilter` - EKF on the full nonlinear model, needed for the
  swing-up where the pole travels through the whole circle and the upright
  linearisation is meaningless.

Both use the *current-estimator* form: at sample ``k`` the prediction uses the
force actually applied during the previous period (known exactly, because the
controller issued it and the delay line is deterministic), then the measurement
``y_k`` corrects it before the control law runs. Both are batched over ``B``
independent loops.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import solve_discrete_are

from pendulum_lab.analysis import c2d_zoh, van_loan_process_noise
from pendulum_lab.model import C_MEAS, NX, integrate, linearize
from pendulum_lab.params import CartPoleParams
from pendulum_lab.types import FloatArray

#: Initial standard deviations for velocity states, which are never measured.
INITIAL_VELOCITY_STD = 0.5
INITIAL_RATE_STD = 1.0


def disturbance_input_matrix(p: CartPoleParams) -> FloatArray:
    """Continuous input matrix of a force on the cart and a torque on the pivot at the top.

    Columns: ``[force on cart, torque on pole]``. Obtained from the inverse mass
    matrix at ``theta = 0``.
    """
    M, m, l = float(p.M), float(p.m), float(p.l)
    inertia = float(p.inertia_pivot)
    D = (M + m) * inertia - (m * l) ** 2
    return np.array(
        [
            [0.0, 0.0],
            [inertia / D, -m * l / D],
            [0.0, 0.0],
            [-m * l / D, (M + m) / D],
        ]
    )


def process_noise_covariance(
    p: CartPoleParams, Ts: float, sigma_force: float, sigma_torque: float
) -> FloatArray:
    """Discrete process-noise covariance for the Kalman filter (Van Loan).

    The filter treats unmodelled effects (friction errors, parameter mismatch,
    external pushes) as white force and torque disturbances with continuous
    intensities ``sigma_force^2`` [N^2 s] and ``sigma_torque^2`` [N^2 m^2 s].

    Args:
        p: Nominal parameters.
        Ts: Sampling period [s].
        sigma_force: Force-disturbance intensity (square root).
        sigma_torque: Torque-disturbance intensity (square root).

    Returns:
        ``Qd`` of shape ``(4, 4)``.
    """
    A, _ = linearize(p)
    G = disturbance_input_matrix(p)
    Qc = np.diag([sigma_force**2, sigma_torque**2])
    return van_loan_process_noise(A, G, Qc, Ts)


def steady_state_kalman_gain(
    Ad: FloatArray, C: FloatArray, Qd: FloatArray, R: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Steady-state (current-estimator) Kalman gain from the filtering DARE.

    Args:
        Ad: Discrete state matrix.
        C: Measurement matrix.
        Qd: Process-noise covariance.
        R: Measurement-noise covariance.

    Returns:
        ``(L, P_prior)``: gain ``(n, p)`` and a-priori error covariance.
    """
    P = solve_discrete_are(Ad.T, C.T, Qd, R)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    return L, P


def initial_covariance(R: FloatArray) -> FloatArray:
    """Initial error covariance when the filter starts from one measurement."""
    return np.diag([R[0, 0], INITIAL_VELOCITY_STD**2, R[1, 1], INITIAL_RATE_STD**2])


class Estimator(ABC):
    """Batched state estimator interface."""

    name: str = "estimator"

    @abstractmethod
    def initialize(self, x0: FloatArray, P0: FloatArray) -> None:
        """Set the prior mean ``(B, 4)`` and covariance ``(4, 4)``."""

    @abstractmethod
    def predict(self, u: FloatArray) -> None:
        """Time update with the force applied during the last period ``(B,)``."""

    @abstractmethod
    def update(self, y: FloatArray) -> FloatArray:
        """Measurement update with ``y = [x, theta]`` ``(B, 2)``; returns ``x_hat``."""

    @property
    @abstractmethod
    def x_hat(self) -> FloatArray:
        """Current estimate ``(B, 4)``."""


class KalmanFilter(Estimator):
    """Linear discrete Kalman filter on the ZOH-discretised upright model.

    The covariance recursion is state-independent for a linear model, so a single
    ``(4, 4)`` covariance is shared by the whole batch.
    """

    def __init__(
        self,
        Ad: FloatArray,
        Bd: FloatArray,
        Qd: FloatArray,
        R: FloatArray,
        C: FloatArray = C_MEAS,
        name: str = "KF",
    ) -> None:
        """Create the filter.

        Args:
            Ad: Discrete state matrix.
            Bd: Discrete input matrix.
            Qd: Process-noise covariance.
            R: Measurement-noise covariance.
            C: Measurement matrix.
            name: Display name.
        """
        self.Ad, self.Bd, self.Qd, self.R, self.C = Ad, Bd, Qd, R, C
        self.name = name
        self._x = np.zeros((1, NX))
        self.P = np.eye(NX)
        self.last_innovation = np.zeros((1, C.shape[0]))
        self.last_S = np.eye(C.shape[0])

    @classmethod
    def design(
        cls, p: CartPoleParams, Ts: float, R: FloatArray, sigma_force: float, sigma_torque: float
    ) -> KalmanFilter:
        """Build the filter from physical noise assumptions (:func:`process_noise_covariance`)."""
        A, B = linearize(p)
        Ad, Bd = c2d_zoh(A, B, Ts)
        Qd = process_noise_covariance(p, Ts, sigma_force, sigma_torque)
        return cls(Ad, Bd, Qd, R)

    @property
    def x_hat(self) -> FloatArray:
        """Current estimate ``(B, 4)``."""
        return self._x

    def initialize(self, x0: FloatArray, P0: FloatArray) -> None:
        """Set the prior."""
        self._x = np.array(x0, dtype=float, copy=True)
        self.P = np.array(P0, dtype=float, copy=True)

    def predict(self, u: FloatArray) -> None:
        """``x^- = Ad x + Bd u``, ``P^- = Ad P Ad^T + Qd``."""
        self._x = self._x @ self.Ad.T + np.outer(u, self.Bd[:, 0])
        self.P = self.Ad @ self.P @ self.Ad.T + self.Qd

    def update(self, y: FloatArray) -> FloatArray:
        """Joseph-form measurement update."""
        C, R = self.C, self.R
        S = C @ self.P @ C.T + R
        K = self.P @ C.T @ np.linalg.inv(S)
        innov = y - self._x @ C.T
        self._x = self._x + innov @ K.T
        I_KC = np.eye(NX) - K @ C
        self.P = I_KC @ self.P @ I_KC.T + K @ R @ K.T
        self.last_innovation, self.last_S = innov, S
        return self._x


class ExtendedKalmanFilter(Estimator):
    """EKF on the nonlinear cart-pole, batched with one covariance per loop.

    The prediction integrates the nominal nonlinear model over one period with
    RK4; the state-transition Jacobian is the central-difference Jacobian of
    that discrete map, evaluated for the whole batch in a single vectorised call.
    """

    def __init__(
        self,
        p: CartPoleParams,
        Ts: float,
        Qd: FloatArray,
        R: FloatArray,
        substeps: int = 2,
        eps: float = 1e-5,
        name: str = "EKF",
    ) -> None:
        """Create the filter.

        Args:
            p: Nominal parameters.
            Ts: Sampling period [s].
            Qd: Process-noise covariance ``(4, 4)``.
            R: Measurement-noise covariance ``(2, 2)``.
            substeps: RK4 substeps of the prediction model.
            eps: Finite-difference step for the Jacobian.
            name: Display name.
        """
        if not p.is_nominal():
            raise ValueError("EKF model must use nominal (scalar) parameters")
        self.p, self.Ts, self.Qd, self.R = p, Ts, Qd, R
        self.substeps, self.eps = substeps, eps
        self.name = name
        self._x = np.zeros((1, NX))
        self.P = np.eye(NX)[None]
        self.last_innovation = np.zeros((1, 2))
        self.last_S = np.eye(2)[None]

    @classmethod
    def design(
        cls, p: CartPoleParams, Ts: float, R: FloatArray, sigma_force: float, sigma_torque: float
    ) -> ExtendedKalmanFilter:
        """Build the EKF with the same noise model as the linear filter."""
        Qd = process_noise_covariance(p, Ts, sigma_force, sigma_torque)
        return cls(p, Ts, Qd, R)

    @property
    def x_hat(self) -> FloatArray:
        """Current estimate ``(B, 4)``."""
        return self._x

    def initialize(self, x0: FloatArray, P0: FloatArray) -> None:
        """Set the prior; the covariance is replicated over the batch."""
        self._x = np.array(x0, dtype=float, copy=True)
        self.P = np.repeat(np.asarray(P0, dtype=float)[None], self._x.shape[0], axis=0)

    def predict(self, u: FloatArray) -> None:
        """Propagate mean through the nonlinear model and covariance through its Jacobian."""
        B = self._x.shape[0]
        offsets = np.concatenate([np.zeros((1, NX)), self.eps * np.eye(NX), -self.eps * np.eye(NX)])
        pts = self._x[:, None, :] + offsets[None]  # (B, 9, 4)
        out = integrate(pts, np.asarray(u)[:, None], self.p, self.Ts, self.substeps)
        F = np.transpose((out[:, 1 : 1 + NX] - out[:, 1 + NX :]) / (2 * self.eps), (0, 2, 1))
        self._x = out[:, 0]
        self.P = F @ self.P @ np.transpose(F, (0, 2, 1)) + self.Qd[None]
        assert self.P.shape == (B, NX, NX)

    def update(self, y: FloatArray) -> FloatArray:
        """Joseph-form update with the linear measurement ``y = [x, theta]``."""
        idx = [0, 2]
        PCt = self.P[:, :, idx]  # (B, 4, 2)
        S = PCt[:, idx, :] + self.R[None]
        K = PCt @ np.linalg.inv(S)
        innov = y - self._x[:, idx]
        self._x = self._x + np.einsum("bij,bj->bi", K, innov)
        KC = np.zeros_like(self.P)
        KC[:, :, idx] = K
        I_KC = np.eye(NX)[None] - KC
        self.P = I_KC @ self.P @ np.transpose(I_KC, (0, 2, 1)) + K @ self.R @ np.transpose(
            K, (0, 2, 1)
        )
        self.last_innovation, self.last_S = innov, S
        return self._x

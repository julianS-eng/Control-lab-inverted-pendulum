"""Cascaded discrete PID: an outer cart-position loop that commands the pole angle.

Structure::

    r --(+)-> [PID_x] -- theta_ref --(+)-> [PID_theta] -- u --> plant
          -x ^                          -theta ^

* **Inner loop** (fast): ``u = Kp e + Ki int(e) + Kd d(theta)/dt`` with
  ``e = theta - theta_ref``. Pushing the cart towards the side the pole leans to
  brings the pole back, hence the *positive* sign on ``theta``.
* **Outer loop** (slow): ``theta_ref = Kp e_x + Ki int(e_x) - Kd dx/dt`` with
  ``e_x = r - x``. Holding the pole at a small angle ``theta_ref`` makes the
  closed inner loop accelerate the cart at roughly ``g * theta_ref``, so the
  outer loop sees an (approximate) double integrator. The right-half-plane zero
  of ``x/u`` at ``+sqrt(m g l / (J + m l^2))`` limits the outer bandwidth.

Implementation details that matter on real hardware:

* derivative **on measurement** (no derivative kick on reference steps),
* first-order derivative filter ``s / (tau s + 1)`` discretised with backward Euler,
* conditional-integration anti-windup on both loops,
* the angle reference is clipped to ``+/- theta_ref_max``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendulum_lab.controllers.base import Controller, Observation
from pendulum_lab.lti import DiscreteLTI
from pendulum_lab.model import wrap_angle
from pendulum_lab.types import FloatArray

#: Internal state layout: [I_x, Dx_prev, x_prev, I_th, Dth_prev, th_prev].
_NQ = 6


@dataclass(frozen=True)
class PIDGains:
    """Gains of one PID loop.

    Attributes:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        tau: Derivative-filter time constant [s].
    """

    kp: float
    ki: float
    kd: float
    tau: float


class CascadePID(Controller):
    """Batched cascaded PID acting only on the measured ``x`` and ``theta``."""

    def __init__(
        self,
        outer: PIDGains,
        inner: PIDGains,
        Ts: float,
        u_max: float,
        theta_ref_max: float = 0.2,
        name: str = "PID cascade",
    ) -> None:
        """Create the controller.

        Args:
            outer: Cart-position loop gains (output in rad).
            inner: Pole-angle loop gains (output in N).
            Ts: Sampling period [s].
            u_max: Actuator saturation used by the anti-windup logic [N].
            theta_ref_max: Clip on the angle reference produced by the outer loop [rad].
            name: Display name.
        """
        self.outer = outer
        self.inner = inner
        self.Ts = Ts
        self.u_max = u_max
        self.theta_ref_max = theta_ref_max
        self.name = name
        self.q: FloatArray = np.zeros((1, _NQ))
        self._primed = False

    def reset(self, batch: int) -> None:
        """Zero integrators and derivative filters."""
        self.q = np.zeros((batch, _NQ))
        self._primed = False

    def _filtered_derivative(
        self, gains: PIDGains, d_prev: FloatArray, m_prev: FloatArray, m: FloatArray
    ) -> FloatArray:
        alpha = gains.tau / (gains.tau + self.Ts)
        beta = 1.0 / (gains.tau + self.Ts)
        return np.asarray(alpha * d_prev + beta * (m - m_prev), dtype=float)

    def _step(
        self, q: FloatArray, x: FloatArray, th: FloatArray, r: FloatArray, *, linear: bool
    ) -> tuple[FloatArray, FloatArray]:
        """One controller update; returns ``(u, q_next)``.

        With ``linear=True`` the reference clip and anti-windup are bypassed, which
        yields the exact LTI model used for margin analysis.
        """
        o, i = self.outer, self.inner
        dx = self._filtered_derivative(o, q[:, 1], q[:, 2], x)
        e_x = r - x
        th_ref_raw = o.kp * e_x + o.ki * q[:, 0] - o.kd * dx
        th_ref = (
            th_ref_raw if linear else np.clip(th_ref_raw, -self.theta_ref_max, self.theta_ref_max)
        )
        dth = self._filtered_derivative(i, q[:, 4], q[:, 5], th)
        e_th = th - th_ref
        u = i.kp * e_th + i.ki * q[:, 3] + i.kd * dth

        if linear:
            integrate_x = np.ones_like(x, dtype=bool)
            integrate_th = np.ones_like(x, dtype=bool)
        else:
            # Conditional integration: freeze an integrator while its loop output is
            # saturated *and* the error would push it further into saturation.
            u_sat = np.abs(u) >= self.u_max
            integrate_th = ~u_sat | (np.sign(e_th) != np.sign(u))
            ref_sat = np.abs(th_ref_raw) >= self.theta_ref_max
            integrate_x = ~ref_sat | (np.sign(e_x) != np.sign(th_ref_raw))

        q_next = np.stack(
            [
                q[:, 0] + self.Ts * e_x * integrate_x,
                dx,
                x,
                q[:, 3] + self.Ts * e_th * integrate_th,
                dth,
                th,
            ],
            axis=1,
        )
        return np.asarray(u, dtype=float), q_next

    def compute(self, obs: Observation) -> FloatArray:
        """PID update from the raw measurement ``y = [x, theta]``."""
        x = obs.y[:, 0]
        th = wrap_angle(obs.y[:, 1])
        if not self._primed:
            # Initialise the derivative memories with the first sample (bumpless start).
            self.q[:, 2] = x
            self.q[:, 5] = th
            self._primed = True
        u, self.q = self._step(self.q, x, th, obs.ref, linear=False)
        return u

    def to_lti(self) -> DiscreteLTI:
        """Exact discrete LTI model ``y -> u`` (reference = 0, no saturation).

        Obtained by probing the linear update with unit vectors, which guarantees
        the analysis model and the simulated controller are the same equations.
        """
        Ac = np.zeros((_NQ, _NQ))
        Bc = np.zeros((_NQ, 2))
        Cc = np.zeros((1, _NQ))
        Dc = np.zeros((1, 2))
        zero = np.zeros(1)
        for j in range(_NQ):
            q = np.zeros((1, _NQ))
            q[0, j] = 1.0
            u, qn = self._step(q, zero, zero, zero, linear=True)
            Ac[:, j] = qn[0]
            Cc[0, j] = u[0]
        for j in range(2):
            v = np.zeros(2)
            v[j] = 1.0
            u, qn = self._step(np.zeros((1, _NQ)), v[:1], v[1:], zero, linear=True)
            Bc[:, j] = qn[0]
            Dc[0, j] = u[0]
        # Drop integrator states that are disabled (ki = 0): they would otherwise
        # appear as uncontrollable/unobservable eigenvalues at z = 1.
        keep = [
            j
            for j in range(_NQ)
            if not ((j == 0 and self.outer.ki == 0.0) or (j == 3 and self.inner.ki == 0.0))
        ]
        return DiscreteLTI(A=Ac[np.ix_(keep, keep)], B=Bc[keep], C=Cc[:, keep], D=Dc, Ts=self.Ts)

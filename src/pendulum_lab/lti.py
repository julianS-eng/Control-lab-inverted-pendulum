"""Discrete LTI models of the loops and classical robustness margins.

The simulator in :mod:`pendulum_lab.simulation` executes the nonlinear plant and
the controller *objects*. This module builds the equivalent **linear** loop
(linearised plant + delay line + controller dynamics) so that stability margins
can be computed exactly and cross-checked against the simulation.

Loop convention: the loop is broken at the plant input (the actuator). With
``G_c`` the controller transfer ``y -> u`` and ``G_p`` the plant ``u -> y``, the
loop transfer is ``L(z) = -G_c(z) G_p(z)`` so the closed loop is stable iff the
Nyquist plot of ``L`` satisfies the Nyquist criterion w.r.t. ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from pendulum_lab.analysis import augment_input_delay, spectral_radius
from pendulum_lab.types import ComplexArray, FloatArray


@dataclass(frozen=True)
class DiscreteLTI:
    """Discrete state-space model ``q+ = A q + B v``, ``w = C q + D v``.

    Attributes:
        A: State matrix ``(n, n)`` (``n`` may be zero).
        B: Input matrix ``(n, p)``.
        C: Output matrix ``(m, n)``.
        D: Feed-through ``(m, p)``.
        Ts: Sampling period [s].
    """

    A: FloatArray
    B: FloatArray
    C: FloatArray
    D: FloatArray
    Ts: float

    @property
    def n_states(self) -> int:
        """Number of internal states."""
        return int(self.A.shape[0])

    def freqresp(self, omega: FloatArray) -> ComplexArray:
        """Frequency response on the unit circle, shape ``(len(omega), m, p)``.

        Args:
            omega: Angular frequencies [rad/s], below the Nyquist frequency.

        Returns:
            Complex frequency response.
        """
        z = np.exp(1j * np.asarray(omega) * self.Ts)
        n = self.n_states
        out = np.empty((z.size, self.D.shape[0], self.D.shape[1]), dtype=complex)
        eye = np.eye(n)
        for i, zi in enumerate(z):
            if n == 0:
                out[i] = self.D
            else:
                out[i] = self.C @ np.linalg.solve(zi * eye - self.A, self.B) + self.D
        return out


def plant_lti(
    Ad: FloatArray, Bd: FloatArray, C_out: FloatArray, delay_steps: int, Ts: float
) -> DiscreteLTI:
    """Linearised plant preceded by a ``d``-sample input delay line.

    Args:
        Ad: Discrete state matrix of the physical plant ``(4, 4)``.
        Bd: Discrete input matrix ``(4, 1)``.
        C_out: Output matrix on the *physical* state; pass ``None``-like identity
            of size ``4 + d`` via :func:`full_state_output` to expose the buffer.
        delay_steps: Actuator delay in samples.
        Ts: Sampling period [s].

    Returns:
        The augmented plant; ``C`` is ``C_out`` padded with zeros on the buffer
        unless ``C_out`` already has ``4 + d`` columns.
    """
    Az, Bz = augment_input_delay(Ad, Bd, delay_steps)
    nz = Az.shape[0]
    if C_out.shape[1] == nz:
        C = C_out
    else:
        C = np.hstack([C_out, np.zeros((C_out.shape[0], nz - C_out.shape[1]))])
    return DiscreteLTI(A=Az, B=Bz, C=C, D=np.zeros((C.shape[0], 1)), Ts=Ts)


def static_gain_lti(K: FloatArray, Ts: float) -> DiscreteLTI:
    """Memory-less state feedback ``u = -K v`` as a zero-state LTI."""
    K = np.atleast_2d(K)
    p = K.shape[1]
    return DiscreteLTI(A=np.zeros((0, 0)), B=np.zeros((0, p)), C=np.zeros((1, 0)), D=-K, Ts=Ts)


def observer_controller_lti(
    Ad: FloatArray,
    Bd: FloatArray,
    C: FloatArray,
    L: FloatArray,
    K: FloatArray,
    delay_steps: int,
    Ts: float,
) -> DiscreteLTI:
    """Observer-based controller (current Kalman estimator + state feedback) as an LTI.

    Controller state ``q = [x_hat^-; b]`` with ``b = [u_{k-d}, ..., u_{k-1}]`` the
    controller's own copy of the commands in the delay line::

        x_hat   = (I - L C) x_hat^- + L y
        u       = -K_x x_hat - K_b b
        x_hat^-+ = Ad x_hat + Bd b_0          (Ad x_hat + Bd u if d = 0)
        b+      = [b_1, ..., b_{d-1}, u]

    Args:
        Ad: Discrete state matrix ``(4, 4)``.
        Bd: Discrete input matrix ``(4, 1)``.
        C: Measurement matrix ``(2, 4)``.
        L: Kalman gain ``(4, 2)``.
        K: Feedback gain ``(1, 4)`` or ``(1, 4 + d)`` (delay-compensated).
        delay_steps: Actuator delay ``d``.
        Ts: Sampling period [s].

    Returns:
        LTI from ``y`` (2 inputs) to ``u`` (1 output).
    """
    n = Ad.shape[0]
    d = int(delay_steps)
    K = np.atleast_2d(K)
    Kx = K[:, :n]
    Kb = K[:, n:] if K.shape[1] > n else np.zeros((1, d))
    if Kb.shape[1] != d:
        raise ValueError("gain buffer columns do not match the delay")
    I_LC = np.eye(n) - L @ C
    Cc = np.hstack([-Kx @ I_LC, -Kb])
    Dc = -Kx @ L
    nq = n + d
    Ac = np.zeros((nq, nq))
    Bc = np.zeros((nq, C.shape[0]))
    if d == 0:
        Ac[:n, :n] = Ad @ I_LC + Bd @ Cc[:, :n]
        Bc[:n] = Ad @ L + Bd @ Dc
    else:
        Ac[:n, :n] = Ad @ I_LC
        Ac[:n, n] = Bd[:, 0]
        Bc[:n] = Ad @ L
        for i in range(d - 1):
            Ac[n + i, n + i + 1] = 1.0
        Ac[-1, :] = Cc[0]
        Bc[-1, :] = Dc[0]
    return DiscreteLTI(A=Ac, B=Bc, C=Cc, D=Dc, Ts=Ts)


def closed_loop_matrix(plant: DiscreteLTI, ctrl: DiscreteLTI, gain: float = 1.0) -> FloatArray:
    """Closed-loop state matrix with an actuator gain perturbation ``k``.

    ``z+ = Ap z + k Bp u``, ``u = Cc q + Dc Cp z``, ``q+ = Ac q + Bc Cp z``.
    """
    Ap, Bp, Cp = plant.A, plant.B, plant.C
    top = np.hstack([Ap + gain * Bp @ ctrl.D @ Cp, gain * Bp @ ctrl.C])
    bot = np.hstack([ctrl.B @ Cp, ctrl.A])
    return np.vstack([top, bot])


def loop_transfer(plant: DiscreteLTI, ctrl: DiscreteLTI, omega: FloatArray) -> ComplexArray:
    """Loop transfer at the plant input ``L = -G_c G_p`` (SISO), shape ``(len(omega),)``."""
    Gp = plant.freqresp(omega)
    Gc = ctrl.freqresp(omega)
    return np.asarray(-(Gc @ Gp)[:, 0, 0])


@dataclass(frozen=True)
class Margins:
    """Classical robustness margins of a loop broken at the actuator.

    Attributes:
        stable: Whether the nominal closed loop is stable.
        spectral_radius: Largest closed-loop eigenvalue magnitude.
        gain_low: Smallest actuator gain factor keeping stability (``< 1``).
        gain_high: Largest actuator gain factor keeping stability (``> 1``, may be inf).
        phase_margin_deg: Smallest phase change (lag or lead) that destabilises [deg].
        crossover_rad_s: Gain-crossover frequency of the critical crossing [rad/s].
        delay_margin_s: Extra pure delay tolerated, from the phase margin [s].
        delay_margin_samples: Largest number of extra whole samples of delay
            that keep the loop stable (exact, via delay augmentation).
    """

    stable: bool
    spectral_radius: float
    gain_low: float
    gain_high: float
    phase_margin_deg: float
    crossover_rad_s: float
    delay_margin_s: float
    delay_margin_samples: int

    @property
    def gain_low_db(self) -> float:
        """Lower gain margin in dB (negative)."""
        return float(20 * np.log10(self.gain_low)) if self.gain_low > 0 else -np.inf

    @property
    def gain_high_db(self) -> float:
        """Upper gain margin in dB (positive)."""
        return float(20 * np.log10(self.gain_high)) if np.isfinite(self.gain_high) else np.inf


def _stable(plant: DiscreteLTI, ctrl: DiscreteLTI, k: float) -> bool:
    return spectral_radius(closed_loop_matrix(plant, ctrl, k)) < 1.0


def _bisect_gain(plant: DiscreteLTI, ctrl: DiscreteLTI, k_in: float, k_out: float) -> float:
    for _ in range(60):
        mid = np.sqrt(k_in * k_out)
        if _stable(plant, ctrl, mid):
            k_in = mid
        else:
            k_out = mid
    return float(np.sqrt(k_in * k_out))


def gain_margin_interval(
    plant: DiscreteLTI, ctrl: DiscreteLTI, k_min: float = 1e-3, k_max: float = 1e3
) -> tuple[float, float]:
    """Interval of actuator gains ``(k_low, k_high)`` around 1 keeping the loop stable.

    Computed directly from closed-loop eigenvalues (no frequency-response
    interpolation), then refined by bisection. Unstable open-loop plants have a
    finite *lower* gain margin: too little actuator authority lets the pole fall.
    """
    if not _stable(plant, ctrl, 1.0):
        return float("nan"), float("nan")
    grid_hi = np.geomspace(1.0, k_max, 200)
    k_high = float("inf")
    for a, b in pairwise(grid_hi):
        if not _stable(plant, ctrl, b):
            k_high = _bisect_gain(plant, ctrl, a, b)
            break
    grid_lo = np.geomspace(1.0, k_min, 200)
    k_low = 0.0
    for a, b in pairwise(grid_lo):
        if not _stable(plant, ctrl, b):
            k_low = _bisect_gain(plant, ctrl, a, b)
            break
    return k_low, k_high


def _extra_delay_stable(plant: DiscreteLTI, ctrl: DiscreteLTI, extra: int) -> bool:
    """Stability with ``extra`` additional samples of delay inserted at the actuator."""
    if extra == 0:
        return _stable(plant, ctrl, 1.0)
    n = plant.n_states
    Az, Bz = augment_input_delay(plant.A, plant.B, extra)
    Cz = np.hstack([plant.C, np.zeros((plant.C.shape[0], extra))])
    delayed = DiscreteLTI(A=Az, B=Bz, C=Cz, D=plant.D, Ts=plant.Ts)
    assert delayed.n_states == n + extra
    return _stable(delayed, ctrl, 1.0)


def compute_margins(
    plant: DiscreteLTI, ctrl: DiscreteLTI, n_freq: int = 4000, max_extra_delay: int = 50
) -> Margins:
    """Gain, phase and delay margins of the loop broken at the actuator.

    Args:
        plant: Plant LTI (including the nominal actuator delay).
        ctrl: Controller LTI.
        n_freq: Number of log-spaced frequency points up to Nyquist.
        max_extra_delay: Search limit for the integer-sample delay margin.

    Returns:
        The :class:`Margins`.
    """
    rho = spectral_radius(closed_loop_matrix(plant, ctrl, 1.0))
    if rho >= 1.0:
        nan = float("nan")
        return Margins(False, rho, nan, nan, nan, nan, nan, 0)
    k_low, k_high = gain_margin_interval(plant, ctrl)

    w_nyq = np.pi / plant.Ts
    omega = np.geomspace(1e-3, w_nyq * 0.9999, n_freq)
    L = loop_transfer(plant, ctrl, omega)
    mag = np.abs(L) - 1.0
    idx = np.nonzero(np.sign(mag[:-1]) != np.sign(mag[1:]))[0]
    pm_best = float("inf")
    wc_best = float("nan")
    dm_best = float("inf")
    for i in idx:
        lo, hi = omega[i], omega[i + 1]
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            m_mid = np.abs(loop_transfer(plant, ctrl, np.array([mid]))[0]) - 1.0
            if np.sign(m_mid) == np.sign(mag[i]):
                lo = mid
            else:
                hi = mid
        wc = float(np.sqrt(lo * hi))
        Lc = loop_transfer(plant, ctrl, np.array([wc]))[0]
        pm = float(np.degrees(np.angle(Lc * np.exp(-1j * np.pi))))  # angle(L) + 180 deg
        lag_needed = np.radians(pm) if pm > 0 else np.radians(pm) + 2 * np.pi
        dm_best = min(dm_best, float(lag_needed / wc))
        if abs(pm) < abs(pm_best):
            pm_best, wc_best = pm, wc

    extra = 0
    while extra < max_extra_delay and _extra_delay_stable(plant, ctrl, extra + 1):
        extra += 1
    return Margins(
        stable=True,
        spectral_radius=rho,
        gain_low=k_low,
        gain_high=k_high,
        phase_margin_deg=abs(pm_best),
        crossover_rad_s=wc_best,
        delay_margin_s=dm_best,
        delay_margin_samples=extra,
    )

"""Performance metrics computed from logged trajectories.

All functions are vectorised over the batch dimension: signals have shape
``(N, B)`` (time first) and results have shape ``(B,)``.

Definitions
-----------
* **Settling time** - the first instant after which the signal stays inside the
  tolerance band for the rest of the record. ``NaN`` if it never settles.
* **Overshoot** - peak excursion beyond the final reference value, as a
  percentage of the step size.
* **Undershoot** - peak initial motion *opposite* to the step (the signature of
  the right-half-plane zero of ``x/u``), as a percentage of the step size.
* **Control effort** - ``int u^2 dt`` [N^2 s], RMS force and peak force.
* **Success** - the loop ends balanced and centred and never left the rail
  nor let the pole fall past horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendulum_lab.model import wrap_angle
from pendulum_lab.simulation import SimResult
from pendulum_lab.types import BoolArray, FloatArray


def settling_time(t: FloatArray, err: FloatArray, band: float | FloatArray) -> FloatArray:
    """Time after which ``|err| <= band`` holds until the end of the record.

    Args:
        t: Times ``(N,)`` [s].
        err: Error signal ``(N,)`` or ``(N, B)``.
        band: Absolute tolerance, scalar or ``(B,)``.

    Returns:
        Settling times ``(B,)`` (``NaN`` when the final sample is outside the band).
    """
    e = np.abs(np.asarray(err, dtype=float))
    if e.ndim == 1:
        e = e[:, None]
    outside = e > band
    N = e.shape[0]
    # index of the last sample outside the band (-1 if always inside)
    rev = outside[::-1]
    any_out = rev.any(axis=0)
    last_out = np.where(any_out, N - 1 - np.argmax(rev, axis=0), -1)
    ts = np.where(last_out + 1 < N, t[np.clip(last_out + 1, 0, N - 1)], np.nan)
    ts = np.where(outside[-1], np.nan, ts)
    return np.asarray(ts, dtype=float)


def overshoot_percent(signal: FloatArray, start: float, final: float) -> FloatArray:
    """Peak overshoot beyond ``final`` as a percentage of ``|final - start|`` (>= 0)."""
    s = np.asarray(signal, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    step = final - start
    excess = (s - final) * np.sign(step)
    return np.asarray(np.maximum(excess.max(axis=0), 0.0) / abs(step) * 100.0)


def undershoot_percent(signal: FloatArray, start: float, final: float) -> FloatArray:
    """Peak motion opposite to the step direction, in percent of the step (>= 0)."""
    s = np.asarray(signal, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    step = final - start
    wrong = (start - s) * np.sign(step)
    return np.asarray(np.maximum(wrong.max(axis=0), 0.0) / abs(step) * 100.0)


def control_energy(u: FloatArray, Ts: float) -> FloatArray:
    """``int u^2 dt`` for a zero-order-hold input [N^2 s]."""
    return np.asarray(np.sum(np.asarray(u) ** 2, axis=0) * Ts)


@dataclass(frozen=True)
class SuccessCriteria:
    """Thresholds that define a successful run.

    Attributes:
        final_window: Length of the final window inspected [s].
        theta_tol: Max ``|theta|`` in the final window [rad].
        x_tol: Max ``|x - r|`` in the final window [m].
        fall_angle: A pole beyond this angle *after* ``fall_check_from`` counts
            as fallen [rad].
        fall_check_from: Time from which falling is checked (0 for balancing
            tasks; the swing-up starts hanging) [s].
    """

    final_window: float = 1.0
    theta_tol: float = 0.05
    x_tol: float = 0.10
    fall_angle: float = np.pi / 2
    fall_check_from: float = 0.0


def success(res: SimResult, crit: SuccessCriteria | None = None) -> BoolArray:
    """Per-loop success flag (see :class:`SuccessCriteria`)."""
    crit = crit or SuccessCriteria()
    n_win = max(1, round(crit.final_window / res.Ts))
    th = wrap_angle(res.x[1:, :, 2])
    x_err = res.x[1:, :, 0] - res.ref
    balanced = np.all(np.abs(th[-n_win:]) < crit.theta_tol, axis=0)
    centred = np.all(np.abs(x_err[-n_win:]) < crit.x_tol, axis=0)
    k0 = round(crit.fall_check_from / res.Ts)
    fell = np.any(np.abs(th[k0:]) > crit.fall_angle, axis=0)
    finite = np.all(np.isfinite(res.x), axis=(0, 2))
    return np.asarray(balanced & centred & ~fell & ~res.track_violation & finite)


@dataclass(frozen=True)
class StepMetrics:
    """Metrics of a cart-position step (each entry is per loop, shape ``(B,)``)."""

    settling_time: FloatArray
    overshoot: FloatArray
    undershoot: FloatArray
    peak_angle: FloatArray
    energy: FloatArray
    rms_force: FloatArray
    peak_force: FloatArray
    saturated_fraction: FloatArray


def step_metrics(
    res: SimResult,
    amplitude: float,
    t_step: float,
    band_fraction: float = 0.02,
    theta_band: float = 0.01,
    u_max: float | None = None,
) -> StepMetrics:
    """Evaluate a cart-position step response on the *true* state.

    Settling requires both ``|x - r| <= band_fraction * |amplitude|`` and
    ``|theta| <= theta_band``; times are measured from the step instant.
    """
    k0 = round(t_step / res.Ts)
    t = res.t[k0:] - t_step
    x = res.x[k0:, :, 0]
    th = wrap_angle(res.x[k0:, :, 2])
    band = band_fraction * abs(amplitude)
    combined = np.maximum(np.abs(x - amplitude) / band, np.abs(th) / theta_band)
    ts = settling_time(t, combined, 1.0)
    u = res.u[k0:]
    sat = (
        np.mean(np.abs(res.u_cmd[k0:]) >= u_max, axis=0)
        if u_max is not None
        else np.zeros(res.batch)
    )
    return StepMetrics(
        settling_time=ts,
        overshoot=overshoot_percent(x, 0.0, amplitude),
        undershoot=undershoot_percent(x, 0.0, amplitude),
        peak_angle=np.abs(th).max(axis=0),
        energy=control_energy(u, res.Ts),
        rms_force=np.sqrt(np.mean(u**2, axis=0)),
        peak_force=np.abs(u).max(axis=0),
        saturated_fraction=np.asarray(sat, dtype=float),
    )


@dataclass(frozen=True)
class RecoveryMetrics:
    """Metrics of recovery from an initial pole deflection (per loop, ``(B,)``)."""

    settling_time: FloatArray
    peak_cart_excursion: FloatArray
    energy: FloatArray
    rms_force: FloatArray
    peak_force: FloatArray
    steady_rms_angle: FloatArray
    steady_rms_force: FloatArray


def recovery_metrics(
    res: SimResult, theta_band: float = 0.01, x_band: float = 0.02, steady_window: float = 2.0
) -> RecoveryMetrics:
    """Evaluate regulation from a perturbed initial state.

    Settling requires ``|theta| <= theta_band`` and ``|x| <= x_band``. The
    steady-state RMS values over the final ``steady_window`` seconds quantify how
    much sensor noise and disturbances leak into the pole angle and the force.
    """
    th = wrap_angle(res.x[:, :, 2])
    x = res.x[:, :, 0]
    combined = np.maximum(np.abs(th) / theta_band, np.abs(x) / x_band)
    n_ss = round(steady_window / res.Ts)
    return RecoveryMetrics(
        settling_time=settling_time(res.t, combined, 1.0),
        peak_cart_excursion=np.abs(x).max(axis=0),
        energy=control_energy(res.u, res.Ts),
        rms_force=np.sqrt(np.mean(res.u**2, axis=0)),
        peak_force=np.abs(res.u).max(axis=0),
        steady_rms_angle=np.sqrt(np.mean(th[-n_ss:] ** 2, axis=0)),
        steady_rms_force=np.sqrt(np.mean(res.u[-n_ss:] ** 2, axis=0)),
    )


@dataclass(frozen=True)
class SwingUpMetrics:
    """Metrics of a swing-up run (per loop, ``(B,)``)."""

    catch_time: FloatArray
    settling_time: FloatArray
    peak_cart_excursion: FloatArray
    energy: FloatArray
    peak_force: FloatArray
    mode_switches: FloatArray


def swingup_metrics(
    res: SimResult, theta_band: float = 0.05, x_band: float = 0.05
) -> SwingUpMetrics:
    """Catch time (first switch to balance), settling to the upright band, excursions."""
    mode = np.nan_to_num(res.mode, nan=0.0)
    balanced = mode > 0.5
    any_bal = balanced.any(axis=0)
    first = np.argmax(balanced, axis=0)
    catch = np.where(any_bal, res.t[first], np.nan)
    th = wrap_angle(res.x[1:, :, 2])
    x = res.x[1:, :, 0]
    combined = np.maximum(np.abs(th) / theta_band, np.abs(x) / x_band)
    switches = np.sum(np.abs(np.diff(mode, axis=0)) > 0.5, axis=0)
    return SwingUpMetrics(
        catch_time=np.asarray(catch, dtype=float),
        settling_time=settling_time(res.t[1:], combined, 1.0),
        peak_cart_excursion=np.abs(res.x[:, :, 0]).max(axis=0),
        energy=control_energy(res.u, res.Ts),
        peak_force=np.abs(res.u).max(axis=0),
        mode_switches=switches.astype(float),
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Clamp against round-off so the interval always contains the point estimate.
    return float(min(max(centre - half, 0.0), p)), float(max(min(centre + half, 1.0), p))

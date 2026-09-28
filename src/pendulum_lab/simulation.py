"""Discrete-time closed-loop simulator with sampling, noise, saturation and delay.

Timeline of one sampling period ``[t_k, t_{k+1})``::

    y_k = [x, theta](t_k) + v_k                     (sensor, noisy)
    x_hat_k = estimator(x_hat_{k-1}, u_app_{k-1}, y_k)
    u_cmd_k = controller(x_hat_k, y_k, r_k)
    u_sat_k = clip(u_cmd_k, -u_max, u_max)           (actuator saturation)
    u_app_k = u_sat_{k-d}                            (d-sample transport delay)
    plant integrated with u_app_k + w_k held constant (ZOH + disturbance)

The plant is integrated with ``substeps`` RK4 steps per period, so the
continuous nonlinear dynamics are resolved much more finely than the controller
rate. Everything is batched: ``B`` loops (possibly with different physical
parameters) run in lock-step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pendulum_lab.controllers.base import Controller, Observation
from pendulum_lab.estimation import Estimator, initial_covariance
from pendulum_lab.model import NX, integrate
from pendulum_lab.params import CartPoleParams, LabConfig, SensorConfig
from pendulum_lab.types import BoolArray, FloatArray

Reference = Callable[[float], float]


@dataclass
class SimResult:
    """Logged closed-loop trajectories.

    Shapes use ``N`` samples and ``B`` loops.

    Attributes:
        t: Sample times ``(N + 1,)`` [s].
        x: True states ``(N + 1, B, 4)``.
        x_hat: Estimates used by the controller ``(N, B, 4)``.
        y: Measurements ``(N, B, 2)``.
        u_cmd: Unsaturated commands ``(N, B)`` [N].
        u: Force actually applied to the cart ``(N, B)`` [N] (saturated, delayed).
        ref: Cart reference ``(N, B)`` [m].
        mode: Controller mode ``(N, B)`` (``NaN`` for single-mode controllers).
        track_violation: Whether each loop ever left the rail ``(B,)``.
        Ts: Sampling period [s].
    """

    t: FloatArray
    x: FloatArray
    x_hat: FloatArray
    y: FloatArray
    u_cmd: FloatArray
    u: FloatArray
    ref: FloatArray
    mode: FloatArray
    track_violation: BoolArray
    Ts: float

    @property
    def batch(self) -> int:
        """Number of loops simulated."""
        return int(self.x.shape[1])


def measure(state: FloatArray, sensor: SensorConfig, rng: np.random.Generator) -> FloatArray:
    """Noisy (optionally quantised) measurement of ``[x, theta]``.

    Args:
        state: True state ``(B, 4)``.
        sensor: Sensor model.
        rng: Random generator.

    Returns:
        Measurement ``(B, 2)``.
    """
    B = state.shape[0]
    y = np.stack([state[:, 0], state[:, 2]], axis=1)
    y = y + rng.standard_normal((B, 2)) * np.array([sensor.sigma_x, sensor.sigma_theta])
    if sensor.quant_x:
        y[:, 0] = np.round(y[:, 0] / sensor.quant_x) * sensor.quant_x
    if sensor.quant_theta:
        y[:, 1] = np.round(y[:, 1] / sensor.quant_theta) * sensor.quant_theta
    return np.asarray(y, dtype=np.float64)


def simulate(
    controller: Controller,
    x0: FloatArray,
    config: LabConfig,
    duration: float,
    *,
    estimator: Estimator | None = None,
    plant: CartPoleParams | None = None,
    reference: Reference | None = None,
    disturbance: Reference | None = None,
    seed: int | np.random.Generator = 0,
    estimator_init: Literal["measurement", "true"] = "measurement",
) -> SimResult:
    """Run a batch of closed loops.

    Args:
        controller: Control law (reset internally).
        x0: Initial true states, ``(4,)`` or ``(B, 4)``.
        config: Lab configuration (sampling, actuator, sensor, disturbance).
        duration: Simulated time [s].
        estimator: State estimator. ``None`` means *ideal* full-state feedback:
            the controller receives the exact, noise-free state.
        plant: Physical parameters of the simulated plant(s). Defaults to the
            nominal ``config.params``. Array-valued parameters define a batch.
        reference: Cart-position reference ``r(t)`` [m]; defaults to 0.
        disturbance: Deterministic external force on the cart ``w(t)`` [N]
            (e.g. a push), added to the random disturbance.
        seed: Seed or generator for sensor noise and disturbances.
        estimator_init: ``"measurement"`` starts the estimator from the first
            measurement with zero velocities (what a real system can do);
            ``"true"`` starts it at the true initial state (used to isolate the
            controller's basin of attraction).

    Returns:
        The logged :class:`SimResult`.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    plant = config.params if plant is None else plant
    x = np.atleast_2d(np.asarray(x0, dtype=float)).copy()
    pb = plant.batch_size
    if pb is not None and x.shape[0] == 1:
        x = np.repeat(x, pb, axis=0)
    B = x.shape[0]
    if pb is not None and pb != B:
        raise ValueError(f"plant batch {pb} does not match initial-state batch {B}")

    Ts = config.Ts
    N = round(duration / Ts)
    d = config.actuator.delay_steps
    u_max = config.actuator.u_max

    xs = np.empty((N + 1, B, NX))
    xhs = np.empty((N, B, NX))
    ys = np.empty((N, B, 2))
    ucmd = np.empty((N, B))
    uapp = np.empty((N, B))
    refs = np.empty((N, B))
    modes = np.full((N, B), np.nan)
    xs[0] = x
    violation = np.abs(x[:, 0]) > config.track_limit
    pending = np.zeros((B, d))
    u_prev = np.zeros(B)

    controller.reset(B)
    for k in range(N):
        t = k * Ts
        y = measure(x, config.sensor, rng)
        if estimator is None:
            x_hat = x.copy()
        else:
            if k == 0:
                if estimator_init == "true":
                    x_init = x.copy()
                else:
                    x_init = np.zeros((B, NX))
                    x_init[:, 0], x_init[:, 2] = y[:, 0], y[:, 1]
                estimator.initialize(x_init, initial_covariance(config.sensor.R))
            else:
                estimator.predict(u_prev)
            x_hat = estimator.update(y).copy()
        r = np.full(B, 0.0 if reference is None else float(reference(t)))
        u_c = controller.compute(
            Observation(x_hat=x_hat, y=y, ref=r, pending=pending.copy(), k=k, t=t)
        )
        u_s = np.clip(u_c, -u_max, u_max)
        if d > 0:
            u_now = pending[:, 0].copy()
            pending = np.concatenate([pending[:, 1:], u_s[:, None]], axis=1)
        else:
            u_now = u_s
        w = rng.standard_normal(B) * config.force_disturbance_std
        if disturbance is not None:
            w = w + float(disturbance(t))
        x = integrate(x, u_now + w, plant, Ts, config.substeps)

        xhs[k], ys[k], ucmd[k], uapp[k], refs[k] = x_hat, y, u_c, u_now, r
        mode = controller.mode
        if mode is not None:
            modes[k] = mode
        xs[k + 1] = x
        violation |= np.abs(x[:, 0]) > config.track_limit
        u_prev = u_now

    return SimResult(
        t=np.asarray(np.arange(N + 1) * Ts, dtype=np.float64),
        x=xs,
        x_hat=xhs,
        y=ys,
        u_cmd=ucmd,
        u=uapp,
        ref=refs,
        mode=modes,
        track_violation=violation,
        Ts=Ts,
    )


def pulse(amplitude: float, t_start: float, width: float) -> Reference:
    """Rectangular pulse of ``amplitude`` between ``t_start`` and ``t_start + width``."""

    def f(t: float) -> float:
        return amplitude if t_start <= t < t_start + width - 1e-12 else 0.0

    return f


def step_reference(amplitude: float, t_step: float) -> Reference:
    """Cart-position step of ``amplitude`` metres at ``t_step`` seconds."""

    def ref(t: float) -> float:
        return amplitude if t >= t_step else 0.0

    return ref

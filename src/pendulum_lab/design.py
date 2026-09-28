"""The lab's reference controller designs and the "strategies" compared in every study.

A *strategy* bundles a controller with the information it is allowed to use
(ideal state, Kalman estimate, or raw measurements). Factories return fresh,
un-shared objects so strategies can be run concurrently or repeatedly.

Design choices (and why) are documented next to each constant; the README and
``docs/LEARNING.md`` discuss them at length.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pendulum_lab.analysis import c2d_zoh
from pendulum_lab.controllers import (
    CascadePID,
    Controller,
    EnergySwingUp,
    LQRDesign,
    PIDGains,
    StateFeedback,
    bryson_weights,
    dlqr,
    dlqr_delay_compensated,
    pole_placement_gain,
)
from pendulum_lab.estimation import (
    Estimator,
    ExtendedKalmanFilter,
    KalmanFilter,
    process_noise_covariance,
    steady_state_kalman_gain,
)
from pendulum_lab.lti import (
    DiscreteLTI,
    observer_controller_lti,
    plant_lti,
    static_gain_lti,
)
from pendulum_lab.model import C_MEAS, linearize
from pendulum_lab.params import LabConfig
from pendulum_lab.types import FloatArray


@dataclass(frozen=True)
class DesignChoices:
    """Tunable design parameters of every controller in the lab.

    Attributes:
        lqr_limits: Bryson's-rule limits ``(x, x_dot, theta, theta_dot, u)``:
            0.25 m of cart travel (a quarter of the rail half-length),
            1 m/s of cart speed, 0.1 rad (5.7 deg) of pole tilt, 1.5 rad/s of
            pole rate, and 3 N of force - under a third of the 10 N saturation,
            so that a 0.5 m step leaves headroom for disturbance rejection.
        pole_placement_poles: s-plane poles: a dominant pair with
            ``zeta = 0.8, wn = 2.5 rad/s`` for the cart, the mirror image of the
            unstable open-loop pole (-5.5 rad/s, the minimum-energy choice for
            that mode) and a fast real pole at -8 rad/s.
        pid_outer: Cart-position loop gains (tuned: see ``docs/LEARNING.md``).
        pid_inner: Pole-angle loop gains.
        kf_sigma_force: Kalman process-noise force intensity [N sqrt(s)].
        kf_sigma_torque: Kalman process-noise torque intensity [N m sqrt(s)].
        swing: Keyword arguments for :class:`EnergySwingUp`.
    """

    lqr_limits: tuple[float, float, float, float, float] = (0.25, 1.0, 0.1, 1.5, 3.0)
    pole_placement_poles: tuple[complex, ...] = (-2.0 + 1.5j, -2.0 - 1.5j, -5.5, -8.0)
    pid_outer: PIDGains = field(
        default_factory=lambda: PIDGains(kp=0.16, ki=0.0, kd=0.14, tau=0.05)
    )
    pid_inner: PIDGains = field(
        default_factory=lambda: PIDGains(kp=28.0, ki=10.0, kd=4.5, tau=0.02)
    )
    kf_sigma_force: float = 0.2
    kf_sigma_torque: float = 0.02
    swing: dict[str, Any] = field(
        default_factory=lambda: {
            "k_energy": 30.0,
            "a_max": 6.0,
            "k_x": 3.0,
            "k_v": 3.75,
            "energy_margin": 0.03,
            "adapt_gain": 1.0,
            "adapt_max": 0.5,
            "theta_catch": 0.35,
            "omega_catch": 2.5,
            "theta_release": 0.9,
        }
    )


@dataclass(frozen=True)
class LinearModel:
    """Continuous and ZOH-discretised linearisation about the upright equilibrium."""

    A: FloatArray
    B: FloatArray
    Ad: FloatArray
    Bd: FloatArray
    Ts: float


def linear_model(config: LabConfig) -> LinearModel:
    """Linearise the nominal plant and discretise it at ``config.Ts``."""
    A, B = linearize(config.params)
    Ad, Bd = c2d_zoh(A, B, config.Ts)
    return LinearModel(A, B, Ad, Bd, config.Ts)


def lqr_design(config: LabConfig, choices: DesignChoices | None = None) -> LQRDesign:
    """Discrete LQR with Bryson's-rule weights."""
    choices = choices or DesignChoices()
    lm = linear_model(config)
    Q, R = bryson_weights(*choices.lqr_limits)
    return dlqr(lm.Ad, lm.Bd, Q, R)


def lqr_delay_design(config: LabConfig, choices: DesignChoices | None = None) -> LQRDesign:
    """Delay-compensated discrete LQR for the configured actuator delay."""
    choices = choices or DesignChoices()
    lm = linear_model(config)
    Q, R = bryson_weights(*choices.lqr_limits)
    return dlqr_delay_compensated(lm.Ad, lm.Bd, Q, R, config.actuator.delay_steps)


def kalman_filter(config: LabConfig, choices: DesignChoices | None = None) -> KalmanFilter:
    """Linear Kalman filter with the lab's noise assumptions."""
    choices = choices or DesignChoices()
    return KalmanFilter.design(
        config.params, config.Ts, config.sensor.R, choices.kf_sigma_force, choices.kf_sigma_torque
    )


def extended_kalman_filter(
    config: LabConfig, choices: DesignChoices | None = None
) -> ExtendedKalmanFilter:
    """EKF with the same noise assumptions as the linear filter."""
    choices = choices or DesignChoices()
    return ExtendedKalmanFilter.design(
        config.params, config.Ts, config.sensor.R, choices.kf_sigma_force, choices.kf_sigma_torque
    )


def kalman_gain(config: LabConfig, choices: DesignChoices | None = None) -> FloatArray:
    """Steady-state Kalman gain (used for the LTI loop models)."""
    choices = choices or DesignChoices()
    lm = linear_model(config)
    Qd = process_noise_covariance(
        config.params, config.Ts, choices.kf_sigma_force, choices.kf_sigma_torque
    )
    L, _ = steady_state_kalman_gain(lm.Ad, C_MEAS, Qd, config.sensor.R)
    return L


def pole_placement_design(config: LabConfig, choices: DesignChoices | None = None) -> FloatArray:
    """Pole-placement gain for the configured sampling period."""
    choices = choices or DesignChoices()
    lm = linear_model(config)
    K, _ = pole_placement_gain(lm.Ad, lm.Bd, list(choices.pole_placement_poles), config.Ts)
    return K


@dataclass(frozen=True)
class Strategy:
    """A controller together with the information it uses.

    Attributes:
        key: Short identifier (used in file names and JSON).
        label: Display name.
        make_controller: Factory for a fresh controller.
        make_estimator: Factory for a fresh estimator, or ``None`` for ideal
            full-state feedback (or measurement-only controllers such as PID).
        make_lti: Factory for the ``(plant, controller)`` LTI pair used in margin
            analysis, for the configured delay.
        realistic: ``False`` for idealised references (noise-free full state).
    """

    key: str
    label: str
    make_controller: Callable[[], Controller]
    make_estimator: Callable[[], Estimator] | None
    make_lti: Callable[[], tuple[DiscreteLTI, DiscreteLTI]]
    realistic: bool = True


def balance_strategies(config: LabConfig, choices: DesignChoices | None = None) -> list[Strategy]:
    """All balancing strategies compared in the lab, in display order.

    1. Cascaded PID on raw measurements.
    2. Pole placement + Kalman filter.
    3. LQR with ideal, noise-free full state (reference upper bound).
    4. LQG: LQR + Kalman filter.
    5. LQG with delay compensation (LQR on the delay-augmented model; identical
       to 4 when the configured delay is zero).

    Note that the Kalman filter is always *delay-aware*: it is propagated with
    the force actually applied, which the controller knows because it issued it.
    Only strategy 5 also *predicts* the effect of the commands in flight.
    """
    choices = choices or DesignChoices()
    lm = linear_model(config)
    d = config.actuator.delay_steps
    Ts = config.Ts
    lqr = lqr_design(config, choices)
    lqr_d = lqr_delay_design(config, choices)
    K_pp = pole_placement_design(config, choices)
    L = kalman_gain(config, choices)
    u_max = config.actuator.u_max

    def pid() -> CascadePID:
        return CascadePID(choices.pid_outer, choices.pid_inner, Ts, u_max)

    def kf() -> Estimator:
        return kalman_filter(config, choices)

    plant_meas = plant_lti(lm.Ad, lm.Bd, C_MEAS, d, Ts)
    plant_full = plant_lti(lm.Ad, lm.Bd, np.eye(4), d, Ts)

    strategies = [
        Strategy(
            "pid",
            "PID cascade",
            pid,
            None,
            lambda: (plant_meas, pid().to_lti()),
        ),
        Strategy(
            "pole_placement",
            "Pole placement + KF",
            lambda: StateFeedback(K_pp, "Pole placement + KF"),
            kf,
            lambda: (plant_meas, observer_controller_lti(lm.Ad, lm.Bd, C_MEAS, L, K_pp, d, Ts)),
        ),
        Strategy(
            "lqr_ideal",
            "LQR (ideal state)",
            lambda: StateFeedback(lqr.K, "LQR (ideal state)"),
            None,
            lambda: (plant_full, static_gain_lti(lqr.K, Ts)),
            realistic=False,
        ),
        Strategy(
            "lqg",
            "LQG (LQR + KF)",
            lambda: StateFeedback(lqr.K, "LQG (LQR + KF)"),
            kf,
            lambda: (plant_meas, observer_controller_lti(lm.Ad, lm.Bd, C_MEAS, L, lqr.K, d, Ts)),
        ),
    ]
    strategies.append(
        Strategy(
            "lqg_delay",
            "LQG + delay comp.",
            lambda: StateFeedback(lqr_d.K, "LQG + delay comp."),
            kf,
            lambda: (plant_meas, observer_controller_lti(lm.Ad, lm.Bd, C_MEAS, L, lqr_d.K, d, Ts)),
        )
    )
    return strategies


def swingup_controller(config: LabConfig, choices: DesignChoices | None = None) -> EnergySwingUp:
    """Energy swing-up that hands over to the (delay-compensated when ``d > 0``) LQR."""
    choices = choices or DesignChoices()
    design = lqr_delay_design(config, choices)
    balance = StateFeedback(design.K, "LQR")
    return EnergySwingUp(config.params, balance, **choices.swing)

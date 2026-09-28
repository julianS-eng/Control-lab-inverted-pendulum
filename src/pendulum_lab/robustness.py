"""Robustness studies on the nonlinear plant: Monte Carlo and region of attraction.

Monte Carlo uses **common random numbers**: every strategy is evaluated on the
very same sampled plants, initial conditions and noise sequences, so differences
in success rate are due to the controllers and not to sampling luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pendulum_lab.design import (
    DesignChoices,
    Strategy,
    extended_kalman_filter,
    swingup_controller,
)
from pendulum_lab.metrics import SuccessCriteria, settling_time, success, wilson_interval
from pendulum_lab.model import wrap_angle
from pendulum_lab.params import CartPoleParams, LabConfig
from pendulum_lab.simulation import Reference, SimResult, pulse, simulate, step_reference
from pendulum_lab.types import BoolArray, FloatArray


@dataclass(frozen=True)
class Uncertainty:
    """Parameter spread for Monte Carlo sampling.

    Masses and length are uniform in ``nominal * (1 +/- spread)``; friction
    coefficients are log-uniform in ``[nominal / f, nominal * f]`` because
    friction is typically known only to within a factor. The pole stays a
    uniform rod, so its inertia follows its mass and length.

    Attributes:
        label: Name used in reports.
        cart_mass: Relative half-width for ``M``.
        pole_mass: Relative half-width for ``m``.
        length: Relative half-width for ``l``.
        friction_factor: Multiplicative spread ``f >= 1`` for ``b_c`` and ``b_p``.
    """

    label: str
    cart_mass: float
    pole_mass: float
    length: float
    friction_factor: float


#: Realistic manufacturing / identification uncertainty.
MODERATE = Uncertainty("moderate", cart_mass=0.2, pole_mass=0.2, length=0.2, friction_factor=2.0)
#: Stress test: badly identified plant.
SEVERE = Uncertainty("severe", cart_mass=0.4, pole_mass=0.5, length=0.4, friction_factor=4.0)
#: Beyond any reasonable identification error; used to separate the controllers.
EXTREME = Uncertainty("extreme", cart_mass=0.6, pole_mass=0.7, length=0.6, friction_factor=8.0)
#: All levels, in increasing severity.
LEVELS = (MODERATE, SEVERE, EXTREME)


def sample_plants(
    nominal: CartPoleParams, unc: Uncertainty, n: int, rng: np.random.Generator
) -> CartPoleParams:
    """Draw ``n`` perturbed plants (array-valued parameters of shape ``(n,)``)."""

    def uni(value: float, spread: float) -> FloatArray:
        return value * rng.uniform(1 - spread, 1 + spread, n)

    def logu(value: float, f: float) -> FloatArray:
        return value * np.exp(rng.uniform(-np.log(f), np.log(f), n))

    return CartPoleParams.uniform_rod(
        M=uni(float(nominal.M), unc.cart_mass),
        m=uni(float(nominal.m), unc.pole_mass),
        l=uni(float(nominal.l), unc.length),
        b_c=logu(float(nominal.b_c), unc.friction_factor),
        b_p=logu(float(nominal.b_p), unc.friction_factor),
        g=float(nominal.g),
    )


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcome of one strategy on one Monte Carlo population.

    Attributes:
        key: Strategy key.
        label: Strategy label.
        uncertainty: Uncertainty label.
        success: Per-sample success flags ``(n,)``.
        settling_time: Per-sample settling time ``(n,)`` (``NaN`` when unsettled).
        energy: Per-sample control energy ``(n,)`` [N^2 s].
    """

    key: str
    label: str
    uncertainty: str
    success: BoolArray
    settling_time: FloatArray
    energy: FloatArray

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.success.size)

    @property
    def rate(self) -> float:
        """Success rate in ``[0, 1]``."""
        return float(np.mean(self.success))

    @property
    def ci95(self) -> tuple[float, float]:
        """Wilson 95 % confidence interval of the success rate."""
        return wilson_interval(int(self.success.sum()), self.n)


def run_strategy(
    strategy: Strategy,
    config: LabConfig,
    x0: FloatArray,
    duration: float,
    *,
    plant: CartPoleParams | None = None,
    reference: Reference | None = None,
    disturbance: Reference | None = None,
    seed: int = 0,
    estimator_init: Literal["measurement", "true"] = "measurement",
) -> SimResult:
    """Simulate one strategy with fresh controller/estimator objects."""
    est = strategy.make_estimator() if strategy.make_estimator is not None else None
    return simulate(
        strategy.make_controller(),
        x0,
        config,
        duration,
        estimator=est,
        plant=plant,
        reference=reference,
        disturbance=disturbance,
        seed=seed,
        estimator_init=estimator_init,
    )


@dataclass(frozen=True)
class BalanceScenario:
    """Balancing task used by the Monte Carlo study.

    The loop starts from a random tilt/motion, must then move the cart to a new
    set-point and finally reject an external push - a compact "acceptance
    test" that exercises regulation, tracking and disturbance rejection.

    Attributes:
        tilt: Initial angle drawn from ``U(-tilt, tilt)`` [rad].
        rate: Initial angular rate drawn from ``U(-rate, rate)`` [rad/s].
        position: Initial cart position drawn from ``U(-position, position)`` [m].
        step: Cart set-point step amplitude [m].
        t_step: Step time [s].
        push: Push force amplitude [N].
        t_push: Push start time [s].
        push_width: Push duration [s].
        duration: Simulated time [s].
    """

    tilt: float = 0.25
    rate: float = 0.5
    position: float = 0.1
    step: float = 0.4
    t_step: float = 2.0
    push: float = 8.0
    t_push: float = 6.0
    push_width: float = 0.1
    duration: float = 10.0

    def initial_states(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Draw ``n`` initial states."""
        x0 = np.zeros((n, 4))
        x0[:, 0] = rng.uniform(-self.position, self.position, n)
        x0[:, 2] = rng.uniform(-self.tilt, self.tilt, n)
        x0[:, 3] = rng.uniform(-self.rate, self.rate, n)
        return x0

    @property
    def reference(self) -> Reference:
        """Cart set-point profile."""
        return step_reference(self.step, self.t_step)

    @property
    def disturbance(self) -> Reference:
        """External push profile."""
        return pulse(self.push, self.t_push, self.push_width)


def settle_times(res: SimResult, theta_band: float = 0.02, x_band: float = 0.03) -> FloatArray:
    """Settling time to a combined ``theta``/``x`` band (wrapped angle)."""
    th = wrap_angle(res.x[1:, :, 2])
    err = res.x[1:, :, 0] - res.ref
    combined = np.maximum(np.abs(th) / theta_band, np.abs(err) / x_band)
    return settling_time(res.t[1:], combined, 1.0)


def _summarise(
    key: str, label: str, unc: Uncertainty, res: SimResult, crit: SuccessCriteria | None = None
) -> MonteCarloResult:
    return MonteCarloResult(
        key=key,
        label=label,
        uncertainty=unc.label,
        success=success(res, crit),
        settling_time=settle_times(res),
        energy=np.sum(res.u**2, axis=0) * res.Ts,
    )


def monte_carlo(
    strategies: list[Strategy],
    config: LabConfig,
    unc: Uncertainty,
    n: int,
    seed: int,
    scenario: BalanceScenario | None = None,
) -> list[MonteCarloResult]:
    """Evaluate every balancing strategy on the same ``n`` plants, initial states and noise."""
    scenario = scenario or BalanceScenario()
    rng = np.random.default_rng(seed)
    plants = sample_plants(config.params, unc, n, rng)
    x0 = scenario.initial_states(n, rng)
    noise_seed = int(rng.integers(2**31))
    out = []
    for s in strategies:
        res = run_strategy(
            s,
            config,
            x0,
            scenario.duration,
            plant=plants,
            reference=scenario.reference,
            disturbance=scenario.disturbance,
            seed=noise_seed,
        )
        out.append(_summarise(s.key, s.label, unc, res))
    return out


def monte_carlo_swingup(
    config: LabConfig,
    unc: Uncertainty,
    n: int,
    seed: int,
    choices: DesignChoices | None = None,
    duration: float = 15.0,
) -> MonteCarloResult:
    """Swing-up from the hanging position on ``n`` sampled plants.

    Success only looks at the final window (the pole necessarily passes
    through horizontal on its way up).
    """
    rng = np.random.default_rng(seed)
    plants = sample_plants(config.params, unc, n, rng)
    x0 = np.zeros((n, 4))
    x0[:, 2] = np.pi + rng.uniform(-0.1, 0.1, n)
    noise_seed = int(rng.integers(2**31))
    res = simulate(
        swingup_controller(config, choices),
        x0,
        config,
        duration,
        estimator=extended_kalman_filter(config, choices),
        plant=plants,
        seed=noise_seed,
    )
    crit = SuccessCriteria(fall_check_from=duration)
    return _summarise("swingup", "Energy swing-up + LQR", unc, res, crit)


def roa_grid(
    strategy: Strategy,
    config: LabConfig,
    thetas: FloatArray,
    omegas: FloatArray,
    duration: float,
    seed: int,
    criteria: SuccessCriteria | None = None,
) -> BoolArray:
    """Empirical region of attraction on the ``(theta_0, theta_dot_0)`` plane.

    Every grid point starts with the cart at rest at the origin. The estimator
    is initialised at the true state, so the map isolates the basin of the
    closed loop itself (saturation, delay and noise included) rather than the
    estimator's start-up transient.

    Returns:
        Boolean array ``(len(omegas), len(thetas))`` of successful recoveries.
    """
    TH, OM = np.meshgrid(thetas, omegas)
    x0 = np.zeros((TH.size, 4))
    x0[:, 2] = TH.ravel()
    x0[:, 3] = OM.ravel()
    res = run_strategy(strategy, config, x0, duration, seed=seed, estimator_init="true")
    return success(res, criteria).reshape(TH.shape)

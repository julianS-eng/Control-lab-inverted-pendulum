"""All studies of the lab. Every number and figure in the README is produced here.

Each ``study_*`` function is deterministic (fixed seeds), returns a JSON-able
dictionary, and optionally writes its figure(s). :func:`run_all` executes them
in order and is what ``pendulum-lab all`` calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, solve_continuous_are

from pendulum_lab.analysis import (
    augment_input_delay,
    c2d_zoh,
    controllability_report,
    finite_horizon_gramian,
    observability_report,
    spectral_radius,
)
from pendulum_lab.animation import animate_cartpole
from pendulum_lab.controllers import StateFeedback, bryson_weights, dlqr
from pendulum_lab.design import (
    DesignChoices,
    Strategy,
    balance_strategies,
    extended_kalman_filter,
    kalman_filter,
    kalman_gain,
    linear_model,
    swingup_controller,
)
from pendulum_lab.estimation import KalmanFilter
from pendulum_lab.lti import (
    closed_loop_matrix,
    compute_margins,
    observer_controller_lti,
    plant_lti,
)
from pendulum_lab.metrics import (
    SuccessCriteria,
    recovery_metrics,
    step_metrics,
    success,
    swingup_metrics,
)
from pendulum_lab.model import (
    C_MEAS,
    integrate,
    linearize,
    pendulum_energy,
    total_energy,
    wrap_angle,
)
from pendulum_lab.params import DEFAULT_SEED, CartPoleParams, LabConfig
from pendulum_lab.plotting import (
    plot_delay_sampling,
    plot_kalman,
    plot_lqr_tradeoff,
    plot_monte_carlo,
    plot_open_loop,
    plot_pole_map,
    plot_roa,
    plot_swingup,
    plot_time_responses,
)
from pendulum_lab.robustness import (
    LEVELS,
    SEVERE,
    BalanceScenario,
    monte_carlo,
    monte_carlo_swingup,
    roa_grid,
    run_strategy,
    sample_plants,
)
from pendulum_lab.simulation import simulate, step_reference
from pendulum_lab.types import FloatArray

#: Balance-comparison scenarios.
STEP_AMPLITUDE = 0.5
STEP_TIME = 0.5
STEP_DURATION = 10.0
RECOVERY_TILT = 0.2
RECOVERY_DURATION = 8.0


@dataclass(frozen=True)
class RunSettings:
    """Sizes of the stochastic studies.

    Attributes:
        mc_samples: Monte Carlo samples per uncertainty level.
        roa_points: Grid points per axis of the region-of-attraction map.
        delay_mc: Nonlinear runs per delay value in the delay study.
        seed: Master seed.
    """

    mc_samples: int = 500
    roa_points: int = 61
    delay_mc: int = 100
    seed: int = DEFAULT_SEED

    @classmethod
    def quick(cls) -> RunSettings:
        """Small sizes for smoke tests and CI."""
        return cls(mc_samples=40, roa_points=15, delay_mc=10)


def _f(x: Any) -> float | None:
    """JSON-safe float (``None`` for NaN/inf)."""
    v = float(x)
    return v if np.isfinite(v) else None


def _labels(strategies: list[Strategy]) -> dict[str, str]:
    return {s.key: s.label for s in strategies}


# --------------------------------------------------------------------------- analysis


def study_structure(config: LabConfig) -> dict[str, Any]:
    """Open-loop poles, controllability and observability of the linearisation."""
    A, B = linearize(config.params)
    Ad, Bd = c2d_zoh(A, B, config.Ts)
    eig = np.linalg.eigvals(A)
    outputs = {
        "x and theta": C_MEAS,
        "x only": C_MEAS[:1],
        "theta only": C_MEAS[1:],
    }
    ctrb = controllability_report(A, B, "force u")
    ctrb_d = controllability_report(Ad, Bd, "force u (discrete)")
    obs = [observability_report(A, C, lab) for lab, C in outputs.items()]
    W = finite_horizon_gramian(Ad, Bd, round(1.0 / config.Ts))
    w_eig = np.linalg.eigvalsh(W)
    return {
        "A": A.tolist(),
        "B": B[:, 0].tolist(),
        "open_loop_poles": sorted(float(np.real(e)) for e in eig),
        "unstable_pole": float(np.max(np.real(eig))),
        "rhp_zero_x": float(
            np.sqrt(
                float(config.params.m * config.params.g * config.params.l)
                / float(config.params.inertia_pivot)
            )
        ),
        "controllability": {
            "rank": ctrb.rank,
            "n": ctrb.n,
            "cond": ctrb.condition_number,
            "rank_discrete": ctrb_d.rank,
        },
        "observability": [
            {
                "output": r.label,
                "rank": r.rank,
                "cond": r.condition_number,
                "unobservable_modes": [float(np.real(m)) for m in r.weak_modes],
            }
            for r in obs
        ],
        "gramian_1s_eig_min": float(w_eig[0]),
        "gramian_1s_eig_max": float(w_eig[-1]),
    }


def study_linearization(config: LabConfig, img: Path | None) -> dict[str, Any]:
    """Validate the linear model against the nonlinear one, and the integrator."""
    p = config.params
    A_up, _ = linearize(p, upright=True)
    A_dn, _ = linearize(p, upright=False)
    dt = 0.001

    def run_nl(x0: FloatArray, T: float) -> FloatArray:
        n = round(T / dt)
        out = np.empty((n + 1, 4))
        x = x0.copy()
        out[0] = x
        for k in range(n):
            x = integrate(x, 0.0, p, dt, 1)
            out[k + 1] = x
        return out

    def run_lin(A: FloatArray, dx0: FloatArray, T: float) -> FloatArray:
        n = round(T / dt)
        Phi = expm(A * dt)
        out = np.empty((n + 1, 4))
        out[0] = dx0
        for k in range(n):
            out[k + 1] = Phi @ out[k]
        return out

    T_up, T_dn = 0.8, 5.0
    x_up = np.array([0.0, 0.0, 0.02, 0.0])
    nl_up, lin_up = run_nl(x_up, T_up), run_lin(A_up, x_up, T_up)
    small = np.array([0.0, 0.0, np.pi + 0.1, 0.0])
    large = np.array([0.0, 0.0, np.pi + 1.0, 0.0])
    nl_s, nl_l = run_nl(small, T_dn), run_nl(large, T_dn)
    lin_s = run_lin(A_dn, small - [0, 0, np.pi, 0], T_dn)
    lin_l = run_lin(A_dn, large - [0, 0, np.pi, 0], T_dn)

    # Integrator check: frictionless energy drift with the simulator's own step size.
    p0 = replace(p, b_c=0.0, b_p=0.0)
    x = np.array([0.0, 0.0, 2.5, 0.0])
    E0 = float(total_energy(x, p0))
    h = config.Ts / config.substeps
    for _ in range(round(10.0 / h)):
        x = integrate(x, 0.0, p0, h, 1)
    drift = abs(float(total_energy(x, p0)) - E0)

    if img is not None:
        t_up = np.asarray(np.arange(nl_up.shape[0]) * dt, dtype=np.float64)
        t_dn = np.asarray(np.arange(nl_s.shape[0]) * dt, dtype=np.float64)
        plot_open_loop(
            {
                "t_up": t_up,
                "nl_up": nl_up[:, 2],
                "lin_up": lin_up[:, 2],
                "t_dn": t_dn,
                "nl_dn_small": nl_s[:, 2] - np.pi,
                "lin_dn_small": lin_s[:, 2],
                "nl_dn_large": nl_l[:, 2] - np.pi,
                "lin_dn_large": lin_l[:, 2],
            },
            img / "open_loop.png",
        )

    def period(sig: FloatArray) -> float:
        s = np.sign(sig)
        idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
        return float(np.mean(np.diff(idx)) * dt) if idx.size > 1 else float("nan")

    return {
        "upright_theta_after_0_8s": {
            "nonlinear": float(nl_up[-1, 2]),
            "linear": float(lin_up[-1, 2]),
        },
        "hanging_period_small": {
            "nonlinear": period(nl_s[:, 2] - np.pi),
            "linear": period(lin_s[:, 2]),
        },
        "hanging_period_large": {
            "nonlinear": period(nl_l[:, 2] - np.pi),
            "linear": period(lin_l[:, 2]),
        },
        "rk4_energy_drift_10s_J": drift,
        "rk4_energy_initial_J": E0,
    }


# ------------------------------------------------------------------- balance comparison


def study_balance(config: LabConfig, img: Path | None, seed: int) -> dict[str, Any]:
    """Cart step and tilt recovery for every balancing strategy (nominal plant)."""
    strategies = balance_strategies(config)
    step_runs: dict[str, tuple[str, Any]] = {}
    rec_runs: dict[str, tuple[str, Any]] = {}
    rows = []
    for s in strategies:
        r_step = run_strategy(
            s,
            config,
            np.zeros(4),
            STEP_DURATION,
            reference=step_reference(STEP_AMPLITUDE, STEP_TIME),
            seed=seed,
        )
        r_rec = run_strategy(
            s, config, np.array([0.0, 0.0, RECOVERY_TILT, 0.0]), RECOVERY_DURATION, seed=seed + 1
        )
        sm = step_metrics(r_step, STEP_AMPLITUDE, STEP_TIME, u_max=config.actuator.u_max)
        rm = recovery_metrics(r_rec)
        step_runs[s.key] = (s.label, r_step)
        rec_runs[s.key] = (s.label, r_rec)
        rows.append(
            {
                "key": s.key,
                "label": s.label,
                "realistic": s.realistic,
                "step_settling_s": _f(sm.settling_time[0]),
                "step_overshoot_pct": _f(sm.overshoot[0]),
                "step_undershoot_pct": _f(sm.undershoot[0]),
                "step_peak_angle_deg": _f(np.degrees(sm.peak_angle[0])),
                "step_energy_N2s": _f(sm.energy[0]),
                "step_peak_force_N": _f(sm.peak_force[0]),
                "step_saturated_pct": _f(100 * sm.saturated_fraction[0]),
                "rec_settling_s": _f(rm.settling_time[0]),
                "rec_peak_cart_m": _f(rm.peak_cart_excursion[0]),
                "rec_energy_N2s": _f(rm.energy[0]),
                "rec_peak_force_N": _f(rm.peak_force[0]),
                "steady_rms_angle_mrad": _f(1e3 * rm.steady_rms_angle[0]),
                "steady_rms_force_N": _f(rm.steady_rms_force[0]),
            }
        )
    if img is not None:
        plot_time_responses(
            step_runs,
            img / "step_comparison.png",
            f"Cart set-point step of {STEP_AMPLITUDE} m (nominal plant, noise, 10 ms delay)",
            reference=True,
            u_max=config.actuator.u_max,
            t_max=6.0,
        )
        plot_time_responses(
            rec_runs,
            img / "recovery_comparison.png",
            f"Recovery from θ₀ = {RECOVERY_TILT} rad ({np.degrees(RECOVERY_TILT):.1f}°)",
            u_max=config.actuator.u_max,
            t_max=5.0,
        )
    return {"rows": rows}


def study_margins(config: LabConfig, img: Path | None) -> dict[str, Any]:
    """Classical margins of every linear loop, plus the closed-loop pole map."""
    strategies = balance_strategies(config)
    rows = []
    poles: dict[str, tuple[str, NDArray[Any]]] = {}
    lm = linear_model(config)
    poles["open_loop"] = ("open loop", np.linalg.eigvals(lm.Ad))
    for s in strategies:
        plant, ctrl = s.make_lti()
        m = compute_margins(plant, ctrl)
        rows.append(
            {
                "key": s.key,
                "label": s.label,
                "stable": m.stable,
                "spectral_radius": m.spectral_radius,
                "gain_low": m.gain_low,
                "gain_high": _f(m.gain_high),
                "gain_low_db": _f(m.gain_low_db),
                "gain_high_db": _f(m.gain_high_db),
                "phase_margin_deg": _f(m.phase_margin_deg),
                "crossover_rad_s": _f(m.crossover_rad_s),
                "delay_margin_ms": _f(1e3 * m.delay_margin_s),
                "delay_margin_samples": m.delay_margin_samples,
                "n_controller_states": ctrl.n_states,
            }
        )
        ev = np.linalg.eigvals(closed_loop_matrix(plant, ctrl))
        poles[s.key] = (s.label, ev[np.abs(ev) > 1e-9])
    if img is not None:
        plot_pole_map(poles, img / "pole_map.png")
    return {"rows": rows}


def study_lqr_tradeoff(
    config: LabConfig, img: Path | None, seed: int, choices: DesignChoices | None = None
) -> dict[str, Any]:
    """Sweep the input weight ``R -> rho R`` around the Bryson design."""
    choices = choices or DesignChoices()
    lm = linear_model(config)
    Q, R = bryson_weights(*choices.lqr_limits)
    L = kalman_gain(config, choices)
    rhos = np.geomspace(0.01, 100.0, 9)
    out: dict[str, list[float]] = {
        k: []
        for k in (
            "rho",
            "settling",
            "energy",
            "peak_force",
            "noise_force",
            "phase_margin",
            "gain_low_db",
            "slowest_pole",
        )
    }
    P = plant_lti(lm.Ad, lm.Bd, C_MEAS, config.actuator.delay_steps, config.Ts)
    for rho in rhos:
        des = dlqr(lm.Ad, lm.Bd, Q, R * rho)
        ctrl = StateFeedback(des.K, f"LQG rho={rho:g}")
        r = simulate(
            ctrl,
            np.zeros(4),
            config,
            STEP_DURATION,
            estimator=kalman_filter(config, choices),
            reference=step_reference(STEP_AMPLITUDE, STEP_TIME),
            seed=seed,
        )
        sm = step_metrics(r, STEP_AMPLITUDE, STEP_TIME)
        r0 = simulate(
            ctrl, np.zeros(4), config, 6.0, estimator=kalman_filter(config, choices), seed=seed + 1
        )
        m = compute_margins(
            P,
            observer_controller_lti(
                lm.Ad, lm.Bd, C_MEAS, L, des.K, config.actuator.delay_steps, config.Ts
            ),
        )
        s_poles = np.log(des.closed_loop_poles.astype(complex)) / config.Ts
        out["rho"].append(float(rho))
        out["settling"].append(float(sm.settling_time[0]))
        out["energy"].append(float(sm.energy[0]))
        out["peak_force"].append(float(sm.peak_force[0]))
        out["noise_force"].append(float(np.sqrt(np.mean(r0.u[-300:, 0] ** 2))))
        out["phase_margin"].append(float(m.phase_margin_deg))
        out["gain_low_db"].append(float(m.gain_low_db))
        out["slowest_pole"].append(float(np.max(np.real(s_poles))))
    if img is not None:
        plot_lqr_tradeoff({k: np.asarray(v) for k, v in out.items()}, 1.0, img / "lqr_tradeoff.png")
    des = dlqr(lm.Ad, lm.Bd, Q, R)
    return {
        "Q_diag": np.diag(Q).tolist(),
        "R": float(R[0, 0]),
        "limits": list(choices.lqr_limits),
        "K": des.K[0].tolist(),
        "closed_loop_poles_s": [
            [float(np.real(z)), float(np.imag(z))]
            for z in np.log(des.closed_loop_poles.astype(complex)) / config.Ts
        ],
        "sweep": {k: [_f(v) for v in vals] for k, vals in out.items()},
    }


# ------------------------------------------------------------------------- estimation


def study_kalman(config: LabConfig, img: Path | None, seed: int) -> dict[str, Any]:
    """Estimation accuracy and consistency of the KF (LQG loop) and KF vs EKF for swing-up."""
    lqg = next(s for s in balance_strategies(config) if s.key == "lqg")
    kf = kalman_filter(config)
    nis: list[float] = []

    class _Recorder(KalmanFilter):
        def update(self, y: FloatArray) -> FloatArray:
            xh = super().update(y)
            e = self.last_innovation[0]
            nis.append(float(e @ np.linalg.solve(self.last_S, e)))
            return xh

    rec = _Recorder(kf.Ad, kf.Bd, kf.Qd, kf.R)
    res = simulate(
        lqg.make_controller(), np.array([0.0, 0.0, 0.1, 0.0]), config, 6.0, estimator=rec, seed=seed
    )
    k0 = round(1.0 / config.Ts)
    err = res.x_hat[k0:, 0] - res.x[k0:-1, 0]
    rms = np.sqrt(np.mean(err**2, axis=0))
    # Naive alternative: finite differences of the raw measurements.
    naive = np.vstack([np.zeros((1, 2)), np.diff(res.y[:, 0], axis=0) / config.Ts])
    naive_vel_err = naive[k0:, 0] - res.x[k0:-1, 0, 1]
    naive_rate_err = naive[k0:, 1] - res.x[k0:-1, 0, 3]
    meas_err = res.y[k0:, 0] - res.x[k0:-1, 0][:, [0, 2]]

    # Swing-up with a linear KF (upright model) vs the EKF: same controller, same noise.
    sw_rows = []
    for name, est in (
        ("linear KF", kalman_filter(config)),
        ("EKF", extended_kalman_filter(config)),
    ):
        r = simulate(
            swingup_controller(config),
            np.array([0.0, 0.0, np.pi, 0.0]),
            config,
            12.0,
            estimator=est,
            seed=seed + 7,
        )
        ok = bool(success(r, SuccessCriteria(fall_check_from=12.0))[0])
        e = r.x_hat[:, 0] - r.x[:-1, 0]
        e[:, 2] = wrap_angle(e[:, 2])
        sw_rows.append(
            {
                "estimator": name,
                "success": ok,
                "rms_rate_error": float(np.sqrt(np.mean(e[:, 3] ** 2))),
                "rms_angle_error": float(np.sqrt(np.mean(e[:, 2] ** 2))),
            }
        )

    if img is not None:
        plot_kalman(res, naive[:, 1], naive[:, 0], img / "kalman_estimation.png")
    return {
        "rms_error": {"x": rms[0], "x_dot": rms[1], "theta": rms[2], "theta_dot": rms[3]},
        "rms_measurement_error": {
            "x": float(np.sqrt(np.mean(meas_err[:, 0] ** 2))),
            "theta": float(np.sqrt(np.mean(meas_err[:, 1] ** 2))),
        },
        "rms_finite_difference_error": {
            "x_dot": float(np.sqrt(np.mean(naive_vel_err**2))),
            "theta_dot": float(np.sqrt(np.mean(naive_rate_err**2))),
        },
        "mean_nis": float(np.mean(nis[k0:])),
        "nis_expected": 2.0,
        "steady_state_gain": kalman_gain(config).tolist(),
        "swingup_estimators": sw_rows,
    }


# --------------------------------------------------------------------------- swing-up


def study_swingup(
    config: LabConfig, img: Path | None, seed: int, gif: bool = True
) -> dict[str, Any]:
    """Nominal swing-up from the hanging position, with figure and GIF."""
    res = simulate(
        swingup_controller(config),
        np.array([0.0, 0.0, np.pi, 0.0]),
        config,
        12.0,
        estimator=extended_kalman_filter(config),
        seed=seed,
    )
    m = swingup_metrics(res)
    E = pendulum_energy(res.x[:-1, 0], config.params)
    if img is not None:
        plot_swingup(res, E, img / "swingup.png")
        if gif:
            animate_cartpole(
                res,
                img / "swingup.gif",
                pole_length=2 * float(config.params.l),
                track_limit=config.track_limit,
            )
    ok = bool(success(res, SuccessCriteria(fall_check_from=12.0))[0])
    return {
        "success": ok,
        "catch_time_s": _f(m.catch_time[0]),
        "settling_time_s": _f(m.settling_time[0]),
        "peak_cart_m": _f(m.peak_cart_excursion[0]),
        "energy_N2s": _f(m.energy[0]),
        "peak_force_N": _f(m.peak_force[0]),
        "mode_switches": int(m.mode_switches[0]),
        "gif_bytes": (img / "swingup.gif").stat().st_size if (img is not None and gif) else None,
    }


# ------------------------------------------------------------------------- robustness


def study_roa(config: LabConfig, img: Path | None, settings: RunSettings) -> dict[str, Any]:
    """Empirical region of attraction for every balancing strategy."""
    n = settings.roa_points
    thetas = np.linspace(-0.9, 0.9, n)
    omegas = np.linspace(-4.0, 4.0, n)
    cell = (thetas[1] - thetas[0]) * (omegas[1] - omegas[0])
    grids = {}
    rows = []
    j0 = int(np.argmin(np.abs(omegas)))
    for s in balance_strategies(config):
        g = roa_grid(s, config, thetas, omegas, 6.0, settings.seed)
        grids[s.key] = (s.label, g)
        on_axis = np.abs(thetas[g[j0]])
        rows.append(
            {
                "key": s.key,
                "label": s.label,
                "fraction": float(g.mean()),
                "area_rad2_s": float(g.sum() * cell),
                "max_theta0_at_rest_deg": float(np.degrees(on_axis.max())) if on_axis.size else 0.0,
            }
        )
    if img is not None:
        plot_roa(grids, thetas, omegas, img / "roa.png")
    return {
        "theta_range": [float(thetas[0]), float(thetas[-1])],
        "omega_range": [float(omegas[0]), float(omegas[-1])],
        "grid": n,
        "rows": rows,
    }


def study_monte_carlo(config: LabConfig, img: Path | None, settings: RunSettings) -> dict[str, Any]:
    """Monte Carlo success rates for balancing strategies and the swing-up."""
    strategies = balance_strategies(config)
    rows = []
    scenario = BalanceScenario()
    choices = DesignChoices()
    ablation_variants = {
        "no adaptation": {"adapt_gain": 0.0},
        "adaptation (default)": {},
        "adaptation + convergence gate": {"adapt_tolerance": 0.2},
    }
    ablation = []
    for i, unc in enumerate(LEVELS):
        results = monte_carlo(
            strategies, config, unc, settings.mc_samples, settings.seed + 10 + i, scenario
        )
        results.append(
            monte_carlo_swingup(config, unc, settings.mc_samples, settings.seed + 20 + i)
        )
        for name, over in ablation_variants.items():
            variant = replace(choices, swing={**choices.swing, **over})
            r_ab = monte_carlo_swingup(
                config, unc, settings.mc_samples, settings.seed + 20 + i, variant
            )
            ablation.append(
                {
                    "variant": name,
                    "uncertainty": unc.label,
                    "rate": r_ab.rate,
                    "ci95": list(r_ab.ci95),
                    "median_settling_s": _f(np.nanmedian(r_ab.settling_time))
                    if np.isfinite(r_ab.settling_time).any()
                    else None,
                }
            )
        for r in results:
            st = r.settling_time
            rows.append(
                {
                    "key": r.key,
                    "label": r.label,
                    "uncertainty": unc.label,
                    "n": r.n,
                    "rate": r.rate,
                    "ci95": list(r.ci95),
                    "median_settling_s": _f(np.nanmedian(st)) if np.isfinite(st).any() else None,
                    "p95_settling_s": _f(np.nanpercentile(st, 95))
                    if np.isfinite(st).any()
                    else None,
                    "median_energy_N2s": _f(np.median(r.energy)),
                    "settling_times": [_f(v) for v in st],
                }
            )
    if img is not None:
        plot_monte_carlo(rows, [u.label for u in LEVELS], img / "monte_carlo.png")
    return {
        "scenario": scenario.__dict__,
        "uncertainty": {u.label: u.__dict__ for u in LEVELS},
        "rows": [{k: v for k, v in r.items() if k != "settling_times"} for r in rows],
        "swingup_ablation": ablation,
    }


def study_delay(config: LabConfig, settings: RunSettings) -> dict[str, Any]:
    """Linear stability (nominal) and nonlinear Monte Carlo success versus actuator delay.

    The nonlinear part runs the full :class:`BalanceScenario` on plants drawn
    from the *severe* uncertainty level, so it measures how delay erodes the
    robustness budget, not just nominal stability.
    """
    delays = list(range(0, 16))
    rho: dict[str, list[float]] = {}
    succ: dict[str, list[float]] = {}
    rng = np.random.default_rng(settings.seed + 30)
    n = settings.delay_mc
    scenario = BalanceScenario()
    plants = sample_plants(config.params, SEVERE, n, rng)
    x0 = scenario.initial_states(n, rng)
    for d in delays:
        cfg_d = config.with_delay(d)
        for s in balance_strategies(cfg_d):
            plant, ctrl = s.make_lti()
            rho.setdefault(s.key, []).append(spectral_radius(closed_loop_matrix(plant, ctrl)))
            r = run_strategy(
                s,
                cfg_d,
                x0,
                scenario.duration,
                plant=plants,
                reference=scenario.reference,
                disturbance=scenario.disturbance,
                seed=settings.seed + d,
            )
            succ.setdefault(s.key, []).append(float(np.mean(success(r))))
    max_stable = {
        k: max((d for d, v in zip(delays, vals, strict=True) if v < 1.0), default=-1)
        for k, vals in rho.items()
    }
    return {
        "delay_steps": delays,
        "delay_ms": [1e3 * d * config.Ts for d in delays],
        "rho": rho,
        "success_rate": succ,
        "max_stable_delay_ms": {k: 1e3 * v * config.Ts for k, v in max_stable.items()},
    }


def study_sampling(config: LabConfig, choices: DesignChoices | None = None) -> dict[str, Any]:
    """Emulated continuous LQR vs direct discrete LQR as the sampling period grows."""
    choices = choices or DesignChoices()
    A, B = linearize(config.params)
    Q, R = bryson_weights(*choices.lqr_limits)
    Pc = solve_continuous_are(A, B, Q, R)
    Kc = np.linalg.solve(R, B.T @ Pc)
    Ts_list = [
        0.005,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.035,
        0.04,
        0.05,
        0.06,
        0.08,
        0.1,
        0.12,
        0.15,
    ]
    out: dict[str, list[float]] = {
        "Ts": [],
        "rho_emulated": [],
        "rho_discrete": [],
        "rho_discrete_comp": [],
    }
    for Ts in Ts_list:
        Ad, Bd = c2d_zoh(A, B, Ts)
        Az, Bz = augment_input_delay(Ad, Bd, 1)
        Kd = dlqr(Ad, Bd, Q, R * 1.0).K
        Kdc = dlqr(Az, Bz, np.pad(Q, ((0, 1), (0, 1))), R).K
        out["Ts"].append(Ts)
        out["rho_emulated"].append(spectral_radius(Az - Bz @ np.hstack([Kc, [[0.0]]])))
        out["rho_discrete"].append(spectral_radius(Az - Bz @ np.hstack([Kd, [[0.0]]])))
        out["rho_discrete_comp"].append(spectral_radius(Az - Bz @ Kdc))

    def max_ts(key: str) -> float:
        ok = [t for t, r in zip(out["Ts"], out[key], strict=True) if r < 1.0]
        return max(ok) if ok else float("nan")

    return {
        **out,
        "K_continuous": Kc[0].tolist(),
        "max_stable_Ts_s": {
            k: max_ts(k) for k in ("rho_emulated", "rho_discrete", "rho_discrete_comp")
        },
    }


# ------------------------------------------------------------------------------ driver


def run_all(
    img: Path | None,
    settings: RunSettings | None = None,
    config: LabConfig | None = None,
    gif: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every study and return the combined results dictionary."""
    settings = settings or RunSettings()
    config = config or LabConfig()
    seed = settings.seed
    results: dict[str, Any] = {"settings": settings.__dict__, "config": _config_dict(config)}
    steps: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("structure", lambda: study_structure(config)),
        ("linearization", lambda: study_linearization(config, img)),
        ("lqr", lambda: study_lqr_tradeoff(config, img, seed)),
        ("balance", lambda: study_balance(config, img, seed)),
        ("margins", lambda: study_margins(config, img)),
        ("kalman", lambda: study_kalman(config, img, seed)),
        ("swingup", lambda: study_swingup(config, img, seed, gif=gif)),
        ("roa", lambda: study_roa(config, img, settings)),
        ("monte_carlo", lambda: study_monte_carlo(config, img, settings)),
        ("delay", lambda: study_delay(config, settings)),
        ("sampling", lambda: study_sampling(config)),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        results[name] = fn()
        log(f"[{name}] done in {time.perf_counter() - t0:.1f} s")
    if img is not None:
        labels = _labels(balance_strategies(config))
        plot_delay_sampling(
            results["delay"], results["sampling"], labels, img / "delay_sampling.png"
        )
    return results


def _config_dict(config: LabConfig) -> dict[str, Any]:
    p: CartPoleParams = config.params
    return {
        "Ts": config.Ts,
        "substeps": config.substeps,
        "track_limit": config.track_limit,
        "force_disturbance_std": config.force_disturbance_std,
        "sigma_x": config.sensor.sigma_x,
        "sigma_theta": config.sensor.sigma_theta,
        "u_max": config.actuator.u_max,
        "delay_steps": config.actuator.delay_steps,
        "params": {k: float(getattr(p, k)) for k in ("M", "m", "l", "J", "b_c", "b_p", "g")},
    }

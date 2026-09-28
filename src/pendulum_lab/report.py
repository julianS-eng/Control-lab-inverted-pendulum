"""Render the results dictionary as Markdown tables and splice them into the README.

README sections delimited by ``<!-- BEGIN GENERATED: name -->`` and
``<!-- END GENERATED: name -->`` are replaced verbatim, so every number shown in
the README is traceable to ``docs/results/results.json``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        return f"{v:.{nd}f}"
    return str(v)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(rate: float, ci: Sequence[float]) -> str:
    return f"{100 * rate:.1f} % [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def render_tables(res: dict[str, Any]) -> dict[str, str]:
    """Build every generated table from a results dictionary."""
    t: dict[str, str] = {}

    st = res["structure"]
    obs_rows = [
        [
            o["output"],
            f"{o['rank']} / 4",
            f"{o['cond']:.3g}",
            ", ".join(f"{m:.3g}" for m in o["unobservable_modes"]) or "none",
        ]
        for o in st["observability"]
    ]
    c = st["controllability"]
    t["structure"] = (
        _table(
            ["Quantity", "Value"],
            [
                ["Open-loop poles [rad/s]", ", ".join(f"{p:.3f}" for p in st["open_loop_poles"])],
                ["Unstable pole (upright)", f"+{st['unstable_pole']:.3f} rad/s"],
                ["RHP zero of x/u", f"+{st['rhp_zero_x']:.3f} rad/s"],
                [
                    "Controllability rank (continuous / discrete)",
                    f"{c['rank']} / {c['rank_discrete']} (of 4)",
                ],
                ["Controllability matrix condition number", f"{c['cond']:.3g}"],
                [
                    "1 s Gramian eigenvalues (min / max)",
                    f"{st['gramian_1s_eig_min']:.3g} / {st['gramian_1s_eig_max']:.3g}",
                ],
            ],
        )
        + "\n\n"
        + _table(
            ["Measured output", "Observability rank", "Condition number", "Unobservable modes"],
            obs_rows,
            "lrrr",
        )
    )

    lin = res["linearization"]
    t["linearization"] = _table(
        ["Check", "Nonlinear", "Linear"],
        [
            [
                "θ after 0.8 s from θ₀ = 0.02 rad (upright) [rad]",
                _fmt(lin["upright_theta_after_0_8s"]["nonlinear"], 4),
                _fmt(lin["upright_theta_after_0_8s"]["linear"], 4),
            ],
            [
                "Hanging period, 0.1 rad swing [s]",
                _fmt(lin["hanging_period_small"]["nonlinear"], 3),
                _fmt(lin["hanging_period_small"]["linear"], 3),
            ],
            [
                "Hanging period, 1.0 rad swing [s]",
                _fmt(lin["hanging_period_large"]["nonlinear"], 3),
                _fmt(lin["hanging_period_large"]["linear"], 3),
            ],
            [
                "RK4 energy drift, 10 s frictionless [J]",
                f"{lin['rk4_energy_drift_10s_J']:.2e}",
                "—",
            ],
        ],
        "lrr",
    )

    lqr = res["lqr"]
    t["lqr"] = _table(
        ["Item", "Value"],
        [
            ["Bryson limits (x, ẋ, θ, θ̇, u)", ", ".join(_fmt(v, 2) for v in lqr["limits"])],
            ["Q = diag(...)", ", ".join(f"{v:.3g}" for v in lqr["Q_diag"])],
            ["R", f"{lqr['R']:.4g}"],
            ["K (u = −K x)", ", ".join(f"{v:.2f}" for v in lqr["K"])],
            [
                "Equivalent s-plane poles",
                ", ".join(
                    f"{re_:.2f}{im:+.2f}j" if abs(im) > 1e-6 else f"{re_:.2f}"
                    for re_, im in lqr["closed_loop_poles_s"]
                ),
            ],
        ],
    )
    sw = lqr["sweep"]
    t["lqr_sweep"] = _table(
        [
            "ρ (R → ρR)",
            "Settling [s]",
            "∫u² [N²s]",
            "Peak \\|u\\| [N]",
            "Noise RMS u [N]",
            "PM [deg]",
            "Lower GM [dB]",
        ],
        [
            [
                f"{sw['rho'][i]:g}",
                _fmt(sw["settling"][i]),
                _fmt(sw["energy"][i], 1),
                _fmt(sw["peak_force"][i], 1),
                _fmt(sw["noise_force"][i], 3),
                _fmt(sw["phase_margin"][i], 1),
                _fmt(sw["gain_low_db"][i], 1),
            ]
            for i in range(len(sw["rho"]))
        ],
    )

    t["balance"] = _table(
        [
            "Controller",
            "Step settling [s]",
            "Overshoot [%]",
            "Undershoot [%]",
            "∫u² step [N²s]",
            "Peak \\|u\\| [N]",
            "Recovery settling [s]",
            "Peak cart [m]",
            "RMS θ at rest [mrad]",
            "RMS u at rest [N]",
        ],
        [
            [
                r["label"],
                _fmt(r["step_settling_s"]),
                _fmt(r["step_overshoot_pct"], 1),
                _fmt(r["step_undershoot_pct"], 1),
                _fmt(r["step_energy_N2s"], 1),
                _fmt(r["step_peak_force_N"], 1),
                _fmt(r["rec_settling_s"]),
                _fmt(r["rec_peak_cart_m"]),
                _fmt(r["steady_rms_angle_mrad"]),
                _fmt(r["steady_rms_force_N"], 3),
            ]
            for r in res["balance"]["rows"]
        ],
    )

    t["margins"] = _table(
        [
            "Loop (broken at the actuator)",
            "Gain margin ↓ [dB]",
            "Gain margin ↑ [dB]",
            "Phase margin [deg]",
            "Crossover [rad/s]",
            "Delay margin [ms]",
            "Extra samples tolerated",
        ],
        [
            [
                r["label"],
                _fmt(r["gain_low_db"], 1),
                _fmt(r["gain_high_db"], 1),
                _fmt(r["phase_margin_deg"], 1),
                _fmt(r["crossover_rad_s"], 1),
                _fmt(r["delay_margin_ms"], 1),
                str(r["delay_margin_samples"]),
            ]
            for r in res["margins"]["rows"]
        ],
    )

    k = res["kalman"]
    t["kalman"] = (
        _table(
            [
                "State",
                "Raw measurement RMS error",
                "Finite-difference RMS error",
                "Kalman RMS error",
            ],
            [
                [
                    "x [mm]",
                    _fmt(1e3 * k["rms_measurement_error"]["x"]),
                    "—",
                    _fmt(1e3 * k["rms_error"]["x"]),
                ],
                [
                    "ẋ [mm/s]",
                    "—",
                    _fmt(1e3 * k["rms_finite_difference_error"]["x_dot"], 1),
                    _fmt(1e3 * k["rms_error"]["x_dot"], 1),
                ],
                [
                    "θ [mrad]",
                    _fmt(1e3 * k["rms_measurement_error"]["theta"]),
                    "—",
                    _fmt(1e3 * k["rms_error"]["theta"]),
                ],
                [
                    "θ̇ [mrad/s]",
                    "—",
                    _fmt(1e3 * k["rms_finite_difference_error"]["theta_dot"], 1),
                    _fmt(1e3 * k["rms_error"]["theta_dot"], 1),
                ],
            ],
        )
        + f"\n\nMean normalised innovation squared (NIS): **{k['mean_nis']:.2f}** (expected 2.00 "
        "for a perfectly tuned filter with two measurements)."
    )
    t["kalman_swingup"] = _table(
        [
            "Estimator in the swing-up loop",
            "Swing-up succeeded",
            "RMS θ error [rad]",
            "RMS θ̇ error [rad/s]",
        ],
        [
            [
                r["estimator"],
                _fmt(r["success"]),
                _fmt(r["rms_angle_error"], 3),
                _fmt(r["rms_rate_error"], 3),
            ]
            for r in k["swingup_estimators"]
        ],
        "lrrr",
    )

    s = res["swingup"]
    t["swingup"] = _table(
        ["Metric", "Value"],
        [
            ["Hand-over to LQR (catch time)", f"{_fmt(s['catch_time_s'])} s"],
            [
                "Settled upright (\\|θ\\| < 0.05 rad, \\|x\\| < 5 cm)",
                f"{_fmt(s['settling_time_s'])} s",
            ],
            ["Peak cart excursion", f"{_fmt(s['peak_cart_m'])} m"],
            ["Peak force", f"{_fmt(s['peak_force_N'], 1)} N"],
            ["Control energy ∫u² dt", f"{_fmt(s['energy_N2s'], 1)} N²s"],
            ["Mode switches", str(s["mode_switches"])],
            ["GIF size", f"{s['gif_bytes'] / 1e6:.2f} MB" if s["gif_bytes"] else "—"],
        ],
    )

    roa = res["roa"]
    t["roa"] = _table(
        [
            "Controller",
            "Recovered fraction of grid",
            "Area [rad²/s]",
            "Largest recoverable θ₀ at rest [deg]",
        ],
        [
            [
                r["label"],
                f"{100 * r['fraction']:.1f} %",
                _fmt(r["area_rad2_s"]),
                _fmt(r["max_theta0_at_rest_deg"], 1),
            ]
            for r in roa["rows"]
        ],
    )

    mc = res["monte_carlo"]
    levels = list(mc["uncertainty"])
    labels = list(dict.fromkeys(r["label"] for r in mc["rows"]))
    rows = []
    for lab in labels:
        by = {r["uncertainty"]: r for r in mc["rows"] if r["label"] == lab}
        first = by[levels[0]]
        rows.append(
            [
                lab,
                *[_pct(by[lv]["rate"], by[lv]["ci95"]) for lv in levels],
                _fmt(first["median_settling_s"]),
                _fmt(first["median_energy_N2s"], 1),
            ]
        )
    n = mc["rows"][0]["n"]
    t["monte_carlo"] = (
        _table(
            [
                "Controller",
                *[f"Success, {lv}" for lv in levels],
                f"Median settling, {levels[0]} [s]",
                f"Median ∫u², {levels[0]} [N²s]",
            ],
            rows,
        )
        + f"\n\n*n = {n} sampled plants per level (common random numbers across controllers); "
        "brackets are Wilson 95 % confidence intervals.*"
    )
    unc_rows = [
        [
            lv,
            f"±{100 * u['cart_mass']:.0f} %",
            f"±{100 * u['pole_mass']:.0f} %",
            f"±{100 * u['length']:.0f} %",
            f"×/÷ {u['friction_factor']:g}",
        ]
        for lv, u in mc["uncertainty"].items()
    ]
    t["uncertainty"] = _table(
        ["Level", "Cart mass M", "Pole mass m", "Length l", "Friction b_c, b_p"], unc_rows
    )
    ab = mc["swingup_ablation"]
    variants = list(dict.fromkeys(r["variant"] for r in ab))
    t["swingup_ablation"] = _table(
        ["Swing-up variant", *[f"Success, {lv}" for lv in levels]],
        [
            [
                v,
                *[
                    _pct(r["rate"], r["ci95"])
                    for lv in levels
                    for r in ab
                    if r["variant"] == v and r["uncertainty"] == lv
                ],
            ]
            for v in variants
        ],
    )

    d = res["delay"]
    names = {r["key"]: r["label"] for r in res["balance"]["rows"]}
    d_max = max(d["delay_ms"])
    cols = [3, 5]
    idx = [d["delay_steps"].index(c) for c in cols if c in d["delay_steps"]]

    def max_delay(key: str) -> str:
        v = d["max_stable_delay_ms"][key]
        return f"≥ {v:.0f} (max tested)" if v >= d_max else f"{v:.0f}"

    t["delay"] = _table(
        [
            "Controller",
            "Largest stable delay, nominal linear loop [ms]",
            *[f"MC success at {d['delay_ms'][i]:.0f} ms (severe)" for i in idx],
        ],
        [
            [
                names.get(key, key),
                max_delay(key),
                *[f"{100 * d['success_rate'][key][i]:.0f} %" for i in idx],
            ]
            for key in d["rho"]
        ],
    )
    smp = res["sampling"]
    ms = smp["max_stable_Ts_s"]
    ts_max = max(smp["Ts"])

    def max_ts(key: str) -> str:
        v = ms[key]
        return f"≥ {1e3 * v:.0f} (max tested)" if v >= ts_max else f"{1e3 * v:.0f}"

    t["sampling"] = (
        _table(
            ["Design (with one sample of computation delay)", "Largest stable Ts [ms]"],
            [
                ["Continuous LQR, gains emulated", max_ts("rho_emulated")],
                ["Discrete LQR (DARE on the ZOH model)", max_ts("rho_discrete")],
                ["Discrete LQR + delay compensation", max_ts("rho_discrete_comp")],
            ],
        )
        + "\n\n*Tested periods: "
        + ", ".join(f"{1e3 * v:g}" for v in smp["Ts"])
        + " ms.*"
    )

    cfg = res["config"]
    p = cfg["params"]
    t["config"] = _table(
        ["Parameter", "Value"],
        [
            ["Cart mass M", f"{p['M']} kg"],
            ["Pole mass m", f"{p['m']} kg"],
            ["Pivot to centre of mass l", f"{p['l']} m (rod length {2 * p['l']:.1f} m)"],
            ["Pole inertia about CoM J", f"{p['J']:.4f} kg·m²"],
            ["Cart / pivot viscous friction", f"{p['b_c']} N·s/m / {p['b_p']} N·m·s/rad"],
            ["Sampling period Ts", f"{1e3 * cfg['Ts']:.0f} ms ({cfg['substeps']} RK4 substeps)"],
            ["Actuator", f"±{cfg['u_max']} N, {cfg['delay_steps']} sample(s) of delay"],
            [
                "Sensor noise (1σ)",
                f"x: {1e3 * cfg['sigma_x']:.1f} mm, θ: {1e3 * cfg['sigma_theta']:.1f} mrad",
            ],
            ["Random force disturbance (1σ)", f"{cfg['force_disturbance_std']} N"],
            ["Rail half-length", f"{cfg['track_limit']} m"],
        ],
    )
    return t


_MARK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n)(?P<body>.*?)"
    r"(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def inject_readme(path: Path, tables: dict[str, str]) -> list[str]:
    """Replace every generated section of ``path`` whose name is in ``tables``.

    Returns:
        Names of the sections that were updated.
    """
    text = path.read_text(encoding="utf-8")
    updated: list[str] = []

    def sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in tables:
            return m.group(0)
        updated.append(name)
        return f"{m.group(1)}{tables[name]}\n{m.group(4)}"

    new = _MARK.sub(sub, text)
    path.write_text(new, encoding="utf-8")
    return updated

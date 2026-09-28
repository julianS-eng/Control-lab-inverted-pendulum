"""Matplotlib figures for the lab report.

Style rules (applied consistently): one colour per controller that never
changes between figures, 2 px lines, recessive grid, one y-axis per panel,
legends whenever two or more series share a panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pendulum_lab.model import wrap_angle
from pendulum_lab.simulation import SimResult
from pendulum_lab.types import BoolArray, FloatArray

#: Fixed colour per strategy key (validated categorical palette, fixed order).
COLORS: dict[str, str] = {
    "pid": "#2a78d6",
    "pole_placement": "#eb6834",
    "lqr_ideal": "#52514e",
    "lqg": "#1baf7a",
    "lqg_delay": "#eda100",
    "swingup": "#4a3aa7",
    "reference": "#8a8984",
    "measured": "#b8b7b0",
}
LINESTYLES: dict[str, str] = {"lqr_ideal": "--"}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def apply_style() -> None:
    """Set global Matplotlib defaults for all lab figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 110,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _style(key: str) -> dict[str, Any]:
    return {"color": COLORS.get(key, INK), "linestyle": LINESTYLES.get(key, "-")}


def plot_open_loop(data: Mapping[str, FloatArray], path: Path) -> Path:
    """Nonlinear vs linearised free response, upright and hanging."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.plot(data["t_up"], data["nl_up"], color=COLORS["pid"], label="nonlinear")
    ax.plot(data["t_up"], data["lin_up"], color=COLORS["pole_placement"], ls="--", label="linear")
    ax.set(title="Upright: unstable divergence", xlabel="time [s]", ylabel="θ [rad]")
    ax.legend()
    ax = axes[1]
    ax.plot(data["t_dn"], data["nl_dn_small"], color=COLORS["pid"], label="nonlinear, 0.1 rad")
    ax.plot(
        data["t_dn"],
        data["lin_dn_small"],
        color=COLORS["pole_placement"],
        ls="--",
        label="linear, 0.1 rad",
    )
    ax.plot(data["t_dn"], data["nl_dn_large"], color=COLORS["lqg"], label="nonlinear, 1.0 rad")
    ax.plot(
        data["t_dn"],
        data["lin_dn_large"],
        color=COLORS["lqg_delay"],
        ls="--",
        label="linear, 1.0 rad",
    )
    ax.set(title="Hanging: oscillation about θ = π", xlabel="time [s]", ylabel="θ − π [rad]")
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


def plot_time_responses(
    runs: Mapping[str, tuple[str, SimResult]],
    path: Path,
    title: str,
    reference: bool = False,
    u_max: float | None = None,
    t_max: float | None = None,
) -> Path:
    """Stack cart position, pole angle and force for several strategies."""
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    for key, (label, res) in runs.items():
        t = res.t
        st = _style(key)
        axes[0].plot(t, res.x[:, 0, 0], label=label, **st)
        axes[1].plot(t, np.degrees(wrap_angle(res.x[:, 0, 2])), **st)
        axes[2].step(t[:-1], res.u[:, 0], where="post", lw=1.4, **st)
    if reference:
        first = next(iter(runs.values()))[1]
        axes[0].plot(
            first.t[:-1],
            first.ref[:, 0],
            color=COLORS["reference"],
            lw=1.2,
            ls=":",
            label="reference",
        )
    if u_max is not None:
        for s in (-1, 1):
            axes[2].axhline(s * u_max, color=COLORS["reference"], lw=1.0, ls=":")
    axes[0].set(ylabel="cart x [m]", title=title)
    axes[1].set(ylabel="pole θ [deg]")
    axes[2].set(ylabel="force u [N]", xlabel="time [s]")
    axes[0].legend(ncol=3, loc="lower right")
    if t_max is not None:
        axes[2].set_xlim(0, t_max)
    fig.tight_layout()
    return _save(fig, path)


def plot_lqr_tradeoff(sweep: Mapping[str, FloatArray], chosen: float, path: Path) -> Path:
    """Effect of the input weight ``rho`` on speed, effort and robustness."""
    apply_style()
    rho = np.asarray(sweep["rho"])
    ci = int(np.argmin(np.abs(np.log(rho / chosen))))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    ax = axes[0]
    ax.plot(sweep["energy"], sweep["settling"], "o-", color=COLORS["lqg"], ms=6)
    ax.plot(sweep["energy"][ci], sweep["settling"][ci], "o", color=INK, ms=10, mfc="none", mew=2)
    ax.annotate(
        "chosen ρ = 1",
        (sweep["energy"][ci], sweep["settling"][ci]),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color=INK,
    )
    ax.set(
        xscale="log",
        xlabel="∫u² dt, 0.5 m step [N²·s]",
        ylabel="settling time [s]",
        title="Speed vs effort (Pareto front)",
    )
    ax = axes[1]
    ax.plot(
        rho,
        sweep["peak_force"],
        "o-",
        color=COLORS["pole_placement"],
        ms=6,
        label="peak force, 0.5 m step",
    )
    ax.plot(
        rho,
        sweep["noise_force"],
        "o-",
        color=COLORS["pid"],
        ms=6,
        label="RMS force at rest (noise)",
    )
    ax.axvline(chosen, color=INK_2, lw=1, ls=":")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="input-weight scale ρ (R → ρR)",
        ylabel="force [N]",
        title="Actuator usage",
    )
    ax.legend(loc="upper right")
    ax = axes[2]
    ax.plot(rho, sweep["phase_margin"], "o-", color=COLORS["lqg"], ms=6)
    ax.axvline(chosen, color=INK_2, lw=1, ls=":")
    ax.set(
        xscale="log",
        xlabel="input-weight scale ρ",
        ylabel="phase margin [deg]",
        title="Robustness (LQG loop)",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_kalman(res: SimResult, naive_rate: FloatArray, naive_vel: FloatArray, path: Path) -> Path:
    """True, measured and estimated states for the LQG loop."""
    apply_style()
    t = res.t[:-1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8), sharex=True)
    specs = [
        (axes[0, 0], 0, "cart x [m]", True, None),
        (axes[0, 1], 2, "pole θ [rad]", True, None),
        (axes[1, 0], 1, "cart velocity [m/s]", False, naive_vel),
        (axes[1, 1], 3, "pole rate [rad/s]", False, naive_rate),
    ]
    for ax, i, lab, measured, naive in specs:
        if measured:
            ax.plot(
                t,
                res.y[:, 0, 0 if i == 0 else 1],
                color=COLORS["measured"],
                lw=1.0,
                label="measurement",
            )
        if naive is not None:
            ax.plot(t, naive, color=COLORS["measured"], lw=1.0, label="finite difference")
        ax.plot(res.t, res.x[:, 0, i], color=INK, lw=2.0, label="true")
        ax.plot(
            t, res.x_hat[:, 0, i], color=COLORS["lqg"], lw=1.6, ls="--", label="Kalman estimate"
        )
        ax.set(ylabel=lab)
        ax.legend(loc="upper right")
    axes[1, 0].set(xlabel="time [s]")
    axes[1, 1].set(xlabel="time [s]")
    fig.suptitle(
        "Kalman filter: reconstructing velocities from noisy x and θ",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_swingup(res: SimResult, energy: FloatArray, path: Path) -> Path:
    """Swing-up trajectory with mode shading."""
    apply_style()
    t = res.t
    mode = np.nan_to_num(res.mode[:, 0])
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, np.degrees(res.x[:, 0, 2]), color=COLORS["swingup"])
    axes[0].set(ylabel="θ [deg] (unwrapped)", title="Energy swing-up with hand-over to LQR")
    axes[1].plot(t, res.x[:, 0, 0], color=COLORS["swingup"])
    axes[1].set(ylabel="cart x [m]")
    axes[2].step(t[:-1], res.u[:, 0], where="post", color=COLORS["swingup"], lw=1.2)
    axes[2].set(ylabel="force u [N]")
    axes[3].plot(t[:-1], energy, color=COLORS["swingup"])
    axes[3].axhline(0.0, color=COLORS["reference"], ls=":", lw=1.2)
    axes[3].set(ylabel="pole energy E [J]", xlabel="time [s]")
    bal = mode > 0.5
    if bal.any():
        t0 = t[int(np.argmax(bal))]
        for ax in axes:
            ax.axvspan(t0, t[-1], color=COLORS["lqg"], alpha=0.08, lw=0)
        axes[0].text(
            t0 + 0.1,
            axes[0].get_ylim()[1] * 0.9,
            "LQR balancing",
            color=INK_2,
            fontsize=9,
            va="top",
        )
    fig.tight_layout()
    return _save(fig, path)


def plot_roa(
    grids: Mapping[str, tuple[str, BoolArray]], thetas: FloatArray, omegas: FloatArray, path: Path
) -> Path:
    """Small multiples of the empirical region of attraction."""
    apply_style()
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), sharey=True)
    axes_list = np.atleast_1d(axes)
    extent = (float(thetas[0]), float(thetas[-1]), float(omegas[0]), float(omegas[-1]))
    for ax, (key, (label, grid)) in zip(axes_list, grids.items(), strict=True):
        cmap = ListedColormap(["#f0efec", COLORS.get(key, INK)])
        ax.imshow(
            grid.astype(float),
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(f"{label}\n{100 * grid.mean():.1f}% of grid", fontsize=9.5)
        ax.set_xlabel("θ₀ [rad]")
        ax.grid(False)
    axes_list[0].set_ylabel("θ̇₀ [rad/s]")
    fig.suptitle(
        "Empirical region of attraction (coloured = recovered)", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_monte_carlo(rows: Sequence[Mapping[str, Any]], levels: Sequence[str], path: Path) -> Path:
    """Success rate with Wilson 95 % CI per strategy and uncertainty level."""
    apply_style()
    labels = list(dict.fromkeys(r["label"] for r in rows))
    x = np.arange(len(labels))
    width = 0.8 / len(levels)
    level_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for j, lev in enumerate(levels):
        vals, lo, hi = [], [], []
        for lab in labels:
            r = next(r for r in rows if r["label"] == lab and r["uncertainty"] == lev)
            vals.append(100 * r["rate"])
            lo.append(max(0.0, 100 * (r["rate"] - r["ci95"][0])))
            hi.append(max(0.0, 100 * (r["ci95"][1] - r["rate"])))
        ax.bar(
            x + (j - (len(levels) - 1) / 2) * width,
            vals,
            width * 0.92,
            color=level_colors[j],
            label=f"{lev} spread",
            yerr=[lo, hi],
            error_kw={"ecolor": INK_2, "lw": 1, "capsize": 3},
        )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="success rate [%]", ylim=(0, 105), title="Monte Carlo success rate (95 % CI)")
    ax.legend(loc="lower left")
    ax = axes[1]
    data, names, cols = [], [], []
    for r in rows:
        if r["uncertainty"] != levels[0]:
            continue
        st = np.asarray(r["settling_times"], dtype=float)
        data.append(st[np.isfinite(st)])
        names.append(r["label"])
        cols.append(COLORS.get(r["key"], INK))
    bp = ax.boxplot(
        data,
        orientation="horizontal",
        patch_artist=True,
        widths=0.6,
        medianprops={"color": INK, "lw": 1.5},
        flierprops={"ms": 3},
    )
    for patch, c in zip(bp["boxes"], cols, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_yticks(np.arange(1, len(names) + 1), names)
    ax.set(
        xlabel="settling time of the whole scenario [s]",
        title=f"Settling time ({levels[0]} spread)",
    )
    fig.tight_layout()
    return _save(fig, path)


def plot_delay_sampling(
    delay: Mapping[str, Any], sampling: Mapping[str, Any], labels: Mapping[str, str], path: Path
) -> Path:
    """Delay robustness (linear and nonlinear) and emulation vs direct discrete design."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    d_ms = np.asarray(delay["delay_ms"])
    order = [k for k in delay["rho"] if k != "lqr_ideal"] + ["lqr_ideal"]
    ax = axes[0]
    for key in order:
        ax.plot(
            d_ms,
            delay["rho"][key],
            marker="o",
            ms=4,
            label=labels[key],
            zorder=5 if key == "lqr_ideal" else 2,
            **_style(key),
        )
    ax.axhline(1.0, color=INK, lw=1)
    ax.set(
        xlabel="actuator delay [ms]",
        ylabel="closed-loop spectral radius",
        title="Linear stability vs delay (< 1 is stable)",
        ylim=(0.95, 1.1),
    )
    ax.legend(loc="upper left")
    ax = axes[1]
    for key in order:
        ax.plot(
            d_ms,
            100 * np.asarray(delay["success_rate"][key]),
            marker="o",
            ms=4,
            zorder=5 if key == "lqr_ideal" else 2,
            **_style(key),
        )
    ax.set(
        xlabel="actuator delay [ms]",
        ylabel="success rate [%]",
        ylim=(-3, 103),
        title="Nonlinear Monte Carlo vs delay (severe spread)",
    )
    ax = axes[2]
    Ts_ms = 1e3 * np.asarray(sampling["Ts"])
    ax.plot(
        Ts_ms,
        sampling["rho_emulated"],
        "o-",
        ms=4,
        color=COLORS["pole_placement"],
        label="continuous LQR, emulated",
    )
    ax.plot(
        Ts_ms,
        sampling["rho_discrete"],
        "o-",
        ms=4,
        color=COLORS["lqg"],
        label="discrete LQR (DARE)",
    )
    ax.plot(
        Ts_ms,
        sampling["rho_discrete_comp"],
        "o-",
        ms=4,
        color=COLORS["lqg_delay"],
        label="discrete LQR + delay comp.",
    )
    ax.axhline(1.0, color=INK, lw=1)
    ax.set(
        xlabel="sampling period Ts [ms] (1-sample compute delay)",
        ylabel="closed-loop spectral radius",
        title="Emulation vs direct discrete design",
        ylim=(0.8, 1.2),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def plot_pole_map(poles: Mapping[str, tuple[str, NDArray[Any]]], path: Path) -> Path:
    """Discrete closed-loop poles (z-plane) of each linear loop."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, zoom in zip(axes, (False, True), strict=True):
        ax.plot(np.cos(th), np.sin(th), color=INK_2, lw=1)
        for key, (label, z) in poles.items():
            marker = "x" if key == "open_loop" else "o"
            color = INK if key == "open_loop" else COLORS.get(key, INK)
            ax.plot(
                np.real(z),
                np.imag(z),
                marker,
                ls="none",
                color=color,
                ms=7,
                mfc="none" if marker == "o" else color,
                mew=1.8,
                label=label,
            )
        ax.set_aspect("equal")
        ax.set(xlabel="Re z", ylabel="Im z")
        if zoom:
            ax.set(xlim=(0.85, 1.07), ylim=(-0.11, 0.11), title="Zoom near z = 1")
        else:
            ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), title="Closed-loop poles (z-plane)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)

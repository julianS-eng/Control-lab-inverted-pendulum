"""GIF animation of a cart-pole trajectory (Pillow writer, size-bounded)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle

from pendulum_lab.plotting import COLORS, INK, INK_2, apply_style
from pendulum_lab.simulation import SimResult

#: README asset budget for animations.
MAX_GIF_BYTES = 5 * 1024 * 1024


def animate_cartpole(
    res: SimResult,
    path: Path,
    pole_length: float,
    track_limit: float,
    fps: int = 20,
    dpi: int = 72,
    index: int = 0,
    title: str = "Energy swing-up → LQR",
) -> Path:
    """Render loop ``index`` of ``res`` to an animated GIF.

    Frames are subsampled from the simulation to ``fps`` frames per second of
    simulated time, so the GIF plays in real time.

    Args:
        res: Simulation result.
        path: Output ``.gif`` path.
        pole_length: Drawn pole length (pivot to tip) [m].
        track_limit: Rail half-length drawn as end stops [m].
        fps: Frames per second.
        dpi: Output resolution.
        index: Which loop of the batch to draw.
        title: Figure title.

    Returns:
        The written path.

    Raises:
        RuntimeError: If the resulting file exceeds :data:`MAX_GIF_BYTES`.
    """
    apply_style()
    stride = max(1, round(1.0 / (fps * res.Ts)))
    frames = np.arange(0, res.x.shape[0], stride)
    x = res.x[:, index, 0]
    th = res.x[:, index, 2]
    mode = np.nan_to_num(res.mode[:, index])

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_xlim(-track_limit - 0.15, track_limit + 0.15)
    ax.set_ylim(-pole_length - 0.15, pole_length + 0.2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_yticks([])
    ax.set_xlabel("cart position [m]")
    ax.set_title(title)
    ax.plot([-track_limit, track_limit], [0, 0], color=INK_2, lw=2)
    for s in (-1, 1):
        ax.plot([s * track_limit] * 2, [-0.05, 0.05], color=INK, lw=3)
    cart_w, cart_h = 0.16, 0.08
    cart = Rectangle((0, -cart_h / 2), cart_w, cart_h, color=COLORS["pid"], zorder=3)
    ax.add_patch(cart)
    (pole,) = ax.plot([], [], color=COLORS["swingup"], lw=4, solid_capstyle="round", zorder=4)
    (tip,) = ax.plot([], [], "o", color=COLORS["swingup"], ms=7, zorder=5)
    (trail,) = ax.plot([], [], color=COLORS["swingup"], lw=1, alpha=0.35)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=9, color=INK)

    def draw(k: int) -> tuple[Artist, ...]:
        px, py = x[k], 0.0
        tx, ty = px + pole_length * np.sin(th[k]), pole_length * np.cos(th[k])
        cart.set_x(px - cart_w / 2)
        pole.set_data([px, tx], [py, ty])
        tip.set_data([tx], [ty])
        lo = max(0, k - 60)
        trail.set_data(
            x[lo : k + 1] + pole_length * np.sin(th[lo : k + 1]),
            pole_length * np.cos(th[lo : k + 1]),
        )
        m = mode[min(k, mode.size - 1)]
        label.set_text(f"t = {res.t[k]:4.1f} s   mode: {'LQR' if m > 0.5 else 'swing-up'}")
        return cart, pole, tip, trail, label

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    size = path.stat().st_size
    if size > MAX_GIF_BYTES:
        raise RuntimeError(f"{path} is {size / 1e6:.1f} MB, above the 5 MB budget")
    return path

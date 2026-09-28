"""Controller interface shared by every control law in the lab.

Controllers are *batched*: every call processes ``B`` independent closed loops at
once (``B = 1`` for a single simulation, thousands for Monte Carlo or
region-of-attraction sweeps). Internal state is therefore stored as arrays whose
leading dimension is the batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pendulum_lab.model import wrap_angle
from pendulum_lab.types import FloatArray


@dataclass(frozen=True)
class Observation:
    """Information available to a controller at sample ``k``.

    Attributes:
        x_hat: State estimate ``(B, 4)`` (true state for ideal full-state feedback).
        y: Raw measurement ``(B, 2)`` = ``[x, theta]`` with noise.
        ref: Cart-position reference ``(B,)`` [m].
        pending: Commands already issued but not yet applied by the delayed
            actuator, oldest first, shape ``(B, d)``.
        k: Sample index.
        t: Time [s].
    """

    x_hat: FloatArray
    y: FloatArray
    ref: FloatArray
    pending: FloatArray
    k: int
    t: float


class Controller(ABC):
    """Abstract batched discrete-time controller."""

    #: Short name used in tables and legends.
    name: str = "controller"

    @abstractmethod
    def reset(self, batch: int) -> None:
        """Clear internal state for a batch of ``batch`` closed loops."""

    @abstractmethod
    def compute(self, obs: Observation) -> FloatArray:
        """Return the (unsaturated) force command, shape ``(B,)`` [N]."""

    @property
    def mode(self) -> FloatArray | None:
        """Optional per-loop operating mode (used by hybrid controllers for logging)."""
        return None


def state_error(x_hat: FloatArray, ref: FloatArray) -> FloatArray:
    """Regulation error ``x_hat - [r, 0, 0, 0]`` with the pole angle wrapped to ``[-pi, pi)``.

    Args:
        x_hat: State estimate ``(B, 4)``.
        ref: Cart-position reference ``(B,)``.

    Returns:
        Error array ``(B, 4)``.
    """
    err = np.array(x_hat, dtype=float, copy=True)
    err[:, 0] -= ref
    err[:, 2] = wrap_angle(err[:, 2])
    return err

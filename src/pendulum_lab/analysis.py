"""Linear-systems analysis: discretisation, controllability, observability, delay augmentation.

Everything here works on the linearised upright model ``x_dot = A x + B u``,
``y = C x`` and its zero-order-hold (ZOH) discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov

from pendulum_lab.types import ComplexArray, FloatArray


def c2d_zoh(A: FloatArray, B: FloatArray, Ts: float) -> tuple[FloatArray, FloatArray]:
    """Exact zero-order-hold discretisation.

    Uses the block-matrix exponential
    ``expm([[A, B], [0, 0]] Ts) = [[Ad, Bd], [0, I]]``.

    Args:
        A: Continuous state matrix ``(n, n)``.
        B: Continuous input matrix ``(n, m)``.
        Ts: Sampling period [s].

    Returns:
        ``(Ad, Bd)``.
    """
    n, m = B.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n] = A
    blk[:n, n:] = B
    E = expm(blk * Ts)
    return E[:n, :n], E[:n, n:]


def van_loan_process_noise(A: FloatArray, G: FloatArray, Qc: FloatArray, Ts: float) -> FloatArray:
    """Discretise continuous white process noise with Van Loan's method.

    For ``x_dot = A x + G w`` with ``E[w w^T] = Qc delta(t)`` returns
    ``Qd = int_0^Ts e^{A s} G Qc G^T e^{A^T s} ds``.

    Args:
        A: Continuous state matrix ``(n, n)``.
        G: Noise input matrix ``(n, q)``.
        Qc: Continuous noise intensity ``(q, q)``.
        Ts: Sampling period [s].

    Returns:
        Discrete process-noise covariance ``(n, n)``.
    """
    n = A.shape[0]
    blk = np.zeros((2 * n, 2 * n))
    blk[:n, :n] = -A
    blk[:n, n:] = G @ Qc @ G.T
    blk[n:, n:] = A.T
    E = expm(blk * Ts)
    Ad_T = E[n:, n:]
    Qd = Ad_T.T @ E[:n, n:]
    return np.asarray(0.5 * (Qd + Qd.T), dtype=float)


def controllability_matrix(A: FloatArray, B: FloatArray) -> FloatArray:
    """Kalman controllability matrix ``[B, AB, ..., A^{n-1} B]``."""
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return np.hstack(blocks)


def observability_matrix(A: FloatArray, C: FloatArray) -> FloatArray:
    """Kalman observability matrix ``[C; CA; ...; C A^{n-1}]``."""
    n = A.shape[0]
    blocks = [C]
    for _ in range(n - 1):
        blocks.append(blocks[-1] @ A)
    return np.vstack(blocks)


def pbh_uncontrollable_modes(A: FloatArray, B: FloatArray, tol: float = 1e-9) -> list[complex]:
    """Popov-Belevitch-Hautus test: eigenvalues ``lam`` where ``[lam I - A, B]`` loses rank.

    Args:
        A: State matrix.
        B: Input matrix.
        tol: Relative singular-value tolerance.

    Returns:
        List of uncontrollable eigenvalues (empty if the pair is controllable).
    """
    n = A.shape[0]
    bad: list[complex] = []
    for lam in np.linalg.eigvals(A):
        M = np.hstack([lam * np.eye(n) - A, B.astype(complex)])
        sv = np.linalg.svd(M, compute_uv=False)
        if sv[-1] < tol * max(sv[0], 1.0):
            bad.append(complex(lam))
    return bad


def pbh_unobservable_modes(A: FloatArray, C: FloatArray, tol: float = 1e-9) -> list[complex]:
    """PBH observability test (dual of :func:`pbh_uncontrollable_modes`)."""
    return pbh_uncontrollable_modes(A.T, C.T, tol)


@dataclass(frozen=True)
class StructuralReport:
    """Summary of controllability/observability for one input/output configuration.

    Attributes:
        label: Human-readable description of the configuration.
        rank: Rank of the controllability or observability matrix.
        n: State dimension.
        condition_number: 2-norm condition number of that matrix.
        weak_modes: Eigenvalues flagged by the PBH test.
    """

    label: str
    rank: int
    n: int
    condition_number: float
    weak_modes: tuple[complex, ...]

    @property
    def full_rank(self) -> bool:
        """Whether the property (controllability or observability) holds."""
        return self.rank == self.n


def controllability_report(A: FloatArray, B: FloatArray, label: str) -> StructuralReport:
    """Controllability rank, conditioning and PBH-uncontrollable modes."""
    Wc = controllability_matrix(A, B)
    return StructuralReport(
        label=label,
        rank=int(np.linalg.matrix_rank(Wc)),
        n=A.shape[0],
        condition_number=float(np.linalg.cond(Wc)),
        weak_modes=tuple(pbh_uncontrollable_modes(A, B)),
    )


def observability_report(A: FloatArray, C: FloatArray, label: str) -> StructuralReport:
    """Observability rank, conditioning and PBH-unobservable modes."""
    Wo = observability_matrix(A, C)
    return StructuralReport(
        label=label,
        rank=int(np.linalg.matrix_rank(Wo)),
        n=A.shape[0],
        condition_number=float(np.linalg.cond(Wo)),
        weak_modes=tuple(pbh_unobservable_modes(A, C)),
    )


def discrete_gramian(A: FloatArray, B: FloatArray) -> FloatArray:
    """Infinite-horizon discrete controllability Gramian (requires Schur-stable ``A``)."""
    return np.asarray(solve_discrete_lyapunov(A, B @ B.T), dtype=float)


def finite_horizon_gramian(A: FloatArray, B: FloatArray, steps: int) -> FloatArray:
    """Finite-horizon discrete controllability Gramian ``sum_k A^k B B^T A^{kT}``.

    Valid for unstable ``A``. Its smallest eigenvalue measures the energy needed
    to reach the hardest direction in state space within ``steps`` samples.
    """
    n = A.shape[0]
    W = np.zeros((n, n))
    Ak = np.eye(n)
    for _ in range(steps):
        W += Ak @ B @ B.T @ Ak.T
        Ak = A @ Ak
    return W


def augment_input_delay(
    Ad: FloatArray, Bd: FloatArray, delay_steps: int
) -> tuple[FloatArray, FloatArray]:
    """Augment a discrete model with a pure input delay of ``d`` samples.

    The augmented state is ``z_k = [x_k; u_{k-d}; ...; u_{k-1}]`` so that
    ``x_{k+1} = Ad x_k + Bd u_{k-d}`` and the buffer shifts by one each step.

    Args:
        Ad: Discrete state matrix ``(n, n)``.
        Bd: Discrete input matrix ``(n, 1)``.
        delay_steps: Delay ``d >= 0``.

    Returns:
        ``(Az, Bz)`` of size ``n + d``. For ``d = 0`` the inputs are returned.
    """
    d = int(delay_steps)
    if d < 0:
        raise ValueError("delay_steps must be non-negative")
    if d == 0:
        return Ad.copy(), Bd.copy()
    n = Ad.shape[0]
    Az = np.zeros((n + d, n + d))
    Az[:n, :n] = Ad
    Az[:n, n] = Bd[:, 0]
    for i in range(d - 1):
        Az[n + i, n + i + 1] = 1.0
    Bz = np.zeros((n + d, 1))
    Bz[-1, 0] = 1.0
    return Az, Bz


def spectral_radius(A: FloatArray) -> float:
    """Largest eigenvalue magnitude; ``< 1`` means the discrete system is stable."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def continuous_to_discrete_poles(poles: ComplexArray | list[complex], Ts: float) -> ComplexArray:
    """Map s-plane poles to the z-plane via ``z = exp(s Ts)``."""
    return np.asarray(np.exp(np.asarray(poles, dtype=complex) * Ts))

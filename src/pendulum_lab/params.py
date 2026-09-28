"""Physical parameters and laboratory configuration.

All physical quantities are SI. Every parameter of :class:`CartPoleParams` may be
either a Python ``float`` (nominal model, used for controller design) or a NumPy
array of shape ``(B,)`` (a batch of perturbed plants, used for Monte Carlo). The
dynamics in :mod:`pendulum_lab.model` broadcast over that batch dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from pendulum_lab.types import FloatArray

Scalar = float | FloatArray


@dataclass(frozen=True)
class CartPoleParams:
    """Physical parameters of a cart with a rigid pole hinged on top of it.

    The pole is a uniform rod unless ``J`` is given explicitly.

    Attributes:
        M: Cart mass [kg].
        m: Pole mass [kg].
        l: Distance from the pivot to the pole centre of mass [m].
        J: Pole moment of inertia about its centre of mass [kg m^2].
        b_c: Viscous friction coefficient of the cart [N s/m].
        b_p: Viscous friction coefficient of the pivot [N m s/rad].
        g: Gravitational acceleration [m/s^2].
    """

    M: Scalar = 0.5
    m: Scalar = 0.2
    l: Scalar = 0.3
    J: Scalar = 0.2 * (2 * 0.3) ** 2 / 12.0
    b_c: Scalar = 0.1
    b_p: Scalar = 0.002
    g: Scalar = 9.81

    @classmethod
    def uniform_rod(
        cls,
        M: Scalar,
        m: Scalar,
        l: Scalar,
        b_c: Scalar,
        b_p: Scalar,
        g: Scalar = 9.81,
    ) -> CartPoleParams:
        """Build parameters for a uniform rod of total length ``2 l``.

        Args:
            M: Cart mass [kg].
            m: Rod mass [kg].
            l: Half-length of the rod (pivot to centre of mass) [m].
            b_c: Cart viscous friction [N s/m].
            b_p: Pivot viscous friction [N m s/rad].
            g: Gravitational acceleration [m/s^2].

        Returns:
            Parameters with ``J = m (2 l)^2 / 12``.
        """
        J = m * (2.0 * np.asarray(l)) ** 2 / 12.0
        J_val: Scalar = float(J) if np.ndim(J) == 0 else J
        return cls(M=M, m=m, l=l, J=J_val, b_c=b_c, b_p=b_p, g=g)

    @property
    def inertia_pivot(self) -> Scalar:
        """Pole moment of inertia about the pivot, ``J + m l^2`` [kg m^2]."""
        return self.J + self.m * self.l**2

    @property
    def batch_size(self) -> int | None:
        """Batch size if any parameter is an array, otherwise ``None``."""
        sizes = {np.size(v) for v in self._values() if np.ndim(v) > 0}
        if not sizes:
            return None
        if len(sizes) != 1:
            raise ValueError(f"inconsistent parameter batch sizes: {sizes}")
        return sizes.pop()

    def _values(self) -> tuple[Scalar, ...]:
        return (self.M, self.m, self.l, self.J, self.b_c, self.b_p, self.g)

    def is_nominal(self) -> bool:
        """Return ``True`` when every parameter is a scalar."""
        return self.batch_size is None

    def scaled(self, **factors: float) -> CartPoleParams:
        """Return a copy with selected parameters multiplied by scalar factors.

        Args:
            **factors: Mapping from attribute name to multiplicative factor.

        Returns:
            New parameter set.
        """
        changes = {k: getattr(self, k) * v for k, v in factors.items()}
        return replace(self, **changes)


@dataclass(frozen=True)
class SensorConfig:
    """Measurement model ``y = [x, theta] + v``.

    Attributes:
        sigma_x: Standard deviation of the cart-position noise [m].
        sigma_theta: Standard deviation of the pole-angle noise [rad].
        quant_x: Optional encoder quantisation step for position [m].
        quant_theta: Optional encoder quantisation step for angle [rad].
    """

    sigma_x: float = 1e-3
    sigma_theta: float = 2e-3
    quant_x: float | None = None
    quant_theta: float | None = None

    @property
    def R(self) -> NDArray[np.float64]:
        """Measurement-noise covariance matrix (2x2)."""
        return np.diag([self.sigma_x**2, self.sigma_theta**2])


@dataclass(frozen=True)
class ActuatorConfig:
    """Actuator model: saturated force with a pure transport delay.

    Attributes:
        u_max: Force saturation limit [N].
        delay_steps: Input delay in samples (the command computed at step ``k``
            is applied during ``[t_{k+d}, t_{k+d+1})``).
    """

    u_max: float = 10.0
    delay_steps: int = 1


@dataclass(frozen=True)
class LabConfig:
    """Discrete-time laboratory configuration shared by every experiment.

    Attributes:
        Ts: Controller sampling period [s].
        substeps: RK4 integration substeps per sampling period for the plant.
        track_limit: Half-length of the cart rail [m]; leaving it is a failure.
        force_disturbance_std: Std-dev of a piecewise-constant random force
            acting on the cart (unknown to the controller) [N].
        sensor: Measurement model.
        actuator: Actuator model.
        params: Nominal physical parameters.
    """

    Ts: float = 0.01
    substeps: int = 4
    track_limit: float = 1.0
    force_disturbance_std: float = 0.05
    sensor: SensorConfig = field(default_factory=SensorConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    params: CartPoleParams = field(default_factory=CartPoleParams)

    def with_delay(self, delay_steps: int) -> LabConfig:
        """Return a copy with a different actuator delay."""
        return replace(self, actuator=replace(self.actuator, delay_steps=delay_steps))

    def with_Ts(self, Ts: float) -> LabConfig:
        """Return a copy with a different sampling period."""
        return replace(self, Ts=Ts)


#: Default seed used by every stochastic experiment in the lab.
DEFAULT_SEED = 20260928

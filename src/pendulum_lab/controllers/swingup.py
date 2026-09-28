r"""Energy-based swing-up (Åström & Furuta, 2000) with a hysteretic switch to LQR.

Energy shaping
--------------
With the pole energy ``E = 1/2 I theta_dot^2 + m g l (cos(theta) - 1)`` (zero at
the upright equilibrium, ``I = J + m l^2``) the pole equation gives

.. math::

    \dot E = -m l \cos\theta\,\dot\theta\,\ddot x - b_p \dot\theta^2 .

Commanding the cart acceleration

.. math::

    \ddot x = \mathrm{sat}_{a_{max}}\big(k_E (E - E_{ref})\,
              \mathrm{sign}(\dot\theta\cos\theta)\big)
              - k_x x - k_v \dot x

makes ``E_dot = -m l k_E (E - E_ref) |theta_dot cos(theta)| + ...`` so the energy
error decays monotonically (ignoring the cart-centring term and friction). The
desired acceleration is converted into a force by inverting the nominal model
(:func:`pendulum_lab.model.cart_force_for_acceleration`).

Switching
---------
When the wrapped angle enters ``|theta| < theta_catch`` with a small enough rate
the balancing LQR takes over; if the pole leaves ``|theta| > theta_release`` the
controller falls back to swing-up. The gap between the two thresholds is a
hysteresis band that prevents chattering between modes. The rate condition is
essential: an LQR with a 10 N actuator cannot absorb a pole arriving at 5 rad/s
without a large cart excursion.

Robustness to model mismatch
----------------------------
The law regulates the *nominal-model* energy. For a plant whose pole is longer
or shorter than modelled, the nominal energy target over- or under-shoots the
top. A per-swing adaptation of the target (``adapt_gain``) raises it after each
swing that turns back short of the catch window, and lowers it after each
fly-over faster than ``omega_catch``. In Monte Carlo this raises the success
rate from roughly 70 % to roughly 99 % (see the README for measured numbers).
"""

from __future__ import annotations

import numpy as np

from pendulum_lab.controllers.base import Controller, Observation
from pendulum_lab.controllers.state_feedback import StateFeedback
from pendulum_lab.model import cart_force_for_acceleration, pendulum_energy, wrap_angle
from pendulum_lab.params import CartPoleParams
from pendulum_lab.types import FloatArray

#: Mode codes logged by :class:`EnergySwingUp`.
MODE_SWING = 0.0
MODE_BALANCE = 1.0


class EnergySwingUp(Controller):
    """Hybrid swing-up + balance controller (batched)."""

    def __init__(
        self,
        params: CartPoleParams,
        balance: StateFeedback,
        k_energy: float = 30.0,
        a_max: float = 6.0,
        k_x: float = 3.0,
        k_v: float = 3.75,
        energy_margin: float = 0.03,
        adapt_gain: float = 1.0,
        adapt_max: float = 0.5,
        direction_band: float = 0.05,
        adapt_tolerance: float | None = None,
        theta_catch: float = 0.35,
        omega_catch: float = 2.5,
        theta_release: float = 0.9,
        name: str = "Energy swing-up + LQR",
    ) -> None:
        """Create the controller.

        Args:
            params: Nominal model used for energy computation and force inversion.
            balance: Balancing state-feedback controller used near the top.
            k_energy: Energy gain ``k_E`` [(m/s^2)/J].
            a_max: Saturation of the energy-pumping acceleration [m/s^2].
            k_x: Cart-centring stiffness during swing-up [1/s^2].
            k_v: Cart-centring damping during swing-up [1/s].
            energy_margin: Target energy above the upright value [J]; a small
                positive margin compensates friction losses.
            adapt_gain: Gain ``kappa`` of the per-swing energy-target adaptation
                (0 disables it). See :meth:`_adapt_energy_target`.
            adapt_max: Bound on the magnitude of the learned energy offset [J].
            direction_band: Hysteresis band on ``theta_dot cos(theta)`` for the
                pumping direction [rad/s].
            adapt_tolerance: Optional energy-convergence gate for the turn-back
                adaptation, as a fraction of ``m g l`` (``None`` disables the gate;
                see the ablation study in the README).
            theta_catch: Angle below which the balance controller engages [rad].
            omega_catch: Angular-rate limit for engaging the balance controller [rad/s].
            theta_release: Angle above which swing-up re-engages [rad].
            name: Display name.
        """
        if not params.is_nominal():
            raise ValueError("EnergySwingUp needs nominal (scalar) parameters")
        if theta_release <= theta_catch:
            raise ValueError("theta_release must exceed theta_catch (hysteresis)")
        self.params = params
        self.balance = balance
        self.k_energy = k_energy
        self.a_max = a_max
        self.k_x = k_x
        self.k_v = k_v
        self.energy_margin = energy_margin
        self.adapt_gain = adapt_gain
        self.adapt_max = adapt_max
        self.direction_band = direction_band
        self.adapt_tolerance = adapt_tolerance
        self.theta_catch = theta_catch
        self.omega_catch = omega_catch
        self.theta_release = theta_release
        self.name = name
        self._mode: FloatArray = np.zeros(1)
        self._extra_energy: FloatArray = np.zeros(1)
        self._prev: FloatArray | None = None
        self._direction: FloatArray = np.ones(1)

    @property
    def mode(self) -> FloatArray:
        """Current mode per loop: 0 = swing-up, 1 = balance."""
        return self._mode

    def reset(self, batch: int) -> None:
        """Start every loop in swing-up mode."""
        self._mode = np.full(batch, MODE_SWING)
        self._extra_energy = np.zeros(batch)
        self._prev = None
        self._direction = np.ones(batch)
        self.balance.reset(batch)

    def swing_acceleration(self, x_hat: FloatArray, ref: FloatArray) -> FloatArray:
        """Cart acceleration requested by the energy law (before force conversion)."""
        E = pendulum_energy(x_hat, self.params)
        th = x_hat[:, 2]
        om = x_hat[:, 3]
        switch = om * np.cos(th)
        # sign() with hysteresis: keeps the previous direction while |switch| is within
        # the noise band, which avoids chattering at rest and still kicks the pole off
        # (the initial direction is +1).
        self._direction = np.where(
            np.abs(switch) > self.direction_band, np.sign(switch), self._direction
        )
        a_energy = np.clip(
            self.k_energy * (E - self.energy_margin - self._extra_energy) * self._direction,
            -self.a_max,
            self.a_max,
        )
        return np.asarray(a_energy - self.k_x * (x_hat[:, 0] - ref) - self.k_v * x_hat[:, 1])

    def _adapt_energy_target(self, x_hat: FloatArray) -> None:
        """Per-swing correction of the energy target (robustness to model mismatch).

        The energy law regulates the energy *of the nominal model*. If the real pole
        is, e.g., shorter than modelled, reaching the nominal target is not enough
        to reach the top. Two events are detected while swinging:

        * **turn-back** in the upper half-plane outside the catch window (the
          angular rate changes sign at ``|theta| > theta_catch``): the energy
          deficit is about ``m g l (1 - cos theta)``, so the target is raised by
          ``kappa`` times that amount. An optional gate (``adapt_tolerance``)
          only accepts turn-backs once the nominal energy is near its target;
        * **fly-over** (the angle crosses the vertical faster than
          ``omega_catch``): the surplus ``1/2 I (omega^2 - omega_catch^2 / 4)`` is
          removed from the target.

        The learned offset is bounded by ``adapt_max``.
        """
        th = wrap_angle(x_hat[:, 2])
        om = x_hat[:, 3]
        if self._prev is not None:
            th_p, om_p = self._prev[:, 0], self._prev[:, 1]
            swinging = self._mode == MODE_SWING
            upper = np.abs(th) < np.pi / 2
            mgl = float(self.params.m * self.params.g * self.params.l)
            target = self.energy_margin + self._extra_energy
            if self.adapt_tolerance is None:
                converged = np.ones_like(th, dtype=bool)
            else:
                e_err = np.abs(pendulum_energy(x_hat, self.params) - target)
                converged = e_err < self.adapt_tolerance * mgl
            turn_back = (
                swinging
                & upper
                & converged
                & (np.sign(om) != np.sign(om_p))
                & (np.abs(th) > self.theta_catch)
            )
            fly_over = (
                swinging
                & (np.sign(th) != np.sign(th_p))
                & (np.abs(th) < 0.5)
                & (np.abs(om) > self.omega_catch)
            )
            inertia = float(self.params.inertia_pivot)
            deficit = mgl * (1.0 - np.cos(th))
            surplus = 0.5 * inertia * (om**2 - 0.25 * self.omega_catch**2)
            delta = self.adapt_gain * (deficit * turn_back - surplus * fly_over)
            self._extra_energy = np.clip(
                self._extra_energy + delta, -self.adapt_max, self.adapt_max
            )
        self._prev = np.stack([th, om], axis=1)

    def compute(self, obs: Observation) -> FloatArray:
        """Update the mode and return the force of the active law."""
        th_w = wrap_angle(obs.x_hat[:, 2])
        om = obs.x_hat[:, 3]
        catch = (np.abs(th_w) < self.theta_catch) & (np.abs(om) < self.omega_catch)
        release = np.abs(th_w) > self.theta_release
        self._mode = np.where(
            self._mode == MODE_SWING,
            np.where(catch, MODE_BALANCE, MODE_SWING),
            np.where(release, MODE_SWING, MODE_BALANCE),
        )
        if self.adapt_gain > 0.0:
            self._adapt_energy_target(obs.x_hat)
        a = self.swing_acceleration(obs.x_hat, obs.ref)
        u_swing = cart_force_for_acceleration(obs.x_hat, a, self.params)
        u_bal = self.balance.compute(obs)
        return np.asarray(np.where(self._mode == MODE_BALANCE, u_bal, u_swing), dtype=float)

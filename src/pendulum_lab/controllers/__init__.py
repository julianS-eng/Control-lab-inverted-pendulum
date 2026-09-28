"""Control laws: cascaded PID, pole placement, LQR (with delay compensation) and swing-up."""

from pendulum_lab.controllers.base import Controller, Observation, state_error
from pendulum_lab.controllers.pid import CascadePID, PIDGains
from pendulum_lab.controllers.state_feedback import (
    LQRDesign,
    StateFeedback,
    bryson_weights,
    dlqr,
    dlqr_delay_compensated,
    pole_placement_gain,
)
from pendulum_lab.controllers.swingup import MODE_BALANCE, MODE_SWING, EnergySwingUp

__all__ = [
    "MODE_BALANCE",
    "MODE_SWING",
    "CascadePID",
    "Controller",
    "EnergySwingUp",
    "LQRDesign",
    "Observation",
    "PIDGains",
    "StateFeedback",
    "bryson_weights",
    "dlqr",
    "dlqr_delay_compensated",
    "pole_placement_gain",
    "state_error",
]

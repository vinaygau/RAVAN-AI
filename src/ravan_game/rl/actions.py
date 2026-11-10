"""Discrete Gymnasium actions mapped to the existing player physics inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DriverAction(IntEnum):
    """One non-conflicting driver command per environment action."""

    COAST = 0
    ACCELERATE = 1
    BRAKE = 2
    STEER_LEFT = 3
    STEER_RIGHT = 4
    HANDBRAKE = 5


@dataclass(frozen=True)
class SimulationCommand:
    throttle: float
    steering: float
    handbrake: bool = False


ACTION_COMMANDS: dict[DriverAction, SimulationCommand] = {
    DriverAction.COAST: SimulationCommand(throttle=0.0, steering=0.0),
    DriverAction.ACCELERATE: SimulationCommand(throttle=1.0, steering=0.0),
    DriverAction.BRAKE: SimulationCommand(throttle=-1.0, steering=0.0),
    DriverAction.STEER_LEFT: SimulationCommand(throttle=0.0, steering=-1.0),
    DriverAction.STEER_RIGHT: SimulationCommand(throttle=0.0, steering=1.0),
    DriverAction.HANDBRAKE: SimulationCommand(throttle=0.0, steering=0.0, handbrake=True),
}


def command_for_action(action: int) -> tuple[DriverAction, SimulationCommand]:
    """Validate an action index and return its explicit physics command."""
    try:
        selected = DriverAction(action)
    except (TypeError, ValueError) as error:
        valid = ", ".join(f"{int(item)}={item.name}" for item in DriverAction)
        raise ValueError(f"Invalid Ravan driving action {action!r}; expected one of {valid}.") from error
    return selected, ACTION_COMMANDS[selected]

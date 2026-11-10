"""Input helpers for mouse-driven controls in the Pygame dashboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MouseDriveControl:
    """Support both held mouse pedals and short click-to-drive pulses."""

    tap_duration_s: float = 0.25
    # One steering click shifts about one lane in the vehicle motion model.
    steering_tap_duration_s: float = 1.15
    action: str | None = None
    held: bool = False
    remaining_s: float = 0.0

    @property
    def active_action(self) -> str | None:
        if self.held or self.remaining_s > 0:
            return self.action
        return None

    def press(self, action: str) -> None:
        if action not in {"left", "right", "brake", "accelerate", "handbrake", "coast"}:
            raise ValueError(f"Unknown mouse driving action: {action}")
        self.action = action
        self.held = True
        self.remaining_s = 0.0

    def release(self) -> None:
        if self.held:
            self.held = False
            duration = self.steering_tap_duration_s if self.action in {"left", "right"} else self.tap_duration_s
            self.remaining_s = max(self.remaining_s, duration)

    def pulse(self, action: str, duration_s: float) -> None:
        """Apply an explicit timed input, used for recognized voice commands."""
        if action not in {"left", "right", "brake", "accelerate", "handbrake", "coast"}:
            raise ValueError(f"Unknown mouse driving action: {action}")
        self.action = action
        self.held = False
        self.remaining_s = max(0.0, float(duration_s))

    def advance(self, dt: float) -> None:
        if not self.held and self.remaining_s > 0:
            self.remaining_s = max(0.0, self.remaining_s - max(0.0, dt))
            if self.remaining_s == 0:
                self.action = None

    def clear(self) -> None:
        self.action = None
        self.held = False
        self.remaining_s = 0.0

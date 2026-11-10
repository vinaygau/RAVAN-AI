"""Small local command grammar; natural questions stay with Ollama."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DrivingCommand:
    intent: str
    mouse_action: str | None
    duration_s: float
    acknowledgement: str


_QUESTIONS = re.compile(
    r"^(what|why|when|where|who|how|is|are|am|can|could|would|should|do|does|did|"
    r"tell me|explain|describe)\b"
)


def parse_driving_command(text: str) -> DrivingCommand | None:
    """Return a concrete game command, or None for ordinary conversation."""
    normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    normalized = " ".join(normalized.split())
    if not normalized or _QUESTIONS.match(normalized):
        return None
    for prefix in ("please ", "hey ravan ai ", "ravan ai ", "ravan "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    if "restart" in normalized or "reset simulation" in normalized:
        return DrivingCommand("restart", None, 0.0, "Simulation restarted. You are in control.")
    if "pause" in normalized or "stop simulation" in normalized:
        return DrivingCommand("pause", None, 0.0, "Simulation paused.")
    if normalized in {"start", "go", "resume"} or any(
        phrase in normalized for phrase in ("resume driving", "continue driving", "start simulation")
    ):
        return DrivingCommand("resume", None, 0.0, "Simulation resumed. You are in control.")
    if any(phrase in normalized for phrase in ("handbrake", "emergency brake", "parking brake")):
        return DrivingCommand("handbrake", "handbrake", 0.35, "Applying the handbrake briefly.")
    if any(phrase in normalized for phrase in ("accelerate", "speed up", "go faster", "increase speed")):
        return DrivingCommand("accelerate", "accelerate", 1.25, "Accelerating briefly.")
    if any(phrase in normalized for phrase in ("brake", "slow down", "decelerate", "reduce speed", "stop the car")):
        return DrivingCommand("brake", "brake", 0.9, "Braking briefly.")
    if any(phrase in normalized for phrase in ("coast", "maintain speed", "hold speed", "release pedals")):
        return DrivingCommand("coast", "coast", 1.5, "Coasting briefly.")

    has_steering_intent = any(
        phrase in normalized for phrase in ("turn ", "steer ", "change lane", "switch lane", "move over", "go left", "go right")
    )
    if has_steering_intent:
        if "left" in normalized:
            return DrivingCommand("steer_left", "left", 1.15, "Steering left briefly.")
        if "right" in normalized:
            return DrivingCommand("steer_right", "right", 1.15, "Steering right briefly.")
    return None

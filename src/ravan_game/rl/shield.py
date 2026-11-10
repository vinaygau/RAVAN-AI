"""Auditable action guard for research-policy recommendations."""

from __future__ import annotations

from typing import Any, Callable

from .actions import DriverAction


class PhysicsSafetyShield:
    """Override PPO suggestions that conflict with the current risk checks.

    The shield is a simulation-only advisory layer. It uses the project's
    explainable physics assessment and returns a short reason for each change.
    """

    def filter(
        self,
        environment,
        proposed: int | DriverAction,
        *,
        fallback_policy: Callable[[Any], int | DriverAction] | None = None,
    ) -> tuple[DriverAction, str | None]:
        action = DriverAction(int(proposed))
        risk = environment._assessment
        front_gap = risk.distance_m
        front_ttc = risk.ttc_s
        front_urgent = (
            risk.collision
            or (front_ttc is not None and front_ttc <= 1.8)
            or (front_gap is not None and front_gap <= 7.0)
        )
        rear_ttc = (risk.nearest_rear or {}).get("ttc_s")
        rear_urgent = isinstance(rear_ttc, (int, float)) and rear_ttc <= 1.8

        def override(reason: str) -> tuple[DriverAction, str]:
            fallback_action = DriverAction.COAST if fallback_policy is None else fallback_policy(environment)
            safe_action = DriverAction(int(fallback_action))
            return safe_action, reason

        if action == DriverAction.STEER_LEFT and not risk.lane_change_left.safe:
            return override("Left lane failed the current front/rear gap safety check.")
        if action == DriverAction.STEER_RIGHT and not risk.lane_change_right.safe:
            return override("Right lane failed the current front/rear gap safety check.")
        if front_urgent and action not in (DriverAction.BRAKE, DriverAction.HANDBRAKE):
            return override("Front TTC or bumper gap crossed the emergency-braking threshold.")
        if rear_urgent and not front_urgent and action in (DriverAction.BRAKE, DriverAction.HANDBRAKE):
            return override("A rear vehicle is closing urgently; abrupt deceleration was rejected.")
        if front_ttc is not None and front_ttc < 3.0 and action == DriverAction.ACCELERATE:
            return override("Acceleration was rejected while front TTC is below 3 seconds.")
        if action == DriverAction.HANDBRAKE and not risk.collision:
            return override("Handbrake was replaced by a smoother response.")
        return action, None

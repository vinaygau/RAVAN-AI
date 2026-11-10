"""Small tabular trial-and-error coach based on simulated risk outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ACTIONS = {"accelerate", "brake", "steer_left", "steer_right"}


class TrialErrorLearner:
    """Remember whether sustained player actions improved risk in similar states.

    This is a transparent contextual bandit, not an autonomous driving policy.
    It only observes actions the player chose and compares the risk score before
    and after a short trial window.
    """

    def __init__(self, path: Path | None = None, trial_window_s: float = 2.0) -> None:
        self.path = path or Path(__file__).resolve().parents[2] / "data" / "learning" / "trial_error.json"
        self.trial_window_s = max(1.0, trial_window_s)
        self.minimum_action_s = 0.2
        self._table: dict[str, dict[str, dict[str, float | int]]] = {}
        self._trial: dict[str, Any] | None = None
        self._load()

    @staticmethod
    def state_bucket(assessment: Any) -> str:
        if assessment.collision or assessment.risk_level == "CRITICAL":
            return "critical_conflict"
        front = assessment.nearest_front or {}
        if (
            front
            and assessment.relative_speed_mps is not None
            and assessment.relative_speed_mps > 0
            and assessment.ttc_s is not None
            and assessment.ttc_s <= 5.0
        ):
            return "closing_on_front"
        rear = assessment.nearest_rear or {}
        if rear.get("ttc_s") is not None and float(rear["ttc_s"]) <= 5.0:
            return "rear_approach"
        if not assessment.lane_change_left.safe and not assessment.lane_change_right.safe:
            return "adjacent_lanes_blocked"
        if assessment.risk_level == "MODERATE":
            return "moderate_risk"
        return "open_traffic"

    def observe(self, time_s: float, assessment: Any, action: str | None) -> bool:
        """Update one action trial; return True when a completed trial was learned."""
        if action not in _ACTIONS:
            if self._trial is None:
                return False
            action_duration = time_s - self._trial["start_time_s"]
            if action_duration < self.minimum_action_s:
                self._trial = None
                return False
            if self._trial.get("released_at_s") is None:
                self._trial["released_at_s"] = time_s
            if time_s - self._trial["released_at_s"] >= self.trial_window_s:
                return self._finish(time_s, assessment, force=True)
            return False

        state = self.state_bucket(assessment)
        if self._trial is None or self._trial["action"] != action or self._trial["state"] != state:
            learned = False
            if self._trial is not None:
                duration = time_s - self._trial["start_time_s"]
                if duration >= self.trial_window_s:
                    learned = self._finish(time_s, assessment, force=True)
                elif duration >= self.minimum_action_s:
                    learned = self._finish(time_s, assessment, force=True)
                else:
                    self._trial = None
            self._trial = {
                "state": state,
                "action": action,
                "start_time_s": time_s,
                "start_risk_score": assessment.risk_score,
                "released_at_s": None,
            }
            return learned

        self._trial["released_at_s"] = None
        if time_s - self._trial["start_time_s"] < self.trial_window_s:
            return False
        learned = self._finish(time_s, assessment, force=True)
        self._trial = {
            "state": state,
            "action": action,
            "start_time_s": time_s,
            "start_risk_score": assessment.risk_score,
            "released_at_s": None,
        }
        return learned

    def _finish(self, time_s: float, assessment: Any, *, force: bool = False) -> bool:
        if self._trial is None:
            return False
        duration = time_s - self._trial["start_time_s"]
        if duration < self.trial_window_s and not force:
            return False

        start_score = float(self._trial["start_risk_score"])
        change = start_score - float(assessment.risk_score)
        reward = max(-1.0, min(1.0, change / 40.0))
        if assessment.collision:
            reward = -1.0
        state = str(self._trial["state"])
        action = str(self._trial["action"])
        state_table = self._table.setdefault(state, {})
        prior = state_table.get(action, {"count": 0, "mean_reward": 0.0, "improved": 0, "worsened": 0})
        count = int(prior["count"]) + 1
        mean_reward = float(prior["mean_reward"]) + (reward - float(prior["mean_reward"])) / count
        state_table[action] = {
            "count": count,
            "mean_reward": round(mean_reward, 4),
            "improved": int(prior["improved"]) + int(reward > 0.05),
            "worsened": int(prior["worsened"]) + int(reward < -0.05),
        }
        self._trial = None
        self.save()
        return True

    def outcomes(self, state: str) -> dict[str, dict[str, float | int]]:
        return {action: dict(values) for action, values in self._table.get(state, {}).items()}

    def hint(self, state: str) -> str:
        outcomes = self._table.get(state, {})
        experienced = [(action, result) for action, result in outcomes.items() if int(result["count"]) >= 2]
        if not experienced:
            return "Trial learning: hold a control for about 2 seconds to compare the risk change."
        action, result = max(experienced, key=lambda item: float(item[1]["mean_reward"]))
        count = int(result["count"])
        improved = int(result["improved"])
        if float(result["mean_reward"]) > 0.05:
            return f"In {state.replace('_', ' ')} trials, {action.replace('_', ' ')} lowered risk in {improved}/{count} observations."
        if float(result["mean_reward"]) < -0.05:
            return f"In {state.replace('_', ' ')} trials, {action.replace('_', ' ')} raised risk in {int(result['worsened'])}/{count} observations."
        return f"Trial learning has {count} {state.replace('_', ' ')} observations; no action has a clear benefit yet."

    def context(self, assessment: Any) -> dict[str, Any]:
        state = self.state_bucket(assessment)
        return {
            "method": "simulated trial-and-error; observed player actions and risk-score changes",
            "state_bucket": state,
            "action_outcomes": self.outcomes(state),
            "summary": self.hint(state),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._table, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._table = loaded
        except (OSError, json.JSONDecodeError):
            self._table = {}

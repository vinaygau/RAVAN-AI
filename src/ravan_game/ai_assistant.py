"""Asynchronous local Ollama client for game-context explanations."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from pathlib import Path
from typing import Any

import requests


def _verified_fallback(question: str, context: dict[str, Any]) -> str:
    """Answer with risk-engine values only if the model output is unsafe."""
    query = question.lower()
    level = str(context.get("risk_level", "UNKNOWN")).upper()
    recommendation = str(context.get("recommendation", "Keep monitoring the simulated traffic."))
    front = context.get("nearest_front") or {}
    nearby = context.get("nearby_vehicles") or []
    learning = context.get("trial_error_learning") or {}
    learning_summary = str(learning.get("summary", "")).strip()
    lane_plan = context.get("rl_lane_plan") or {}

    if ("path" in query or "route" in query) and lane_plan:
        route = " to ".join(lane_plan.get("lane_names", []))
        action = str(lane_plan.get("first_action", "stay")).replace("_", " ")
        rationale = str(lane_plan.get("rationale", ""))
        return (
            f"The Q-learning lane planner suggests {route or 'holding the current lane'} "
            f"over about {float(lane_plan.get('horizon_distance_m', 0.0)):.0f} m. "
            f"Its immediate suggestion is to {action}. {rationale}"
        )

    if "ttc" in query or "time to collision" in query:
        value = context.get("ttc_s")
        if value is None:
            return "TTC is undefined because the current same-lane closing condition is not present."
        return f"The risk engine reports a TTC of {float(value):.2f} seconds to the nearest same-lane vehicle."
    if "thw" in query or "headway" in query:
        value = context.get("thw_s")
        if value is None:
            return "Time headway is unavailable because there is no valid positive same-lane gap and ego speed."
        return f"The risk engine reports a time headway of {float(value):.2f} seconds."
    if "drac" in query or "deceleration" in query:
        value = context.get("drac_mps2")
        if value is None:
            return "DRAC is undefined because there is no positive same-lane closing speed and gap."
        return f"The risk engine reports DRAC of {float(value):.2f} m/s² for the nearest same-lane front vehicle."
    if "lane" in query or "change" in query:
        left = context.get("lane_change_left", {})
        right = context.get("lane_change_right", {})
        left_text = "clear" if left.get("safe") else "wait"
        right_text = "clear" if right.get("safe") else "wait"
        route = " → ".join((lane_plan.get("lane_names") or [])[:5])
        route_text = f" Q-learning path: {route}." if route else ""
        return f"Lane check: left is {left_text}; right is {right_text}.{route_text} {recommendation}"
    if "speed" in query or "km/h" in query:
        ego_speed = float(context.get("ego_speed_kmh", 0.0))
        if front:
            return (
                f"Your speed is {ego_speed:.1f} km/h. The nearest same-lane front vehicle is "
                f"moving at {float(front['speed_kmh']):.1f} km/h."
            )
        return f"Your speed is {ego_speed:.1f} km/h. No same-lane vehicle is ahead."
    if "distance" in query or "gap" in query or "close" in query:
        gap = context.get("distance_m")
        if gap is None:
            return "There is no valid same-lane front gap at this moment."
        return f"The nearest same-lane front bumper gap is {float(gap):.1f} m."
    if "behind" in query or "side" in query or "around" in query:
        descriptions = []
        for vehicle in nearby:
            relation = vehicle.get("lane_relation", "nearby")
            if relation in {"same-lane-rear", "alongside", "adjacent-lane-rear"}:
                descriptions.append(
                    f"{vehicle.get('vehicle', 'A vehicle')} is {relation.replace('-', ' ')} "
                    f"at {float(vehicle.get('distance_m', 0.0)):.1f} m and "
                    f"{float(vehicle.get('speed_kmh', 0.0)):.1f} km/h"
                )
        if descriptions:
            return "; ".join(descriptions[:3]) + "."
        return "No nearby rear or side vehicle is currently detected."

    factors = " ".join(str(factor) for factor in context.get("main_factors", [])[:2])
    experience = f" Trial feedback: {learning_summary}" if learning_summary else ""
    return f"The physics engine rates this situation {level.lower()} ({int(context.get('risk_score', 0))}/100). {factors} {recommendation}{experience}"


class OllamaAssistant:
    """Run local Ollama requests off the Pygame thread.

    A user question may run while a background coaching request is in flight;
    background coaching may not start while a user question is pending.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        path = config_path or Path(__file__).resolve().parents[2] / "config" / "assistant.json"
        config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.url = os.environ.get("RAVAN_OLLAMA_URL", config.get("url", "http://localhost:11434/api/chat"))
        self.model = os.environ.get("RAVAN_OLLAMA_MODEL", config.get("model", "qwen2.5:3b"))
        self.timeout_s = float(config.get("timeout_seconds", 45))
        self._results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._pending_purposes: set[str] = set()
        self._request_ids = {"question": 0, "guidance": 0}
        self._closed = False

    @property
    def pending(self) -> bool:
        with self._lock:
            return bool(self._pending_purposes)

    @property
    def question_pending(self) -> bool:
        with self._lock:
            return "question" in self._pending_purposes

    @property
    def guidance_pending(self) -> bool:
        with self._lock:
            return "guidance" in self._pending_purposes

    def submit(
        self,
        question: str,
        simulation_context: dict[str, Any],
        *,
        purpose: str = "question",
    ) -> bool:
        question = question.strip()
        if not question:
            return False
        request_purpose = "guidance" if purpose == "guidance" else "question"
        with self._lock:
            if self._closed:
                return False
            if request_purpose in self._pending_purposes:
                return False
            if request_purpose == "guidance" and "question" in self._pending_purposes:
                return False
            self._request_ids[request_purpose] += 1
            request_id = self._request_ids[request_purpose]
            self._pending_purposes.add(request_purpose)

        worker = threading.Thread(
            target=self._request,
            args=(question, simulation_context, request_purpose, request_id),
            name="ravan-ollama-chat",
            daemon=True,
        )
        worker.start()
        return True

    def submit_guidance(self, simulation_context: dict[str, Any]) -> bool:
        return self.submit(
            "Speak to the driver naturally, like a calm and helpful co-driver. "
            "Give one practical suggestion for this moment, or briefly say what to watch if the road is clear.",
            simulation_context,
            purpose="guidance",
        )

    def cancel_guidance(self) -> None:
        """Invalidate any live coaching request, for example when play pauses."""
        with self._lock:
            self._request_ids["guidance"] += 1
            self._pending_purposes.discard("guidance")

    def close(self) -> None:
        """Prevent late worker replies from being delivered after the game exits."""
        with self._lock:
            self._closed = True
            for purpose in self._request_ids:
                self._request_ids[purpose] += 1
            self._pending_purposes.clear()
            while True:
                try:
                    self._results.get_nowait()
                except queue.Empty:
                    break

    def _publish_result(self, purpose: str, request_id: int, result: dict[str, Any]) -> None:
        with self._lock:
            if self._closed or self._request_ids[purpose] != request_id:
                return
            result["request_id"] = request_id
            self._results.put(result)

    def _request(
        self,
        question: str,
        context: dict[str, Any],
        purpose: str = "question",
        request_id: int = 0,
    ) -> None:
        with self._lock:
            if self._closed or self._request_ids[purpose] != request_id:
                return
        tone = (
            "For live coaching, sound like a calm, natural co-driver speaking to a person. "
            "Give one useful action in one or two short sentences; avoid canned openings and repeated risk labels. "
            if purpose == "guidance"
            else "Answer directly and conversationally, in two to four short sentences."
        )
        system_prompt = (
            "You are Ravan AI, a calm, helpful assistant inside a driving education game. "
            "Speak naturally in clear Indian English, without forced slang or overly formal wording. "
            f"{tone} "
            "Use the supplied simulation JSON as the source of truth. The risk engine "
            "has already calculated every number: do not recalculate, alter, or invent "
            "speeds, distances, TTC, THW, DRAC, risk scores, or lane-change results. "
            "Recommendations apply only inside this simulation. "
            "If trial_error_learning is supplied, describe its observations as outcomes "
            "from this simulation only, not as proof or a universal rule. "
            "If rl_lane_plan is supplied, explain its short-horizon lane sequence as a "
            "Q-learning advisory; do not claim it controls the vehicle or guarantees safety. "
            "If a value is null, say it is unavailable rather than guessing."
        )
        user_content = (
            f"Player question: {question}\n\n"
            "Authoritative current simulation state (JSON):\n"
            f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "options": {"temperature": 0.35, "num_predict": 80},
        }

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            body = response.json()
            content = body.get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError("Ollama returned an empty answer.")
            # The local risk engine owns all numerical claims. Qwen occasionally
            # paraphrases a supplied measurement incorrectly, so use a precise
            # engine-backed reply when the model emits numbers or contradicts
            # the computed risk class.
            risk_phrase = re.search(r"\b(low|moderate|high|critical)\s+risk\b", content, re.IGNORECASE)
            wrong_risk = bool(
                risk_phrase
                and risk_phrase.group(1).upper() != str(context.get("risk_level", "")).upper()
            )
            if re.search(r"\d", content) or wrong_risk:
                content = _verified_fallback(question, context)
            self._publish_result(purpose, request_id, {"ok": True, "content": content, "purpose": purpose})
        except requests.exceptions.ConnectionError:
            self._publish_result(purpose, request_id, {
                "ok": False,
                "purpose": purpose,
                "content": "I couldn't reach local Ollama. The on-screen physics risk assessment is still available.",
            })
        except requests.exceptions.Timeout:
            self._publish_result(purpose, request_id, {
                "ok": False,
                "purpose": purpose,
                "content": "Ollama took too long to respond. The on-screen physics risk assessment is still available.",
            })
        except (requests.exceptions.RequestException, ValueError, KeyError) as error:
            self._publish_result(purpose, request_id, {
                "ok": False,
                "purpose": purpose,
                "content": f"The local AI request failed: {error}. The physics assessment is still available.",
            })
        finally:
            with self._lock:
                if self._request_ids[purpose] == request_id:
                    self._pending_purposes.discard(purpose)

    def poll(self) -> dict[str, Any] | None:
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return None
            purpose = result.get("purpose", "question")
            with self._lock:
                if not self._closed and purpose in self._request_ids:
                    # Only deliver a reply if it still belongs to the latest request.
                    request_id = result.get("request_id")
                    if request_id is None or self._request_ids[purpose] == request_id:
                        return result

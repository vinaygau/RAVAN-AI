"""Render deterministic dashboard screenshots for the project README."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from src.ravan_game.agent import RuleBasedDrivingAgent
from src.ravan_game.renderer import HEIGHT, WIDTH, GameRenderer
from src.ravan_game.risk import analyze_risk
from src.ravan_game.route_rl import plan_lane_path
from src.ravan_game.simulation import DrivingSimulation


def render(output_path: Path, *, agent_drive: bool) -> None:
    simulation = DrivingSimulation(seed=2026)
    risk = analyze_risk(simulation)

    if agent_drive:
        agent = RuleBasedDrivingAgent()
        actions = {
            "coast": (0.0, 0.0),
            "accelerate": (1.0, 0.0),
            "brake": (-1.0, 0.0),
            "steer_left": (0.0, -1.0),
            "steer_right": (0.0, 1.0),
        }
        for _ in range(180):
            throttle, steering = actions[agent.choose_action(simulation, risk)]
            simulation.update(0.05, throttle=throttle, steering=steering)
            risk = analyze_risk(simulation)
            if risk.collision:
                break

    renderer = GameRenderer()
    surface = pygame.Surface((WIDTH, HEIGHT))
    renderer.draw(
        surface,
        simulation,
        risk,
        paused=not agent_drive or risk.collision,
        started=agent_drive,
        chat_history=[
            ("assistant", "I’m checking the traffic around you. Agent Drive can manage the simulated car; switch to Manual at any time."),
        ],
        chat_input="",
        chat_focused=False,
        ai_pending=False,
        model_name="Local assistant",
        status_message=(
            "Agent Drive is managing the simulated car."
            if agent_drive and not risk.collision
            else "Simulation ready · start when you are ready."
        ),
        voice_enabled=True,
        selected_tab="Drive",
        agent_drive=agent_drive,
        live_guidance="Keep a steady pace and leave room around nearby vehicles.",
        learning_hint="The rule agent checks current gaps before changing lanes.",
        lane_plan=plan_lane_path(simulation, risk),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, output_path)


def main() -> None:
    pygame.init()
    output_dir = Path(__file__).resolve().parents[1] / "assets" / "screenshots"
    render(output_dir / "ravan-ai-dashboard.png", agent_drive=False)
    render(output_dir / "ravan-ai-agent-drive.png", agent_drive=True)
    pygame.quit()


if __name__ == "__main__":
    main()

"""Render a short MP4 demo from the deterministic Ravan AI simulation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from src.ravan_game.agent import RuleBasedDrivingAgent
from src.ravan_game.renderer import HEIGHT, WIDTH, GameRenderer
from src.ravan_game.risk import RiskAssessment, analyze_risk
from src.ravan_game.route_rl import LanePathPlan, plan_lane_path
from src.ravan_game.simulation import DrivingSimulation


FPS = 24
DEFAULT_DURATION_S = 18

ACTION_INPUTS = {
    "coast": (0.0, 0.0),
    "accelerate": (1.0, 0.0),
    "brake": (-1.0, 0.0),
    "steer_left": (0.0, -1.0),
    "steer_right": (0.0, 1.0),
}


def guidance_for(risk: RiskAssessment, agent: RuleBasedDrivingAgent) -> str:
    if risk.collision:
        return "The simulation detected a collision. Agent Drive has stopped the run."
    if risk.risk_level in {"HIGH", "CRITICAL"} or (risk.ttc_s is not None and risk.ttc_s < 3.0):
        return "Traffic is closing quickly. Braking to create more space."
    if agent.target_lane is not None:
        return "Checking the gaps and moving toward a clearer lane."
    if risk.risk_level == "MODERATE":
        return "Nearby traffic is being monitored. Holding a careful following gap."
    return "The road ahead is clear. Holding a steady pace."


def make_video(output: Path, *, seed: int, duration_s: int, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("FFmpeg was not found. Install FFmpeg and make sure it is on PATH.")

    pygame.init()
    simulation = DrivingSimulation(seed=seed)
    agent = RuleBasedDrivingAgent()
    renderer = GameRenderer()
    surface = pygame.Surface((WIDTH, HEIGHT))
    assessment = analyze_risk(simulation)
    lane_plan: LanePathPlan = plan_lane_path(simulation, assessment)
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if encoder.stdin is None:
        pygame.quit()
        raise RuntimeError("Could not open the FFmpeg video stream.")

    frame_count = duration_s * fps
    dt = 1.0 / fps
    frames_written = 0
    try:
        for frame_index in range(frame_count):
            if not assessment.collision:
                action = agent.choose_action(simulation, assessment)
                throttle, steering = ACTION_INPUTS[action]
                simulation.update(dt, throttle=throttle, steering=steering)
                assessment = analyze_risk(simulation)

            if frame_index % max(1, fps // 2) == 0:
                lane_plan = plan_lane_path(simulation, assessment)

            renderer.draw(
                surface,
                simulation,
                assessment,
                paused=assessment.collision,
                started=True,
                chat_history=[
                    (
                        "assistant",
                        "I’m monitoring nearby traffic. Agent Drive is active; switch to Manual at any time.",
                    ),
                ],
                chat_input="",
                chat_focused=False,
                ai_pending=False,
                model_name="Physics coach",
                status_message=(
                    "Simulation stopped after collision."
                    if assessment.collision
                    else "Agent Drive is managing the simulated car."
                ),
                voice_enabled=False,
                selected_tab="Drive",
                agent_drive=True,
                live_guidance=guidance_for(assessment, agent),
                learning_hint="The agent checks front and rear gaps before changing lanes.",
                lane_plan=lane_plan,
            )
            encoder.stdin.write(pygame.image.tostring(surface, "RGB"))
            frames_written += 1
    except BrokenPipeError as error:
        message = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        encoder.wait()
        raise RuntimeError(f"FFmpeg stopped before the video was complete. {message}") from error
    finally:
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()

    error_text = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    return_code = encoder.wait()
    pygame.quit()
    if return_code:
        raise RuntimeError(f"FFmpeg could not create the video. {error_text}")
    print(f"Rendered {frames_written} frames ({frames_written / fps:.1f} s) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic simulation seed")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION_S, help="Video length in seconds")
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "assets"
        / "videos"
        / "ravan-ai-simulation-demo.mp4",
        help="Output MP4 path",
    )
    args = parser.parse_args()
    if args.duration < 1 or args.fps < 1:
        parser.error("--duration and --fps must be positive integers")
    make_video(args.output, seed=args.seed, duration_s=args.duration, fps=args.fps)


if __name__ == "__main__":
    main()

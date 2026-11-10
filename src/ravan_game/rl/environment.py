"""Gymnasium wrapper around the project's existing three-lane simulation."""

from __future__ import annotations

from typing import Any
from random import Random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..risk import RiskAssessment, analyze_risk
from ..simulation import DrivingSimulation
from .actions import DriverAction, command_for_action
from .observations import build_observation, make_observation_space
from .reward import RewardConfig, calculate_reward
from .scenarios import ALL_SCENARIOS, apply_scenario


class RavanDrivingEnv(gym.Env):
    """Headless-capable environment using the same physics as the Pygame game.

    Actions are held for ``action_repeat`` fixed simulation updates. One
    ``step`` is therefore ``fixed_dt_s * action_repeat`` seconds of simulation.
    Rewards are calculated in a separate module and all components are
    returned in ``info`` for experiment analysis.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 1200,
        fixed_dt_s: float = 0.05,
        action_repeat: int = 2,
        progress_reward_per_m: float = 0.1,
        step_cost: float = -0.01,
        collision_penalty: float = -100.0,
        scenario_families: tuple[str, ...] | list[str] | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode {render_mode!r}.")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least 1.")
        if not 0 < fixed_dt_s <= 0.05:
            raise ValueError("fixed_dt_s must be greater than 0 and no more than 0.05 s.")
        if action_repeat < 1:
            raise ValueError("action_repeat must be at least 1.")

        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.fixed_dt_s = float(fixed_dt_s)
        self.action_repeat = int(action_repeat)
        # Retain the original arguments for callers that used the Phase 1 API.
        self.reward_config = reward_config or RewardConfig(
            progress_per_m=float(progress_reward_per_m),
            step_cost=float(step_cost),
            collision=float(collision_penalty),
        )
        self.scenario_families = tuple(scenario_families) if scenario_families else None
        if self.scenario_families:
            unknown = set(self.scenario_families) - set(ALL_SCENARIOS)
            if unknown:
                raise ValueError(f"Unknown scenario families: {', '.join(sorted(unknown))}")
        self.action_space = spaces.Discrete(len(DriverAction))
        self.observation_space = make_observation_space()

        self.simulation: DrivingSimulation | None = None
        self._assessment: RiskAssessment | None = None
        self._episode_seed: int | None = None
        self._scenario_family: str | None = None
        self._episode_step = 0
        self._episode_done = True
        self._steering_input = 0.0
        self._throttle_input = 0.0
        self._handbrake_input = False
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._renderer: Any | None = None

    def _require_simulation(self) -> tuple[DrivingSimulation, RiskAssessment]:
        if self.simulation is None or self._assessment is None:
            raise RuntimeError("Call reset() before step() or render().")
        return self.simulation, self._assessment

    def _observation(self) -> dict[str, np.ndarray]:
        simulation, assessment = self._require_simulation()
        return build_observation(
            simulation,
            assessment,
            steering_input=self._steering_input,
            throttle_input=self._throttle_input,
            handbrake_input=self._handbrake_input,
            episode_step=self._episode_step,
            max_episode_steps=self.max_episode_steps,
            max_distance_m=(
                simulation.player.max_speed_mps
                * self.max_episode_steps
                * self.fixed_dt_s
                * self.action_repeat
            ),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start a fresh deterministic episode and return observation/info."""
        super().reset(seed=seed)
        options = dict(options or {})
        requested_family = options.pop("scenario_family", None)
        if options:
            unknown = ", ".join(sorted(options))
            raise ValueError(f"Unsupported reset options: {unknown}.")
        episode_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))

        # Constructing a fresh simulator with the selected seed avoids the
        # restart counter changing the traffic distribution between resets.
        self.simulation = DrivingSimulation(seed=episode_seed)
        if requested_family is not None:
            if requested_family not in ALL_SCENARIOS:
                raise ValueError(f"Unknown scenario family {requested_family!r}.")
            if self.scenario_families and requested_family not in self.scenario_families:
                raise ValueError(f"Scenario {requested_family!r} is not enabled for this environment.")
            self._scenario_family = str(requested_family)
        elif self.scenario_families:
            self._scenario_family = str(self.np_random.choice(self.scenario_families))
        else:
            self._scenario_family = None
        if self._scenario_family is not None:
            apply_scenario(self.simulation, self._scenario_family, Random(episode_seed))
        self._assessment = analyze_risk(self.simulation)
        self._episode_seed = episode_seed
        self._episode_step = 0
        self._episode_done = False
        self._steering_input = 0.0
        self._throttle_input = 0.0
        self._handbrake_input = False
        observation = self._observation()
        info = {
            "seed": episode_seed,
            "episode_step": 0,
            "simulation_time_s": 0.0,
            "risk": self._assessment.to_dict(),
            "scenario_family": self._scenario_family,
        }
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance the real game simulation and return Gymnasium's five values."""
        if self._episode_done:
            raise RuntimeError("Episode is finished; call reset() before the next step().")
        simulation, _ = self._require_simulation()
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}.")
        selected_action, command = command_for_action(int(action))
        previous_steering = self._steering_input
        previous_throttle = self._throttle_input
        self._steering_input = command.steering
        self._throttle_input = command.throttle
        self._handbrake_input = command.handbrake

        previous_position_m = simulation.player.longitudinal_m
        for _ in range(self.action_repeat):
            simulation.update(
                self.fixed_dt_s,
                throttle=command.throttle,
                steering=command.steering,
                handbrake=command.handbrake,
            )
            self._assessment = analyze_risk(simulation)
            if self._assessment.collision:
                break

        self._episode_step += 1
        terminated = bool(self._assessment and self._assessment.collision)
        truncated = self._episode_step >= self.max_episode_steps and not terminated
        self._episode_done = terminated or truncated

        progress_m = max(0.0, simulation.player.longitudinal_m - previous_position_m)
        reward, reward_components = calculate_reward(
            simulation,
            self._assessment,
            action=selected_action,
            previous_position_m=previous_position_m,
            previous_steering=previous_steering,
            previous_throttle=previous_throttle,
            steering=command.steering,
            throttle=command.throttle,
            collision=terminated,
            config=self.reward_config,
        )

        observation = self._observation()
        info: dict[str, Any] = {
            "seed": self._episode_seed,
            "episode_step": self._episode_step,
            "simulation_time_s": round(simulation.elapsed_s, 4),
            "action_name": selected_action.name,
            "action_command": {
                "throttle": command.throttle,
                "steering": command.steering,
                "handbrake": command.handbrake,
            },
            "collision": terminated,
            "progress_m": progress_m,
            "reward_components": reward_components,
            "scenario_family": self._scenario_family,
            "risk": self._assessment.to_dict() if self._assessment else {},
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render using the existing dashboard or return an RGB HWC frame."""
        if self.render_mode is None:
            return None
        simulation, assessment = self._require_simulation()

        if self._pygame is None:
            import pygame

            self._pygame = pygame
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                from ..renderer import HEIGHT, WIDTH, GameRenderer

                self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Ravan AI — Gymnasium Environment")
            else:
                from ..renderer import HEIGHT, WIDTH, GameRenderer

                self._screen = pygame.Surface((WIDTH, HEIGHT))
            self._renderer = GameRenderer()

        from ..renderer import HEIGHT, WIDTH

        if self.render_mode == "rgb_array" and self._screen is None:
            self._screen = self._pygame.Surface((WIDTH, HEIGHT))
        self._renderer.draw(
            self._screen,
            simulation,
            assessment,
            paused=False,
            started=True,
            chat_history=[],
            chat_input="",
            chat_focused=False,
            ai_pending=False,
            model_name="Gymnasium",
            status_message="RL environment preview · controls are supplied through step(action)",
        )

        if self.render_mode == "human":
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self.close()
                    return None
            self._pygame.display.flip()
            return None

        pixels = self._pygame.surfarray.array3d(self._screen)
        return np.transpose(pixels, (1, 0, 2)).copy()

    def close(self) -> None:
        """Release rendering resources; headless use has nothing to close."""
        if self._pygame is not None:
            if self._pygame.display.get_init():
                self._pygame.display.quit()
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._renderer = None

    @property
    def episode_seed(self) -> int | None:
        return self._episode_seed

"""Dashboard-style Pygame interface for the Ravan AI driving simulation."""

from __future__ import annotations

from math import sin
import time

import pygame

from .risk import RiskAssessment
from .route_rl import LanePathPlan
from .simulation import DrivingSimulation, LANE_COUNT, LANE_WIDTH_M, ROAD_WIDTH_M


WIDTH = 1440
HEIGHT = 900
TOPBAR_HEIGHT = 64

NAV_BUTTONS = ["Drive", "Analytics", "AI Assistant", "Results", "Settings"]
NAV_BUTTON_RECTS = [pygame.Rect(493 + index * 116, 13, 108, 38) for index in range(len(NAV_BUTTONS))]

# Shared mouse hit regions for app.py.
CHAT_INPUT_RECT = pygame.Rect(1007, 305, 218, 30)
CHAT_MIC_RECT = pygame.Rect(1230, 305, 48, 30)
CHAT_VOICE_RECT = pygame.Rect(1283, 305, 65, 30)
CHAT_SEND_RECT = pygame.Rect(1353, 305, 64, 30)
DRIVE_MODE_RECT = pygame.Rect(24, 118, 247, 43)

MOUSE_LEFT_RECT = pygame.Rect(512, 798, 46, 55)
MOUSE_RIGHT_RECT = pygame.Rect(660, 798, 46, 55)
MOUSE_WHEEL_RECT = pygame.Rect(565, 759, 95, 124)
MOUSE_WHEEL_CENTER_X = 626
MOUSE_BRAKE_RECT = pygame.Rect(730, 788, 45, 74)
MOUSE_ACCEL_RECT = pygame.Rect(786, 788, 45, 74)

PAUSE_RECT = pygame.Rect(1164, 849, 57, 28)
RESTART_RECT = pygame.Rect(1226, 849, 67, 28)
SETTINGS_RECT = pygame.Rect(1298, 849, 75, 28)
QUIT_RECT = pygame.Rect(1378, 849, 44, 28)

INK = (232, 242, 249)
MUTED = (137, 161, 178)
CYAN = (44, 220, 242)
BLUE = (87, 183, 247)
GREEN = (42, 231, 89)
PANEL = (5, 14, 24)
PANEL_CARD = (8, 21, 34)
PANEL_BORDER = (28, 118, 148)
RISK_COLORS = {
    "LOW": (55, 227, 115),
    "MODERATE": (255, 209, 43),
    "HIGH": (255, 137, 61),
    "CRITICAL": (255, 66, 77),
}
TRAFFIC_COLORS = (
    (195, 221, 227),
    (73, 180, 198),
    (243, 184, 63),
    (155, 191, 177),
    (196, 167, 205),
    (218, 206, 179),
)


class GameRenderer:
    def __init__(self) -> None:
        self.title_font = pygame.font.SysFont("segoeui", 23, bold=True)
        self.heading_font = pygame.font.SysFont("segoeui", 17, bold=True)
        self.label_font = pygame.font.SysFont("segoeui", 12, bold=True)
        self.body_font = pygame.font.SysFont("segoeui", 14)
        self.small_font = pygame.font.SysFont("segoeui", 11)
        self.metric_font = pygame.font.SysFont("segoeui", 25, bold=True)
        self.speed_font = pygame.font.SysFont("segoeui", 42, bold=True)
        self.speed_history: list[tuple[float, float, float | None]] = []
        self._last_graph_sample = -1.0

    @staticmethod
    def _text(
        surface: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
        x: int,
        y: int,
    ) -> None:
        surface.blit(font.render(text, True, color), (x, y))

    @staticmethod
    def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
        lines: list[str] = []
        line = ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if line and font.size(candidate)[0] > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        return lines

    @staticmethod
    def _card(
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        fill: tuple[int, int, int] = PANEL_CARD,
        border: tuple[int, int, int] = PANEL_BORDER,
        radius: int = 11,
        glow: bool = False,
    ) -> None:
        if glow:
            halo_rect = rect.inflate(10, 8)
            halo = pygame.Surface(halo_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(halo, (*border, 35), halo.get_rect(), border_radius=radius + 5)
            surface.blit(halo, halo_rect.topleft)
        pygame.draw.rect(surface, fill, rect, border_radius=radius)
        pygame.draw.rect(surface, border, rect, width=1, border_radius=radius)

    def _draw_topbar(self, surface: pygame.Surface, selected_tab: str) -> None:
        pygame.draw.rect(surface, (4, 11, 19), (0, 0, WIDTH, TOPBAR_HEIGHT))
        pygame.draw.line(surface, (19, 78, 104), (0, TOPBAR_HEIGHT - 1), (WIDTH, TOPBAR_HEIGHT - 1), 1)
        pygame.draw.polygon(surface, CYAN, [(23, 46), (39, 11), (63, 11), (48, 47), (42, 47), (54, 30), (40, 30), (32, 47)])
        pygame.draw.line(surface, (41, 102, 125), (77, 10), (77, 53), 2)
        self._text(surface, self.title_font, "Ravan AI", INK, 91, 8)
        self._text(surface, self.body_font, "Driving Assistant", CYAN, 91, 35)

        live_rect = pygame.Rect(285, 15, 188, 34)
        self._card(surface, live_rect, fill=(6, 33, 22), border=(22, 154, 68), radius=12)
        pygame.draw.circle(surface, GREEN, (live_rect.x + 18, live_rect.centery), 6)
        self._text(surface, self.label_font, "LIVE SIMULATION", GREEN, live_rect.x + 32, live_rect.y + 11)

        for label, rect in zip(NAV_BUTTONS, NAV_BUTTON_RECTS):
            active = label == selected_tab
            fill = (7, 50, 73) if active else (7, 20, 32)
            border = CYAN if active else (53, 89, 106)
            self._card(surface, rect, fill=fill, border=border, radius=10, glow=active)
            color = CYAN if active else (210, 224, 234)
            label_surface = self.body_font.render(label, True, color)
            self._draw_nav_icon(surface, label, rect.x + 18, rect.centery, color)
            surface.blit(label_surface, (rect.x + 31, rect.y + 9))

        self._text(surface, self.small_font, "LOCAL ONLY", (98, 216, 167), 1115, 25)
        self._text(surface, self.body_font, time.strftime("%H:%M"), INK, 1350, 20)

    @staticmethod
    def _draw_nav_icon(surface: pygame.Surface, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        if label == "Drive":
            pygame.draw.rect(surface, color, (x - 8, y - 3, 16, 8), border_radius=3)
            pygame.draw.polygon(surface, color, [(x - 5, y - 3), (x - 2, y - 8), (x + 5, y - 8), (x + 8, y - 3)])
            pygame.draw.circle(surface, (7, 20, 32), (x - 4, y + 5), 2)
            pygame.draw.circle(surface, (7, 20, 32), (x + 4, y + 5), 2)
        elif label == "Analytics":
            for index, height in enumerate((8, 13, 18, 11)):
                pygame.draw.rect(surface, color, (x - 9 + index * 5, y + 8 - height, 3, height), border_radius=1)
        elif label == "AI Assistant":
            pygame.draw.circle(surface, color, (x, y - 3), 5, 2)
            pygame.draw.arc(surface, color, (x - 9, y + 1, 18, 13), 0.1, 3.05, 2)
        elif label == "Results":
            for index in range(3):
                pygame.draw.line(surface, color, (x - 8, y - 6 + index * 6), (x + 8, y - 6 + index * 6), 2)
        else:
            pygame.draw.circle(surface, color, (x, y), 7, 2)
            pygame.draw.circle(surface, color, (x, y), 2)
            for dx, dy in ((0, -10), (0, 10), (-10, 0), (10, 0)):
                pygame.draw.line(surface, color, (x + dx * .7, y + dy * .7), (x + dx, y + dy), 2)

    def _draw_drive_mode(self, surface: pygame.Surface, agent_drive: bool) -> None:
        rect = pygame.Rect(10, 70, 275, 144)
        self._card(surface, rect)
        self._text(surface, self.heading_font, "Driving Mode", INK, rect.x + 16, rect.y + 11)
        mode_rect = DRIVE_MODE_RECT
        accent = GREEN if agent_drive else CYAN
        fill = (7, 58, 39) if agent_drive else (8, 52, 70)
        self._card(surface, mode_rect, fill=fill, border=accent, radius=9, glow=agent_drive)
        pygame.draw.circle(surface, accent, (mode_rect.x + 22, mode_rect.centery), 8, 2)
        pygame.draw.line(surface, accent, (mode_rect.x + 22, mode_rect.centery - 6), (mode_rect.x + 22, mode_rect.centery + 5), 2)
        pygame.draw.line(surface, accent, (mode_rect.x + 14, mode_rect.centery + 2), (mode_rect.x + 30, mode_rect.centery + 2), 2)
        label = "AGENT DRIVE  ·  ON" if agent_drive else "MANUAL  ·  YOU DRIVE"
        self._text(surface, self.body_font, label, INK, mode_rect.x + 41, mode_rect.y + 12)
        detail = "Rule agent manages speed and steering" if agent_drive else "Tap to let the rule agent drive"
        self._text(surface, self.small_font, detail, MUTED, rect.x + 19, rect.y + 105)

    def _draw_vehicle_status(self, surface: pygame.Surface, simulation: DrivingSimulation) -> None:
        rect = pygame.Rect(10, 222, 275, 209)
        self._card(surface, rect)
        self._text(surface, self.heading_font, "Vehicle Status", INK, rect.x + 16, rect.y + 11)
        center = (rect.x + 87, rect.y + 117)
        gauge = pygame.Rect(center[0] - 65, center[1] - 65, 130, 130)
        pygame.draw.arc(surface, (37, 63, 83), gauge, 0.35, 2.79, 11)
        fraction = max(0.0, min(1.0, simulation.player.speed_mps * 3.6 / 120.0))
        pygame.draw.arc(surface, CYAN, gauge, 0.35, 0.35 + 2.44 * fraction, 11)
        self._text(surface, self.metric_font, f"{simulation.player.speed_mps * 3.6:.0f}", INK, center[0] - 32, center[1] - 22)
        self._text(surface, self.small_font, "km/h", CYAN, center[0] - 17, center[1] + 8)
        self._text(surface, self.small_font, "0", MUTED, center[0] - 58, center[1] + 48)
        self._text(surface, self.small_font, "120", MUTED, center[0] + 40, center[1] + 48)
        self._text(surface, self.small_font, "ACCEL.", MUTED, rect.x + 174, rect.y + 74)
        self._text(surface, self.body_font, f"{simulation.player.acceleration_mps2:+.1f}", INK, rect.x + 174, rect.y + 94)
        self._text(surface, self.small_font, "m/s²", MUTED, rect.x + 174, rect.y + 116)
        self._text(surface, self.small_font, "GEAR", MUTED, rect.x + 174, rect.y + 142)
        self._text(surface, self.body_font, "D", CYAN, rect.x + 174, rect.y + 158)

    def _draw_traffic_summary(self, surface: pygame.Surface, simulation: DrivingSimulation) -> None:
        rect = pygame.Rect(10, 439, 275, 131)
        self._card(surface, rect)
        self._text(surface, self.heading_font, "Traffic Summary", INK, rect.x + 16, rect.y + 11)
        lane = max(0, min(LANE_COUNT - 1, int(simulation.player.lateral_m / LANE_WIDTH_M)))
        lane_name = ("Left", "Middle", "Right")[lane]
        front = simulation.nearest_ahead
        if front is None:
            front_text = "Clear"
        else:
            gap = max(0.0, front.longitudinal_m - simulation.player.longitudinal_m
                      - (front.length_m + simulation.player.length_m) / 2.0)
            front_text = f"{front.speed_mps * 3.6:.0f} km/h · {gap:.0f} m"
        rows = [
            ("V", "Vehicles Detected", str(len(simulation.traffic))),
            ("L", "Current Lane", f"{lane + 1} ({lane_name})"),
            ("D", "Traffic Density", simulation.traffic_density),
            ("F", "Nearest Front", front_text),
        ]
        for index, (icon, label, value) in enumerate(rows):
            y = rect.y + 43 + index * 22
            self._text(surface, self.label_font, icon, CYAN, rect.x + 16, y + 1)
            self._text(surface, self.small_font, label, (201, 214, 225), rect.x + 37, y + 2)
            value_surface = self.small_font.render(value, True, (255, 211, 58) if index == 2 else INK)
            surface.blit(value_surface, (rect.right - 12 - value_surface.get_width(), y + 2))

    def _draw_traffic_header(self, surface: pygame.Surface) -> None:
        badge = pygame.Rect(472, 81, 334, 29)
        self._card(surface, badge, fill=(5, 39, 65), border=(34, 145, 191), radius=10)
        label = self.small_font.render("CONTINUOUS TRAFFIC  ·  YOU CONTROL THE CAR", True, INK)
        surface.blit(label, label.get_rect(center=badge.center))

    @staticmethod
    def _road_half_width(y: float, horizon: float, height: float, width: float) -> float:
        depth = max(0.0, min(1.0, (y - horizon) / (height - horizon)))
        return 42.0 + (width * 0.465 - 42.0) * depth**0.92

    def _draw_car(
        self,
        surface: pygame.Surface,
        center_x: int,
        center_y: int,
        scale: float,
        color: tuple[int, int, int],
        outline: tuple[int, int, int],
        *,
        player: bool = False,
    ) -> pygame.Rect:
        car_w = max(15, int((68 if player else 59) * scale))
        car_h = max(23, int((82 if player else 77) * scale))
        body = pygame.Rect(center_x - car_w // 2, center_y - car_h // 2, car_w, car_h)
        if not player:
            box = body.inflate(max(8, int(car_w * 0.38)), max(9, int(car_h * 0.27)))
            pygame.draw.rect(surface, outline, box, width=max(1, int(scale * 2.2)), border_radius=max(3, int(9 * scale)))
        shadow = body.inflate(max(6, int(car_w * .2)), max(4, int(car_h * .1))).move(0, max(2, int(4 * scale)))
        pygame.draw.ellipse(surface, (9, 17, 22), shadow)
        pygame.draw.polygon(surface, color, [
            (body.x + int(car_w * .2), body.y + int(car_h * .1)),
            (body.x + int(car_w * .8), body.y + int(car_h * .1)),
            (body.right, body.y + int(car_h * .38)),
            (body.right - int(car_w * .05), body.bottom - int(car_h * .08)),
            (body.x + int(car_w * .05), body.bottom - int(car_h * .08)),
            (body.x, body.y + int(car_h * .38)),
        ])
        window = pygame.Rect(body.x + int(car_w * .17), body.y + int(car_h * .18), max(4, int(car_w * .66)), max(4, int(car_h * .19)))
        pygame.draw.rect(surface, (29, 58, 73), window, border_radius=max(2, int(5 * scale)))
        rear_window = pygame.Rect(body.x + int(car_w * .21), body.y + int(car_h * .51), max(4, int(car_w * .58)), max(4, int(car_h * .13)))
        pygame.draw.rect(surface, (35, 53, 64), rear_window, border_radius=max(2, int(4 * scale)))
        lamp = (255, 67, 63) if player else (255, 98, 72)
        lamp_h = max(2, int(car_h * .045))
        pygame.draw.rect(surface, lamp, (body.x + 2, body.bottom - int(car_h * .18), max(4, int(car_w * .2)), lamp_h), border_radius=2)
        pygame.draw.rect(surface, lamp, (body.right - int(car_w * .22), body.bottom - int(car_h * .18), max(4, int(car_w * .2)), lamp_h), border_radius=2)
        if player:
            glow_rect = body.inflate(38, 30)
            glow = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (10, 188, 247, 34), glow.get_rect())
            surface.blit(glow, glow_rect.topleft)
        return body

    def _draw_road_scene(self, surface: pygame.Surface, simulation: DrivingSimulation, assessment: RiskAssessment) -> None:
        width, height = surface.get_size()
        horizon = int(height * .22)
        center = width // 2
        for y in range(horizon + 15):
            mix = y / max(1, horizon + 15)
            pygame.draw.line(surface, (int(47 + 48 * mix), int(137 + 58 * mix), int(192 + 30 * mix)), (0, y), (width, y))
        pygame.draw.circle(surface, (255, 232, 170), (int(width * .82), int(horizon * .45)), max(15, int(horizon * .22)))
        pygame.draw.polygon(surface, (65, 125, 130), [(0, horizon + 25), (width * .16, horizon - 22), (width * .32, horizon + 25), (width * .52, horizon - 30), (width * .73, horizon + 30), (width * .91, horizon - 15), (width, horizon + 17), (width, height), (0, height)])
        pygame.draw.polygon(surface, (40, 115, 79), [(0, horizon + 50), (width * .18, horizon + 10), (width * .36, horizon + 54), (width * .55, horizon + 6), (width * .76, horizon + 53), (width * .92, horizon + 13), (width, horizon + 51), (width, height), (0, height)])
        pygame.draw.rect(surface, (35, 105, 68), (0, horizon + 43, width, height - horizon - 43))

        top_half = 42.0
        bottom_half = width * .465
        road_points = [
            (int(center - top_half), horizon),
            (int(center + top_half), horizon),
            (int(center + bottom_half), height),
            (int(center - bottom_half), height),
        ]
        shoulder_points = [
            (int(center - top_half - 8), horizon),
            (int(center + top_half + 8), horizon),
            (int(center + bottom_half + 24), height),
            (int(center - bottom_half - 24), height),
        ]
        pygame.draw.polygon(surface, (166, 176, 170), shoulder_points)
        pygame.draw.polygon(surface, (39, 48, 59), road_points)

        # Rails converge toward the horizon; thin trees add depth without image assets.
        for side in (-1, 1):
            previous = None
            for y in range(horizon + 4, height, 12):
                half = self._road_half_width(y, horizon, height, width) + 17
                point = (int(center + side * half), y)
                if previous is not None:
                    pygame.draw.line(surface, (189, 201, 191), previous, point, max(1, int((y - horizon) / 120)))
                    pygame.draw.line(surface, (89, 107, 101), (previous[0], previous[1] + 5), (point[0], point[1] + 5), 2)
                previous = point
            for index in range(10):
                y = horizon + 35 + index * 31
                half = self._road_half_width(y, horizon, height, width) + 45 + (index % 2) * 18
                x = int(center + side * half)
                radius = max(3, int((y - horizon) * .08))
                pygame.draw.circle(surface, (31, 90, 58), (x, y + radius), radius)

        phase = int((simulation.player.longitudinal_m * 3.2) % 64)
        for lane_boundary in range(1, LANE_COUNT):
            fraction = lane_boundary / LANE_COUNT
            for y in range(horizon + 8 + phase, height, 64):
                end_y = min(height, y + max(5, int((y - horizon) * .11)))
                half_start = self._road_half_width(y, horizon, height, width)
                half_end = self._road_half_width(end_y, horizon, height, width)
                x1 = int(center + (fraction * 2 - 1) * half_start)
                x2 = int(center + (fraction * 2 - 1) * half_end)
                pygame.draw.line(surface, (228, 233, 229), (x1, y), (x2, end_y), max(1, int((y - horizon) / 85)))

        # A restrained cyan guidance line connects the player to the selected lane.
        player_lane_center = (int(simulation.player.lateral_m / LANE_WIDTH_M) + .5) / LANE_COUNT
        top_x = int(center + (player_lane_center * 2 - 1) * top_half)
        bottom_x = int(center + (player_lane_center * 2 - 1) * bottom_half)
        trail = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.line(trail, (19, 205, 249, 36), (top_x, horizon + 15), (bottom_x, height - 30), max(4, int(width * .025)))
        pygame.draw.line(trail, (35, 222, 252, 178), (top_x, horizon + 15), (bottom_x, height - 30), max(1, int(width * .004)))
        surface.blit(trail, (0, 0))

        # Forward traffic is projected from actual relative position and lane offset.
        for index, agent in enumerate(simulation.traffic):
            vehicle = agent.vehicle
            longitudinal = vehicle.longitudinal_m - simulation.player.longitudinal_m
            if longitudinal < -1.5 or longitudinal > 190:
                continue
            perspective = 62.0 / (max(0.0, longitudinal) + 62.0)
            y = horizon + (height - horizon - 54) * perspective
            half = self._road_half_width(y, horizon, height, width)
            lateral = vehicle.lateral_m - simulation.player.lateral_m
            x = int(center + lateral * (half / (ROAD_WIDTH_M / 2.0)))
            scale = max(.17, min(.95, perspective * 1.35))
            current_front = assessment.nearest_front or {}
            high = vehicle.name == current_front.get("name") and assessment.risk_level in {"HIGH", "CRITICAL"}
            outline = RISK_COLORS[assessment.risk_level] if high else CYAN
            body = self._draw_car(surface, x, int(y), scale, TRAFFIC_COLORS[index % len(TRAFFIC_COLORS)], outline)
            if longitudinal < 75:
                gap = max(0.0, longitudinal - (vehicle.length_m + simulation.player.length_m) / 2)
                tag = f"{vehicle.speed_mps * 3.6:.0f} km/h  |  {gap:.0f} m"
                text = self.small_font.render(tag, True, outline)
                surface.blit(text, text.get_rect(center=(x, body.top - 13)))
                behavior = self.small_font.render(vehicle.behavior_state.replace("_", " ").upper(), True, INK)
                if abs(longitudinal) < 45 or vehicle.behavior_state in {"braking", "merging"}:
                    surface.blit(behavior, behavior.get_rect(center=(x, body.top - 1)))

        # The player's car sits low in the road window, as in a third-person driver view.
        player_lane_offset = simulation.player.lateral_m - ROAD_WIDTH_M / 2
        player_x = int(center + player_lane_offset * (bottom_half / (ROAD_WIDTH_M / 2)))
        self._draw_car(surface, player_x, height - 43, 1.17, (32, 143, 255), CYAN, player=True)

        if assessment.risk_level in {"HIGH", "CRITICAL"}:
            banner = pygame.Rect(12, 11, 196, 27)
            pygame.draw.rect(surface, (62, 15, 24), banner, border_radius=7)
            pygame.draw.rect(surface, RISK_COLORS[assessment.risk_level], banner, 1, border_radius=7)
            self._text(surface, self.small_font, f"!  {assessment.risk_level} COLLISION RISK", RISK_COLORS[assessment.risk_level], banner.x + 11, banner.y + 7)

    def _draw_scene_card(
        self,
        surface: pygame.Surface,
        simulation: DrivingSimulation,
        assessment: RiskAssessment,
        paused: bool,
        started: bool,
    ) -> None:
        rect = pygame.Rect(294, 70, 690, 500)
        self._card(surface, rect, fill=(4, 12, 20), border=(23, 133, 174), radius=14, glow=True)
        self._draw_traffic_header(surface)
        view_rect = pygame.Rect(303, 119, 672, 441)
        road = pygame.Surface(view_rect.size)
        self._draw_road_scene(road, simulation, assessment)
        surface.blit(road, view_rect.topleft)
        pygame.draw.rect(surface, (50, 154, 183), view_rect, width=1, border_radius=6)
        if paused:
            overlay_rect = pygame.Rect(view_rect.centerx - 157, view_rect.centery - 43, 314, 86)
            overlay = pygame.Surface(overlay_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(overlay, (4, 12, 20, 225), overlay.get_rect(), border_radius=13)
            pygame.draw.rect(overlay, (65, 204, 229, 190), overlay.get_rect(), width=1, border_radius=13)
            surface.blit(overlay, overlay_rect.topleft)
            title = (
                "CONTACT  ·  RESTART REQUIRED"
                if assessment.collision
                else ("READY TO DRIVE" if not started else "SIMULATION PAUSED")
            )
            title_color = RISK_COLORS["CRITICAL"] if assessment.collision else CYAN
            title_surface = self.heading_font.render(title, True, title_color)
            surface.blit(title_surface, title_surface.get_rect(center=(overlay_rect.centerx, overlay_rect.y + 29)))
            hint = (
                "Click RESTART or press R to drive again"
                if assessment.collision
                else ("Click START or press W to begin" if not started else "Click RESUME or press P")
            )
            hint_surface = self.small_font.render(hint, True, INK)
            surface.blit(hint_surface, hint_surface.get_rect(center=(overlay_rect.centerx, overlay_rect.y + 58)))
        scene_title = self.small_font.render("3-LANE HIGHWAY  |  LIVE AI TRAFFIC", True, INK)
        title_rect = scene_title.get_rect(center=(view_rect.centerx, view_rect.y + 13))
        surface.blit(scene_title, title_rect)
        if assessment.nearest_rear:
            mirror = pygame.Rect(view_rect.right - 140, view_rect.y + 12, 124, 31)
            pygame.draw.rect(surface, (8, 22, 31), mirror, border_radius=9)
            pygame.draw.rect(surface, (44, 126, 151), mirror, 1, border_radius=9)
            self._text(surface, self.small_font, f"REAR  {assessment.nearest_rear['speed_kmh']:.0f} km/h", CYAN, mirror.x + 10, mirror.y + 10)
        elif assessment.nearby_vehicles:
            mirror = pygame.Rect(view_rect.right - 140, view_rect.y + 12, 124, 31)
            pygame.draw.rect(surface, (8, 22, 31), mirror, border_radius=9)
            pygame.draw.rect(surface, (44, 126, 151), mirror, 1, border_radius=9)
            self._text(surface, self.small_font, "REAR  CLEAR", (118, 221, 173), mirror.x + 18, mirror.y + 10)

    def _draw_avatar(self, surface: pygame.Surface, rect: pygame.Rect, listening: bool, elapsed_s: float) -> None:
        center = rect.centerx
        glow = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for radius, alpha in ((64, 10), (51, 18), (39, 22)):
            pygame.draw.ellipse(glow, (13, 191, 243, alpha), (center - radius, rect.y + 12, radius * 2, radius * 1.65))
        surface.blit(glow, (0, 0))
        color = (67, 224, 246) if listening else (27, 168, 210)
        pygame.draw.ellipse(surface, (12, 36, 51), (center - 38, rect.y + 18, 76, 94))
        pygame.draw.ellipse(surface, color, (center - 35, rect.y + 21, 70, 82), 2)
        # Friendly, abstract holographic guide avatar built entirely from Pygame shapes.
        pygame.draw.polygon(surface, (16, 37, 54), [(center - 46, rect.y + 65), (center - 34, rect.y + 25), (center - 12, rect.y + 12), (center + 23, rect.y + 20), (center + 39, rect.y + 49), (center + 35, rect.y + 79), (center + 20, rect.y + 53), (center - 18, rect.y + 51), (center - 34, rect.y + 76)])
        pygame.draw.ellipse(surface, (207, 231, 235), (center - 24, rect.y + 37, 48, 57))
        pygame.draw.polygon(surface, (183, 216, 223), [(center - 16, rect.y + 83), (center, rect.y + 104), (center + 16, rect.y + 83), (center + 43, rect.y + 120), (center - 43, rect.y + 120)])
        pygame.draw.line(surface, (25, 112, 141), (center - 42, rect.y + 115), (center + 42, rect.y + 115), 4)
        pygame.draw.circle(surface, (17, 40, 56), (center - 9, rect.y + 61), 3)
        pygame.draw.circle(surface, (17, 40, 56), (center + 9, rect.y + 61), 3)
        pygame.draw.arc(surface, (18, 90, 106), (center - 8, rect.y + 66, 16, 10), 3.4, 5.9, 2)
        self._text(surface, self.small_font, "R", CYAN, center - 5, rect.y + 99)
        if listening:
            baseline = rect.bottom - 2
            for index in range(13):
                amp = 6 + int(15 * abs(sin(elapsed_s * 5.5 + index * .72)))
                pygame.draw.line(surface, color, (rect.x + 13 + index * 10, baseline), (rect.x + 13 + index * 10, baseline - amp), 2)

    def _situation_text(self, assessment: RiskAssessment, live_guidance: str) -> str:
        if live_guidance:
            return live_guidance
        front = assessment.nearest_front
        if front:
            reason = assessment.main_factors[0] if assessment.main_factors else "Traffic is moving in your lane."
            return (
                f"You are at {assessment.ego_speed_kmh:.0f} km/h. The front vehicle is "
                f"{front['distance_m']:.0f} m away at {front['speed_kmh']:.0f} km/h. "
                f"{assessment.risk_level} risk: {reason} {assessment.recommendation}"
            )
        return (
            f"You are at {assessment.ego_speed_kmh:.0f} km/h. No same-lane front vehicle is detected. "
            f"Risk is {assessment.risk_level}. {assessment.recommendation}"
        )

    def _draw_assistant_panel(
        self,
        surface: pygame.Surface,
        assessment: RiskAssessment,
        live_guidance: str,
        learning_hint: str,
        chat_input: str,
        chat_focused: bool,
        ai_pending: bool,
        model_name: str,
        status_message: str,
        voice_enabled: bool,
        mic_listening: bool,
        simulation: DrivingSimulation,
    ) -> None:
        rect = pygame.Rect(993, 70, 437, 272)
        self._card(surface, rect, border=(15, 164, 204), radius=14, glow=True)
        for index in range(4):
            h = 9 + index * 5 if mic_listening else 14
            pygame.draw.line(surface, CYAN, (rect.x + 18 + index * 7, rect.y + 24 - h // 2), (rect.x + 18 + index * 7, rect.y + 24 + h // 2), 3)
        self._text(surface, self.heading_font, "Ravan AI Assistant", INK, rect.x + 55, rect.y + 11)
        self._text(surface, self.small_font, f"LOCAL OLLAMA  ·  {model_name}", (105, 191, 214), rect.x + 55, rect.y + 34)
        state_label = "LISTENING" if mic_listening else ("THINKING" if ai_pending else ("VOICE ON" if voice_enabled else "MUTED"))
        state_color = CYAN if ai_pending and not mic_listening else (GREEN if mic_listening or voice_enabled else MUTED)
        pill = pygame.Rect(rect.right - 93, rect.y + 13, 79, 24)
        self._card(surface, pill, fill=(4, 40, 26) if voice_enabled else (21, 30, 40), border=state_color, radius=11)
        pygame.draw.circle(surface, state_color, (pill.x + 12, pill.centery), 4)
        self._text(surface, self.small_font, state_label, state_color, pill.x + 21, pill.y + 7)

        avatar_rect = pygame.Rect(rect.x + 12, rect.y + 58, 116, 142)
        self._draw_avatar(surface, avatar_rect, mic_listening, simulation.elapsed_s)
        bubble = pygame.Rect(rect.x + 135, rect.y + 58, 285, 143)
        self._card(surface, bubble, fill=(8, 36, 52), border=(17, 164, 203), radius=12)
        self._text(surface, self.label_font, "CURRENT SITUATION", CYAN, bubble.x + 13, bubble.y + 10)
        situation = self._situation_text(assessment, live_guidance)
        for index, line in enumerate(self._wrap(situation, self.body_font, bubble.width - 25)[:4]):
            self._text(surface, self.body_font, line, INK, bubble.x + 13, bubble.y + 33 + index * 19)
        learning_line = learning_hint if len(learning_hint) <= 49 else learning_hint[:46] + "..."
        self._text(surface, self.small_font, learning_line, CYAN, bubble.x + 13, bubble.bottom - 17)

        if status_message:
            short_status = status_message if len(status_message) < 57 else status_message[:54] + "..."
            self._text(surface, self.small_font, short_status, MUTED, rect.x + 15, rect.y + 211)
        pygame.draw.rect(surface, (4, 12, 19), CHAT_INPUT_RECT, border_radius=7)
        pygame.draw.rect(surface, CYAN if chat_focused else (51, 94, 113), CHAT_INPUT_RECT, width=1, border_radius=7)
        displayed = chat_input[-27:] or ("Ask about this traffic..." if not chat_focused else "")
        self._text(surface, self.small_font, displayed, INK if chat_input else MUTED, CHAT_INPUT_RECT.x + 8, CHAT_INPUT_RECT.y + 10)
        for button, label, enabled in (
            (CHAT_MIC_RECT, "LIVE" if mic_listening else "MIC", not mic_listening),
            (CHAT_VOICE_RECT, "VOICE" if voice_enabled else "MUTE", True),
            (CHAT_SEND_RECT, "WAIT" if ai_pending else "ASK AI", not ai_pending),
        ):
            fill = (7, 83, 102) if enabled else (47, 56, 63)
            pygame.draw.rect(surface, fill, button, border_radius=7)
            rendered = self.small_font.render(label, True, INK)
            surface.blit(rendered, rendered.get_rect(center=button.center))

    def _draw_recommendation(self, surface: pygame.Surface, assessment: RiskAssessment) -> None:
        rect = pygame.Rect(993, 350, 437, 82)
        color = RISK_COLORS[assessment.risk_level]
        fill = (46, 18, 26) if assessment.risk_level in {"HIGH", "CRITICAL"} else (8, 28, 39)
        self._card(surface, rect, fill=fill, border=color, radius=11, glow=assessment.risk_level in {"HIGH", "CRITICAL"})
        pygame.draw.polygon(surface, color, [(rect.x + 18, rect.y + 17), (rect.x + 31, rect.y + 43), (rect.x + 5, rect.y + 43)])
        self._text(surface, self.small_font, "RECOMMENDED ACTION", INK, rect.x + 42, rect.y + 12)
        for index, line in enumerate(self._wrap(assessment.recommendation, self.body_font, rect.width - 57)[:2]):
            self._text(surface, self.body_font, line, color, rect.x + 42, rect.y + 33 + index * 18)

    def _draw_next_steps(
        self,
        surface: pygame.Surface,
        assessment: RiskAssessment,
        simulation: DrivingSimulation,
        lane_plan: LanePathPlan | None = None,
    ) -> None:
        rect = pygame.Rect(993, 441, 437, 129)
        self._card(surface, rect, radius=11)
        self._text(surface, self.heading_font, "Next Steps", INK, rect.x + 15, rect.y + 10)
        front = assessment.nearest_front
        if assessment.risk_level in {"HIGH", "CRITICAL"} and front:
            target = max(0, round(front["speed_kmh"] - 3))
            first = f"Gently brake toward about {target} km/h."
        elif front and assessment.distance_m is not None:
            safe_gap = max(20, round(simulation.player.speed_mps * 1.7))
            first = f"Keep a following gap of at least {safe_gap} m."
        else:
            first = "Maintain a steady speed and scan the road ahead."
        safe_gap = max(20, round(simulation.player.speed_mps * 1.7))
        if front and assessment.distance_m is not None and assessment.distance_m < safe_gap:
            second = f"Increase your gap toward {safe_gap} m."
        else:
            second = "Keep a safe following distance."
        if lane_plan is not None and lane_plan.first_action == "stay_and_reduce_speed":
            third = "RL: hold lane and reduce speed; recheck the route ahead."
        elif lane_plan is not None and lane_plan.first_action.startswith("change_"):
            direction = "left" if "left" in lane_plan.first_action else "right"
            third = f"RL path: change {direction} when clear · {lane_plan.display_path}"
        elif lane_plan is not None:
            third = f"RL path ({lane_plan.horizon_distance_m:.0f} m): {lane_plan.display_path}"
        elif assessment.lane_change_left.safe or assessment.lane_change_right.safe:
            clear_side = "left" if assessment.lane_change_left.safe else "right"
            third = f"{clear_side.title()} lane checked clear; move only if needed."
        else:
            third = "Keep your lane; adjacent gaps need attention."
        steps = (first, second, third)
        for index, text in enumerate(steps):
            y = rect.y + 43 + index * 25
            pygame.draw.circle(surface, CYAN, (rect.x + 23, y + 7), 9)
            number = self.small_font.render(str(index + 1), True, (4, 18, 28))
            surface.blit(number, number.get_rect(center=(rect.x + 23, y + 7)))
            font = self.small_font if index == 2 and lane_plan is not None else self.body_font
            self._text(surface, font, text, INK, rect.x + 43, y)

    @staticmethod
    def _metric_bar(surface: pygame.Surface, x: int, y: int, width: int, fraction: float, color: tuple[int, int, int]) -> None:
        count = 8
        gap = 3
        segment_width = (width - gap * (count - 1)) // count
        lit = max(0, min(count, round(fraction * count)))
        for index in range(count):
            segment = pygame.Rect(x + index * (segment_width + gap), y, segment_width, 6)
            fill = color if index < lit else (39, 53, 66)
            pygame.draw.rect(surface, fill, segment, border_radius=3)

    def _draw_risk_cards(self, surface: pygame.Surface, assessment: RiskAssessment) -> None:
        cards = [
            ("COLLISION RISK", assessment.risk_level, f"Score  {assessment.risk_score} / 100", RISK_COLORS[assessment.risk_level], assessment.risk_score / 100),
            ("TTC", "—" if assessment.ttc_s is None else f"{assessment.ttc_s:.2f} s", "Time to collision", RISK_COLORS["MODERATE"], 0 if assessment.ttc_s is None else max(0, 1 - assessment.ttc_s / 5)),
            ("FRONT GAP", "—" if assessment.distance_m is None else f"{assessment.distance_m:.1f} m", "Nearest same-lane gap", CYAN, 0 if assessment.distance_m is None else min(1, assessment.distance_m / 50)),
            ("REQUIRED DECEL.", "—" if assessment.drac_mps2 is None else f"{assessment.drac_mps2:.2f} m/s²", "DRAC", BLUE, 0 if assessment.drac_mps2 is None else min(1, assessment.drac_mps2 / 8)),
            ("LEFT LANE", "SAFE" if assessment.lane_change_left.safe else "WAIT", assessment.lane_change_left.label.title(), GREEN if assessment.lane_change_left.safe else RISK_COLORS["HIGH"], 1 if assessment.lane_change_left.safe else .35),
            ("RIGHT LANE", "SAFE" if assessment.lane_change_right.safe else "WAIT", assessment.lane_change_right.label.title(), GREEN if assessment.lane_change_right.safe else RISK_COLORS["HIGH"], 1 if assessment.lane_change_right.safe else .35),
        ]
        start_x, gap, width = 10, 8, 230
        for index, (label, value, detail, color, fraction) in enumerate(cards):
            rect = pygame.Rect(start_x + index * (width + gap), 580, width, 124)
            self._card(surface, rect, border=color if index == 0 else (35, 93, 117), radius=11, glow=index == 0 and assessment.risk_level in {"HIGH", "CRITICAL"})
            self._text(surface, self.label_font, label, MUTED, rect.x + 13, rect.y + 12)
            value_font = self.heading_font if len(value) > 12 else self.metric_font
            self._text(surface, value_font, value, color, rect.x + 13, rect.y + 36)
            self._text(surface, self.small_font, detail, INK, rect.x + 13, rect.y + 76)
            self._metric_bar(surface, rect.x + 13, rect.y + 101, rect.width - 26, fraction, color)

    def _draw_keyboard_controls(self, surface: pygame.Surface) -> None:
        rect = pygame.Rect(10, 714, 455, 176)
        self._card(surface, rect)
        pygame.draw.circle(surface, GREEN, (rect.x + 25, rect.y + 21), 7)
        self._text(surface, self.heading_font, "Controls (You Drive)", INK, rect.x + 43, rect.y + 10)
        keys = [
            ("W / ↑", "Accelerate"),
            ("S / ↓", "Brake"),
            ("A / D", "Steer left/right"),
            ("SPACE", "Handbrake"),
        ]
        card_width = 99
        for index, (key, label) in enumerate(keys):
            button = pygame.Rect(rect.x + 17 + index * 106, rect.y + 53, card_width, 67)
            self._card(surface, button, fill=(9, 27, 42), border=(54, 102, 125), radius=9)
            key_surface = self.heading_font.render(key, True, INK)
            surface.blit(key_surface, key_surface.get_rect(center=(button.centerx, button.y + 23)))
            label_surface = self.small_font.render(label, True, (193, 208, 219))
            surface.blit(label_surface, label_surface.get_rect(center=(button.centerx, button.y + 50)))
        self._text(surface, self.small_font, "P Pause   ·   R Restart   ·   Esc Quit", MUTED, rect.x + 17, rect.y + 141)

    def _draw_control_panel(self, surface: pygame.Surface, active_action: str | None) -> None:
        rect = pygame.Rect(473, 714, 375, 176)
        self._card(surface, rect)
        self._text(surface, self.heading_font, "Control Panel", INK, rect.x + 16, rect.y + 10)
        wheel_center = (rect.x + 153, rect.y + 107)
        wheel_active = active_action in {"left", "right"}
        wheel_color = CYAN if wheel_active else (42, 70, 88)
        pygame.draw.circle(surface, (9, 18, 28), wheel_center, 61)
        pygame.draw.circle(surface, wheel_color, wheel_center, 57, width=12)
        pygame.draw.circle(surface, CYAN if wheel_active else (64, 222, 242), wheel_center, 30, width=2)
        pygame.draw.line(surface, (55, 92, 108), (wheel_center[0], wheel_center[1] - 31), (wheel_center[0], wheel_center[1] + 43), 10)
        pygame.draw.line(surface, (55, 92, 108), (wheel_center[0] - 43, wheel_center[1] + 12), (wheel_center[0] + 43, wheel_center[1] + 12), 10)
        self._text(surface, self.small_font, "R", CYAN, wheel_center[0] - 4, wheel_center[1] - 5)

        for button, label, direction, action in (
            (MOUSE_LEFT_RECT, "LEFT", -1, "left"),
            (MOUSE_RIGHT_RECT, "RIGHT", 1, "right"),
        ):
            active = active_action == action
            self._card(surface, button, fill=(8, 107, 127) if active else (7, 49, 68), border=(114, 244, 255) if active else CYAN, radius=8, glow=active)
            cx, cy = button.centerx, button.y + 18
            points = [(cx + direction * 8, cy), (cx - direction * 6, cy - 7), (cx - direction * 6, cy + 7)]
            pygame.draw.polygon(surface, CYAN, points)
            hint = self.small_font.render(label, True, INK)
            surface.blit(hint, hint.get_rect(center=(button.centerx, button.y + 41)))

        for button, label, pedal, action in (
            (MOUSE_BRAKE_RECT, "BRAKE", "brake", "brake"),
            (MOUSE_ACCEL_RECT, "ACCEL", "accel", "accelerate"),
        ):
            active = active_action == action
            self._card(surface, button, fill=(104, 52, 37) if active and action == "brake" else ((8, 92, 75) if active else (14, 31, 43)), border=(255, 145, 87) if active and action == "brake" else ((94, 245, 178) if active else (79, 124, 143)), radius=8, glow=active)
            if pedal == "brake":
                for row in range(3):
                    for col in range(3):
                        pygame.draw.circle(surface, CYAN, (button.x + 12 + col * 10, button.y + 14 + row * 9), 2)
            else:
                for row in range(3):
                    pygame.draw.line(surface, CYAN, (button.x + 12 + row * 9, button.y + 12), (button.x + 12 + row * 9, button.y + 37), 3)
            hint = self.small_font.render(label, True, INK)
            surface.blit(hint, hint.get_rect(center=(button.centerx, button.y + 57)))

    def _update_graph(self, simulation: DrivingSimulation, assessment: RiskAssessment) -> None:
        now = simulation.elapsed_s
        if self._last_graph_sample >= 0 and now < self._last_graph_sample:
            self.speed_history.clear()
            self._last_graph_sample = -1.0
        if self._last_graph_sample < 0 or now - self._last_graph_sample >= .25:
            self.speed_history.append((now, simulation.player.speed_mps * 3.6, assessment.distance_m))
            self._last_graph_sample = now
        self.speed_history = [item for item in self.speed_history if now - item[0] <= 10.0]

    def _draw_graph_and_controls(self, surface: pygame.Surface, simulation: DrivingSimulation, assessment: RiskAssessment, paused: bool, started: bool, status_message: str) -> None:
        rect = pygame.Rect(856, 714, 574, 176)
        self._card(surface, rect)
        self._text(surface, self.heading_font, "Speed & Gap", INK, rect.x + 15, rect.y + 9)
        self._text(surface, self.small_font, "Last 10 seconds", MUTED, rect.x + 117, rect.y + 14)
        pygame.draw.line(surface, CYAN, (rect.x + 372, rect.y + 18), (rect.x + 389, rect.y + 18), 3)
        self._text(surface, self.small_font, "Your speed", INK, rect.x + 394, rect.y + 12)
        pygame.draw.line(surface, GREEN, (rect.x + 470, rect.y + 18), (rect.x + 486, rect.y + 18), 3)
        self._text(surface, self.small_font, "Front gap", INK, rect.x + 491, rect.y + 12)

        graph = pygame.Rect(rect.x + 41, rect.y + 39, rect.width - 66, 74)
        for index in range(5):
            y = graph.y + index * graph.height // 4
            pygame.draw.line(surface, (29, 56, 72), (graph.x, y), (graph.right, y), 1)
            self._text(surface, self.small_font, str(80 - index * 20), MUTED, rect.x + 9, y - 6)
        for index in range(6):
            x = graph.x + index * graph.width // 5
            pygame.draw.line(surface, (24, 48, 64), (x, graph.y), (x, graph.bottom), 1)
            if index in (0, 2, 4, 5):
                self._text(surface, self.small_font, f"{index * 2}s", MUTED, x - 9, graph.bottom + 1)

        self._update_graph(simulation, assessment)
        if len(self.speed_history) > 1:
            speed_points = []
            gap_points = []
            first_time = self.speed_history[0][0]
            span = max(.01, self.speed_history[-1][0] - first_time)
            for sample_time, speed, gap_value in self.speed_history:
                x = int(graph.x + (sample_time - first_time) / max(10.0, span) * graph.width)
                speed_y = int(graph.bottom - min(80, max(0, speed)) / 80 * graph.height)
                speed_points.append((x, speed_y))
                if gap_value is not None:
                    gap_y = int(graph.bottom - min(80, max(0, gap_value)) / 80 * graph.height)
                    gap_points.append((x, gap_y))
            if len(speed_points) > 1:
                pygame.draw.lines(surface, CYAN, False, speed_points, 3)
            if len(gap_points) > 1:
                pygame.draw.lines(surface, GREEN, False, gap_points, 3)

        controls_y = 849
        for button, text in (
            (PAUSE_RECT, "RESTART" if paused and assessment.collision else ("START" if not started else ("RESUME" if paused else "PAUSE"))),
            (RESTART_RECT, "RESTART"),
            (SETTINGS_RECT, "SETTINGS"),
            (QUIT_RECT, "QUIT"),
        ):
            self._card(surface, button, fill=(9, 29, 43), border=(57, 113, 137), radius=7)
            rendered = self.small_font.render(text, True, INK)
            surface.blit(rendered, rendered.get_rect(center=button.center))
        if status_message:
            state = "LIVE" if started and not paused else ("READY" if not started else "PAUSED")
            state_color = (120, 223, 165) if state == "LIVE" else (255, 208, 111)
            self._text(surface, self.small_font, state, state_color, rect.x + 18, controls_y + 8)

    def draw(
        self,
        surface: pygame.Surface,
        simulation: DrivingSimulation,
        assessment: RiskAssessment,
        *,
        paused: bool,
        started: bool,
        chat_history: list[tuple[str, str]],
        chat_input: str,
        chat_focused: bool,
        ai_pending: bool,
        model_name: str,
        status_message: str = "",
        voice_enabled: bool = True,
        mic_listening: bool = False,
        selected_tab: str = "Drive",
        mouse_control_action: str | None = None,
        agent_drive: bool = False,
        live_guidance: str = "",
        learning_hint: str = "Trial learning is comparing your controls with risk changes.",
        lane_plan: LanePathPlan | None = None,
    ) -> None:
        surface.fill(PANEL)
        self._draw_topbar(surface, selected_tab)
        self._draw_drive_mode(surface, agent_drive)
        self._draw_vehicle_status(surface, simulation)
        self._draw_traffic_summary(surface, simulation)
        self._draw_scene_card(surface, simulation, assessment, paused, started)
        self._draw_assistant_panel(
            surface,
            assessment,
            live_guidance,
            learning_hint,
            chat_input,
            chat_focused,
            ai_pending,
            model_name,
            status_message,
            voice_enabled,
            mic_listening,
            simulation,
        )
        self._draw_recommendation(surface, assessment)
        self._draw_next_steps(surface, assessment, simulation, lane_plan)
        self._draw_risk_cards(surface, assessment)
        self._draw_keyboard_controls(surface)
        self._draw_control_panel(surface, mouse_control_action)
        self._draw_graph_and_controls(surface, simulation, assessment, paused, started, status_message)

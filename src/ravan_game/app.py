"""Pygame game loop, player controls, continuous traffic, and local AI chat."""

from __future__ import annotations

import pygame

from .agent import RuleBasedDrivingAgent
from .ai_assistant import OllamaAssistant
from .controls import MouseDriveControl
from .learning import TrialErrorLearner
from .logger import SimulationLogger
from .renderer import (
    CHAT_INPUT_RECT,
    CHAT_MIC_RECT,
    CHAT_SEND_RECT,
    CHAT_VOICE_RECT,
    DRIVE_MODE_RECT,
    HEIGHT,
    MOUSE_ACCEL_RECT,
    MOUSE_BRAKE_RECT,
    MOUSE_LEFT_RECT,
    MOUSE_RIGHT_RECT,
    MOUSE_WHEEL_RECT,
    MOUSE_WHEEL_CENTER_X,
    NAV_BUTTON_RECTS,
    NAV_BUTTONS,
    PAUSE_RECT,
    QUIT_RECT,
    RESTART_RECT,
    SETTINGS_RECT,
    WIDTH,
    GameRenderer,
)
from .risk import analyze_risk
from .route_rl import LanePathPlan, plan_lane_path
from .simulation import DrivingSimulation
from .voice import VoiceAssistant
from .voice_commands import DrivingCommand, parse_driving_command


FPS = 60


def run() -> None:
    pygame.init()
    pygame.display.set_caption("Ravan AI — Driving Assistant")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    renderer = GameRenderer()
    simulation = DrivingSimulation()
    assistant = OllamaAssistant()
    voice = VoiceAssistant()
    logger = SimulationLogger()
    learner = TrialErrorLearner()
    autonomous_agent = RuleBasedDrivingAgent()
    agent_drive = False
    paused = True
    started = False
    running = True
    chat_focused = False
    chat_input = ""
    queued_question: str | None = None
    latest_ai_reply = ""
    selected_tab = "Drive"
    mouse_drive = MouseDriveControl()
    chat_history: list[tuple[str, str]] = [
        ("assistant", "Ask a driving question or give a voice command. Click MIC, then ask or say accelerate, brake, turn left/right, pause, resume, or restart."),
    ]
    status_message = f"Risk calculated locally · telemetry: {logger.path.name}"
    previous_risk_level = "LOW"
    last_voice_alert_s = -100.0
    last_guidance_s = -30.0
    last_guidance_signature: tuple[object, ...] | None = None
    risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

    def stop_live_coaching() -> None:
        assistant.cancel_guidance()
        voice.cancel_speech()

    def set_chat_focus(focused: bool) -> None:
        nonlocal chat_focused
        chat_focused = focused
        if focused:
            pygame.key.start_text_input()
        else:
            pygame.key.stop_text_input()

    def send_question() -> None:
        nonlocal chat_input, queued_question, status_message
        question = chat_input.strip()
        if not question:
            question = "Explain my current driving risk and what I should do now."
        command = parse_driving_command(question)
        if command is not None:
            chat_input = ""
            execute_user_command(command, question)
            return
        if assistant.pending:
            if queued_question is None:
                queued_question = question
                chat_input = ""
                status_message = "Your question is queued behind the current live AI update."
            else:
                status_message = "One question is already queued; wait for the local AI reply."
            return

        assessment = analyze_risk(simulation)
        context = assessment.to_dict()
        context["simulation_time_s"] = round(simulation.elapsed_s, 2)
        context["ego_lane"] = int(simulation.player.lateral_m / 3.6)
        context["traffic_count"] = len(simulation.traffic)
        context["traffic_density"] = simulation.traffic_density
        context["trial_error_learning"] = learner.context(assessment)
        context["rl_lane_plan"] = lane_plan.to_dict()
        chat_history.append(("you", question))
        if assistant.submit(question, context):
            chat_input = ""
            status_message = f"Asking local model {assistant.model}…"
        else:
            chat_history.append(("assistant", "I couldn't queue that question. Please try again."))

    def reset_guidance_state() -> None:
        nonlocal last_guidance_s, last_guidance_signature, latest_ai_reply
        nonlocal previous_risk_level, last_voice_alert_s
        last_guidance_s = -30.0
        last_guidance_signature = None
        latest_ai_reply = ""
        previous_risk_level = "LOW"
        last_voice_alert_s = -100.0
        stop_live_coaching()

    def activate_mouse_control(action: str, message: str) -> None:
        nonlocal paused, started, status_message, agent_drive
        set_chat_focus(False)
        if agent_drive:
            agent_drive = False
            autonomous_agent.reset()
        if started and paused:
            mouse_drive.clear()
            if risk_assessment.collision:
                status_message = "Contact stopped the simulation. Click RESTART or press R before driving."
            else:
                status_message = "Simulation paused. Click RESUME or press P before driving."
            return
        started = True
        paused = False
        mouse_drive.press(action)
        status_message = message

    def execute_user_command(command: DrivingCommand, utterance: str) -> None:
        nonlocal paused, started, status_message, latest_ai_reply, risk_assessment
        nonlocal chat_history, previous_risk_level, last_voice_alert_s
        nonlocal agent_drive
        set_chat_focus(False)
        chat_history.append(("you", utterance))
        if command.intent == "restart":
            simulation.restart()
            mouse_drive.clear()
            autonomous_agent.reset()
            reset_guidance_state()
            risk_assessment = analyze_risk(simulation)
            started = True
            paused = False
        elif command.intent == "pause":
            mouse_drive.clear()
            paused = True
            stop_live_coaching()
        elif command.intent == "resume":
            if risk_assessment.collision:
                command = DrivingCommand(
                    "blocked", None, 0.0,
                    "Contact stopped the simulation. Say restart to begin again.",
                )
            else:
                started = True
                paused = False
        else:
            if risk_assessment.collision:
                command = DrivingCommand(
                    "blocked", None, 0.0,
                    "Contact stopped the simulation. Say restart before driving again.",
                )
            else:
                agent_drive = False
                autonomous_agent.reset()
                started = True
                paused = False
                mouse_drive.pulse(command.mouse_action or "coast", command.duration_s)

        latest_ai_reply = command.acknowledgement
        status_message = command.acknowledgement
        chat_history.append(("assistant", command.acknowledgement))
        chat_history = chat_history[-12:]
        if not paused:
            voice.speak(command.acknowledgement)

    risk_assessment = analyze_risk(simulation)
    lane_plan: LanePathPlan = plan_lane_path(simulation, risk_assessment)
    last_lane_plan_s = simulation.elapsed_s

    try:
        while running:
            dt = min(clock.tick(FPS) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    mouse_drive.release()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    nav_tab = next(
                        (tab for tab, rect in zip(NAV_BUTTONS, NAV_BUTTON_RECTS) if rect.collidepoint(event.pos)),
                        None,
                    )
                    if nav_tab is not None:
                        selected_tab = nav_tab
                        if nav_tab == "AI Assistant":
                            set_chat_focus(True)
                        else:
                            set_chat_focus(False)
                        if nav_tab == "Analytics":
                            status_message = "Live speed and front-gap history is shown in the lower-right graph."
                        elif nav_tab == "Results":
                            status_message = "Run telemetry is saved under data/runs; evaluate it with scripts/evaluate_game_runs.py."
                        elif nav_tab == "Settings":
                            status_message = "The local Ollama model and endpoint are set in config/assistant.json."
                    elif DRIVE_MODE_RECT.collidepoint(event.pos):
                        set_chat_focus(False)
                        mouse_drive.clear()
                        if risk_assessment.collision:
                            status_message = "Contact stopped the simulation. Restart before enabling Agent Drive."
                        else:
                            agent_drive = not agent_drive
                            autonomous_agent.reset()
                            if agent_drive:
                                started = True
                                paused = False
                                status_message = "Agent Drive is on. The local rule agent controls the simulated car."
                            else:
                                status_message = "Manual drive is on. You have control of the simulated car."
                    elif MOUSE_LEFT_RECT.collidepoint(event.pos):
                        activate_mouse_control("left", "Steering left. Click again or hold the control to continue.")
                    elif MOUSE_RIGHT_RECT.collidepoint(event.pos):
                        activate_mouse_control("right", "Steering right. Click again or hold the control to continue.")
                    elif MOUSE_WHEEL_RECT.collidepoint(event.pos):
                        direction = "left" if event.pos[0] < MOUSE_WHEEL_CENTER_X else "right"
                        activate_mouse_control(direction, f"Steering {direction} from the wheel.")
                    elif MOUSE_BRAKE_RECT.collidepoint(event.pos):
                        activate_mouse_control("brake", "Braking from the on-screen control.")
                    elif MOUSE_ACCEL_RECT.collidepoint(event.pos):
                        activate_mouse_control("accelerate", "Accelerating from the on-screen control.")
                    elif PAUSE_RECT.collidepoint(event.pos):
                        set_chat_focus(False)
                        mouse_drive.clear()
                        if not started:
                            started = True
                            paused = False
                            status_message = (
                                "Agent Drive is controlling the simulated car."
                                if agent_drive
                                else "Simulation started. You are in control."
                            )
                        elif risk_assessment.collision:
                            simulation.restart()
                            autonomous_agent.reset()
                            reset_guidance_state()
                            risk_assessment = analyze_risk(simulation)
                            started = True
                            paused = False
                            status_message = (
                                "Contact reset. Agent Drive is controlling the simulated car."
                                if agent_drive
                                else "Contact reset. Simulation restarted under your control."
                            )
                        else:
                            paused = not paused
                            status_message = "Simulation paused." if paused else "Simulation resumed."
                            if paused:
                                stop_live_coaching()
                    elif RESTART_RECT.collidepoint(event.pos):
                        simulation.restart()
                        reset_guidance_state()
                        mouse_drive.clear()
                        started = True
                        paused = False
                        autonomous_agent.reset()
                        risk_assessment = analyze_risk(simulation)
                        set_chat_focus(False)
                        status_message = (
                            "Simulation restarted. Agent Drive is controlling the car."
                            if agent_drive
                            else "Simulation restarted. You are in control."
                        )
                    elif SETTINGS_RECT.collidepoint(event.pos):
                        selected_tab = "Settings"
                        set_chat_focus(False)
                        status_message = "The local Ollama model and endpoint are set in config/assistant.json."
                    elif QUIT_RECT.collidepoint(event.pos):
                        running = False
                    elif CHAT_SEND_RECT.collidepoint(event.pos):
                        send_question()
                        set_chat_focus(True)
                    elif CHAT_MIC_RECT.collidepoint(event.pos):
                        set_chat_focus(False)
                        if voice.start_listening():
                            status_message = "Starting local push-to-talk…"
                        else:
                            status_message = "Microphone is already listening."
                    elif CHAT_VOICE_RECT.collidepoint(event.pos):
                        set_chat_focus(False)
                        enabled = voice.toggle_voice()
                        status_message = "Local voice enabled." if enabled else "Voice muted."
                    elif CHAT_INPUT_RECT.collidepoint(event.pos):
                        set_chat_focus(True)
                    else:
                        set_chat_focus(False)
                elif event.type == pygame.TEXTINPUT and chat_focused:
                    chat_input = (chat_input + event.text)[:180]
                elif event.type == pygame.KEYDOWN:
                    if chat_focused:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            send_question()
                        elif event.key == pygame.K_BACKSPACE:
                            chat_input = chat_input[:-1]
                        elif event.key == pygame.K_ESCAPE:
                            set_chat_focus(False)
                    else:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_TAB:
                            set_chat_focus(True)
                        elif event.key in {
                            pygame.K_UP,
                            pygame.K_DOWN,
                            pygame.K_LEFT,
                            pygame.K_RIGHT,
                            pygame.K_w,
                            pygame.K_a,
                            pygame.K_s,
                            pygame.K_d,
                            pygame.K_SPACE,
                        }:
                            if agent_drive:
                                agent_drive = False
                                autonomous_agent.reset()
                                status_message = "Manual override is on. You have control of the simulated car."
                            if not started:
                                started = True
                                paused = False
                                if not agent_drive:
                                    status_message = "Simulation started. You are in control."
                        elif event.key == pygame.K_p:
                            mouse_drive.clear()
                            if not started:
                                started = True
                                paused = False
                                status_message = (
                                    "Agent Drive is controlling the simulated car."
                                    if agent_drive
                                    else "Simulation started. You are in control."
                                )
                            elif risk_assessment.collision:
                                simulation.restart()
                                autonomous_agent.reset()
                                reset_guidance_state()
                                risk_assessment = analyze_risk(simulation)
                                started = True
                                paused = False
                                status_message = (
                                    "Contact reset. Agent Drive is controlling the simulated car."
                                    if agent_drive
                                    else "Contact reset. Simulation restarted under your control."
                                )
                            else:
                                paused = not paused
                                status_message = "Simulation paused." if paused else "Simulation resumed."
                                if paused:
                                    stop_live_coaching()
                        elif event.key == pygame.K_r:
                            simulation.restart()
                            reset_guidance_state()
                            mouse_drive.clear()
                            autonomous_agent.reset()
                            started = True
                            paused = False
                            status_message = (
                                "Simulation restarted. Agent Drive is controlling the car."
                                if agent_drive
                                else "Simulation restarted. You are in control."
                            )

            if not running:
                break

            driver_action: str | None = None
            if not paused:
                if agent_drive:
                    agent_action = autonomous_agent.choose_action(simulation, risk_assessment)
                    throttle, steering, handbrake = {
                        "coast": (0.0, 0.0, False),
                        "accelerate": (1.0, 0.0, False),
                        "brake": (-1.0, 0.0, False),
                        "steer_left": (0.0, -1.0, False),
                        "steer_right": (0.0, 1.0, False),
                    }[agent_action]
                    driver_action = agent_action
                else:
                    keys = pygame.key.get_pressed()
                    if chat_focused:
                        throttle = 0.0
                        steering = 0.0
                        handbrake = False
                    else:
                        throttle = float(keys[pygame.K_UP] or keys[pygame.K_w])
                        throttle -= float(keys[pygame.K_DOWN] or keys[pygame.K_s])
                        steering = float(keys[pygame.K_RIGHT] or keys[pygame.K_d])
                        steering -= float(keys[pygame.K_LEFT] or keys[pygame.K_a])
                        handbrake = bool(keys[pygame.K_SPACE])
                        if mouse_drive.active_action:
                            if mouse_drive.active_action == "accelerate":
                                throttle = 1.0
                            elif mouse_drive.active_action == "brake":
                                throttle = -1.0
                            elif mouse_drive.active_action == "left":
                                steering = -1.0
                            elif mouse_drive.active_action == "right":
                                steering = 1.0
                            elif mouse_drive.active_action == "handbrake":
                                throttle = 0.0
                                steering = 0.0
                                handbrake = True
                            elif mouse_drive.active_action == "coast":
                                throttle = 0.0
                                steering = 0.0
                                handbrake = False
                    if handbrake or throttle < -0.1:
                        driver_action = "brake"
                    elif throttle > 0.1:
                        driver_action = "accelerate"
                    elif steering < -0.2:
                        driver_action = "steer_left"
                    elif steering > 0.2:
                        driver_action = "steer_right"
                simulation.update(
                    dt,
                    throttle=throttle,
                    steering=steering,
                    handbrake=handbrake,
                )
                mouse_drive.advance(dt)

            risk_assessment = analyze_risk(simulation)
            result = assistant.poll()
            if result is not None:
                purpose = result.get("purpose", "question")
                if purpose == "guidance" and (paused or risk_assessment.collision):
                    pass
                elif result["ok"]:
                    latest_ai_reply = result["content"]
                    if purpose == "question":
                        chat_history.append(("assistant", result["content"]))
                        chat_history = chat_history[-12:]
                        status_message = "Answered with local Ollama."
                    else:
                        status_message = "Live driving update from local Ollama."
                    if not paused and not risk_assessment.collision:
                        voice.speak(result["content"])
                else:
                    status_message = result["content"]

            if queued_question is not None and not assistant.pending:
                chat_input = queued_question
                queued_question = None
                send_question()

            while True:
                voice_event = voice.poll()
                if voice_event is None:
                    break
                if voice_event["type"] == "transcript":
                    chat_input = voice_event["text"][:180]
                    send_question()
                else:
                    status_message = voice_event["text"]

            if simulation.elapsed_s < last_lane_plan_s or simulation.elapsed_s - last_lane_plan_s >= 0.5:
                lane_plan = plan_lane_path(simulation, risk_assessment)
                last_lane_plan_s = simulation.elapsed_s
            learning_bucket = learner.state_bucket(risk_assessment)
            learner.observe(simulation.elapsed_s, risk_assessment, driver_action)
            learning_hint = learner.hint(learning_bucket)
            current_severity = risk_order.get(risk_assessment.risk_level, 0)
            previous_severity = risk_order.get(previous_risk_level, 0)
            risk_alert_due = (
                current_severity >= 1
                and (
                    current_severity > previous_severity
                    or simulation.elapsed_s - last_voice_alert_s >= 15.0
                )
            )
            if started and not paused and risk_alert_due:
                voice.speak(risk_assessment.recommendation)
                last_voice_alert_s = simulation.elapsed_s
            previous_risk_level = risk_assessment.risk_level

            guidance_signature = (
                risk_assessment.risk_level,
                risk_assessment.risk_score,
                (risk_assessment.nearest_front or {}).get("vehicle_id"),
                risk_assessment.lane_change_left.safe,
                risk_assessment.lane_change_right.safe,
            )
            elapsed_since_guidance = simulation.elapsed_s - last_guidance_s
            material_risk_change = (
                last_guidance_signature is not None
                and guidance_signature != last_guidance_signature
                and current_severity >= 1
            )
            should_request_guidance = (
                started
                and not paused
                and not assistant.pending
                and (
                    last_guidance_s < 0
                    or (material_risk_change and elapsed_since_guidance >= 8.0)
                    or elapsed_since_guidance >= 25.0
                )
            )
            if should_request_guidance:
                guidance_context = risk_assessment.to_dict()
                guidance_context["simulation_time_s"] = round(simulation.elapsed_s, 2)
                guidance_context["ego_lane"] = simulation.player.lane_index
                guidance_context["traffic_count"] = len(simulation.traffic)
                guidance_context["traffic_density"] = simulation.traffic_density
                guidance_context["trial_error_learning"] = learner.context(risk_assessment)
                guidance_context["rl_lane_plan"] = lane_plan.to_dict()
                if assistant.submit_guidance(guidance_context):
                    last_guidance_s = simulation.elapsed_s
                    last_guidance_signature = guidance_signature
                    status_message = "Local AI is reviewing the live driving situation…"

            if started and not paused:
                logger.log(simulation, risk_assessment, lane_plan=lane_plan)
            if risk_assessment.collision:
                paused = True
                stop_live_coaching()
                status_message = "Contact detected in the simulation. Press R or click RESTART."

            renderer.draw(
                screen,
                simulation,
                risk_assessment,
                paused=paused,
                started=started,
                chat_history=chat_history,
                chat_input=chat_input,
                chat_focused=chat_focused,
                ai_pending=assistant.pending,
                model_name=assistant.model,
                status_message=status_message,
                voice_enabled=voice.enabled,
                mic_listening=voice.listening,
                selected_tab=selected_tab,
                mouse_control_action=mouse_drive.active_action,
                agent_drive=agent_drive,
                live_guidance=latest_ai_reply,
                learning_hint=learning_hint,
                lane_plan=lane_plan,
            )
            pygame.display.flip()
    finally:
        assistant.close()
        voice.close()
        learner.save()
        logger.close()
        pygame.key.stop_text_input()
        pygame.quit()

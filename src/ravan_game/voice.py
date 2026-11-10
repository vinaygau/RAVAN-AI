"""Optional offline push-to-talk STT and local Windows text-to-speech."""

from __future__ import annotations

import queue
import threading
from typing import Any


TTS_SPEECH_RATE = 145


class VoiceAssistant:
    """Keep all audio/model work off the Pygame thread.

    The microphone is accessed only after ``start_listening`` is called by the
    explicit MIC button. Speech recognition uses faster-whisper on CPU and
    speech output uses a slower local Windows SAPI voice through pyttsx3 and
    prefers an Indian English voice when SAPI exposes one.
    """

    def __init__(self) -> None:
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._speech: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._listening = False
        self._enabled = True
        self._closed = False
        self._speech_generation = 0
        self._whisper_model = None
        self._tts_thread = threading.Thread(target=self._tts_worker, name="ravan-local-tts", daemon=True)
        self._tts_thread.start()

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening

    def toggle_voice(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._enabled = not self._enabled
            enabled = self._enabled
        if not enabled:
            self.cancel_speech()
        self._events.put({"type": "status", "text": "Local voice enabled." if enabled else "Voice muted."})
        return enabled

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            if self._enabled and not self._closed:
                self._speech.put((self._speech_generation, text.strip()))

    def cancel_speech(self) -> None:
        """Drop queued speech so paused or stale coaching is not played later."""
        with self._lock:
            self._speech_generation += 1
            self._clear_speech_queue()
            self._speech.put(None)

    def close(self) -> None:
        """Stop accepting speech and let the worker shut down with the game."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._enabled = False
            self._speech_generation += 1
            self._listening = False
            self._clear_speech_queue()
            self._speech.put(None)
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _clear_speech_queue(self) -> None:
        while True:
            try:
                self._speech.get_nowait()
            except queue.Empty:
                return

    def start_listening(self) -> bool:
        with self._lock:
            if self._listening or self._closed:
                return False
            self._listening = True
        threading.Thread(target=self._record_and_transcribe, name="ravan-local-stt", daemon=True).start()
        return True

    def _record_and_transcribe(self) -> None:
        try:
            self._events.put({"type": "status", "text": "Loading local speech model…"})
            if self._whisper_model is None:
                from faster_whisper import WhisperModel

                self._whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

            import numpy as np
            import sounddevice as sd

            device = sd.query_devices(kind="input")
            source_rate = int(device["default_samplerate"])
            self._events.put({"type": "status", "text": "Listening for up to 5 seconds. Speak now…"})
            audio = sd.rec(
                int(5 * source_rate),
                samplerate=source_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            samples = audio[:, 0]
            if source_rate != 16000:
                sample_count = max(1, round(len(samples) * 16000 / source_rate))
                old_axis = np.arange(len(samples), dtype=np.float64)
                new_axis = np.arange(sample_count, dtype=np.float64) * source_rate / 16000
                samples = np.interp(new_axis, old_axis, samples).astype(np.float32)
            segments, _ = self._whisper_model.transcribe(samples, beam_size=1, vad_filter=True)
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            if transcript:
                self._events.put({"type": "transcript", "text": transcript})
            else:
                self._events.put({"type": "status", "text": "No speech detected. Click MIC to try again."})
        except Exception as error:  # optional audio hardware/model; keep driving usable
            self._events.put({"type": "status", "text": f"Voice input unavailable: {error}"})
        finally:
            with self._lock:
                self._listening = False

    def _tts_worker(self) -> None:
        engine = None
        while True:
            item = self._speech.get()
            if item is None:
                if engine is not None:
                    try:
                        engine.stop()
                    except Exception:
                        engine = None
                with self._lock:
                    if self._closed:
                        return
                continue
            generation, text = item
            with self._lock:
                should_speak = (
                    self._enabled
                    and not self._closed
                    and generation == self._speech_generation
                )
            if not should_speak:
                continue
            try:
                if engine is None:
                    import pyttsx3

                    engine = pyttsx3.init()
                    engine.setProperty("rate", TTS_SPEECH_RATE)
                    self._prefer_indian_english_voice(engine)
                engine.say(text)
                engine.runAndWait()
            except Exception as error:  # a missing SAPI voice must not affect the game
                engine = None
                self._events.put({"type": "status", "text": f"Voice output unavailable: {error}"})

    @staticmethod
    def _prefer_indian_english_voice(engine: Any) -> None:
        """Choose an installed en-IN voice while preserving the system fallback."""
        voices = engine.getProperty("voices") or []
        markers = ("en-in", "english (india)", "english india", "indian english")
        for voice in voices:
            details = " ".join(
                str(getattr(voice, attribute, ""))
                for attribute in ("id", "name", "languages")
            ).casefold().replace("_", "-")
            if any(marker in details for marker in markers):
                voice_id = getattr(voice, "id", None)
                if voice_id:
                    engine.setProperty("voice", voice_id)
                    return

    def poll(self) -> dict[str, Any] | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

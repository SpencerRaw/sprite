"""Interaction Models for Sprite.

Multi-modal interaction:
- Touch: poke, pet, drag, double-tap, long-press, shake
- Voice: on-device ASR → intent classification
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


# --- Gestures ---

class Gesture(Enum):
    TAP = ("tap", "poke")
    DOUBLE_TAP = ("double_tap", "call")
    SWIPE = ("swipe", "pet")
    LONG_PRESS = ("long_press", "pick_up")
    DRAG = ("drag", "move")
    SHAKE = ("shake", "surprise")

    def __init__(self, name: str, pet_action: str):
        self.name = name
        self.pet_action = pet_action


@dataclass
class TouchEvent:
    gesture: Gesture
    x: float = 0.5
    y: float = 0.5
    intensity: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class TouchRecognizer:
    """Gesture recognizer for Sprite interactions.

    Designed for playful, forgiving touch — not precision gestures.
    """

    TAP_MAX_DURATION = 300       # ms
    TAP_MAX_MOVEMENT = 30        # px
    DOUBLE_TAP_MAX_GAP = 500     # ms
    LONG_PRESS_MIN = 600         # ms
    SWIPE_MIN_DISTANCE = 40      # px

    def __init__(self):
        self._last_tap_time = 0.0
        self._last_tap_pos = (0.0, 0.0)
        self._down_pos = (0.0, 0.0)
        self._down_time = 0.0

    def touch_down(self, x: float, y: float):
        self._down_pos = (x, y)
        self._down_time = time.time()

    def touch_up(self, x: float, y: float) -> TouchEvent:
        now = time.time()
        duration = (now - self._down_time) * 1000
        dx = x - self._down_pos[0]
        dy = y - self._down_pos[1]
        distance = (dx * dx + dy * dy) ** 0.5

        if duration > self.LONG_PRESS_MIN and distance < self.TAP_MAX_MOVEMENT:
            gesture = Gesture.LONG_PRESS
        elif distance > self.SWIPE_MIN_DISTANCE:
            gesture = Gesture.SWIPE
        elif duration < self.TAP_MAX_DURATION and distance < self.TAP_MAX_MOVEMENT:
            if (now - self._last_tap_time) * 1000 < self.DOUBLE_TAP_MAX_GAP:
                gesture = Gesture.DOUBLE_TAP
            else:
                gesture = Gesture.TAP
            self._last_tap_time = now
            self._last_tap_pos = (x, y)
        else:
            gesture = Gesture.TAP  # default

        intensity = min(1.0, max(0.1, 1.0 - distance / 200))
        return TouchEvent(gesture=gesture, x=x, y=y, intensity=intensity, timestamp=now)


# --- Voice ---

class VoiceIntent(Enum):
    GREET = ("greet", ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"])
    PRAISE = ("praise", ["good", "good boy", "good girl", "well done", "clever", "cute", "love you"])
    SUMMON = ("summon", ["come here", "come", "here", "over here"])
    QUERY = ("query", ["hungry", "tired", "okay", "how are you", "what", "are you"])
    COMMAND_SIT = ("command_sit", ["sit", "stay", "wait"])
    COMMAND_TRICK = ("command_trick", ["roll over", "spin", "jump", "dance", "fetch"])
    DISMISS = ("dismiss", ["goodnight", "sleep", "go to sleep", "bye", "bye bye"])
    SCARE = ("scare", ["boo", "scared", "surprise"])

    def __init__(self, action: str, triggers: list[str]):
        self.action = action
        self.triggers = triggers


@dataclass
class VoiceEvent:
    raw_text: str
    intent: Optional[VoiceIntent] = None
    confidence: float = 0.0


class VoiceRecognizer:
    """Voice intent classifier for Sprite.

    In production: on-device ASR (Whisper.cpp) → LLM intent classification.
    For MVP: keyword matching with pet name awareness.
    """

    def __init__(self, pet_name: str = "Sprite"):
        self.pet_name = pet_name.lower()

    def recognize(self, text: str) -> VoiceEvent:
        text_lower = text.lower().strip()

        # Check if addressed to this pet
        addressed = self.pet_name in text_lower

        best_intent = None
        best_score = 0.0

        for intent in VoiceIntent:
            for trigger in intent.triggers:
                if trigger in text_lower:
                    score = len(trigger) / len(text_lower) if text_lower else 0
                    if addressed:
                        score += 0.3
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        return VoiceEvent(
            raw_text=text,
            intent=best_intent,
            confidence=min(1.0, best_score),
        )

    def get_intent_action(self, intent: VoiceIntent) -> str:
        """Map voice intent to pet interaction type."""
        mapping = {
            VoiceIntent.GREET: "greet",
            VoiceIntent.PRAISE: "praise",
            VoiceIntent.SUMMON: "call",
            VoiceIntent.QUERY: "greet",
            VoiceIntent.COMMAND_SIT: "praise",
            VoiceIntent.COMMAND_TRICK: "praise",
            VoiceIntent.DISMISS: "greet",
            VoiceIntent.SCARE: "scare",
        }
        return mapping.get(intent, "greet")

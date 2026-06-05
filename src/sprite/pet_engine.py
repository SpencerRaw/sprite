"""Pet Behavior Engine for Sprite.

State machine, personality, emotions, and autonomous behavior
for the generative AR pet.
"""

from __future__ import annotations

import random
import time
import math
from dataclasses import dataclass, field
from enum import Enum


# --- States ---

class PetState(Enum):
    IDLE = "idle"
    SLEEPING = "sleeping"
    PLAYING = "playing"
    HUNGRY = "hungry"
    EATING = "eating"
    CURIOUS = "curious"
    EXCITED = "excited"
    SCARED = "scared"
    FOLLOWING = "following"


# --- Emotions ---

class Emotion(Enum):
    HAPPY = ("happy", 1.0, "bouncing, bright colors")
    NEUTRAL = ("neutral", 0.0, "still, normal colors")
    CURIOUS = ("curious", 0.5, "tilting head, approaching")
    SLEEPY = ("sleepy", -0.3, "yawning, dimmed colors")
    HUNGRY = ("hungry", -0.5, "following, making sounds")
    EXCITED = ("excited", 1.5, "running circles, glowing")
    SCARED = ("scared", -1.0, "hiding, dark colors")
    AFFECTIONATE = ("affectionate", 0.8, "leaning in, soft glow")

    def __init__(self, label: str, valence: float, description: str):
        self.label = label
        self.valence = valence
        self.description = description


# --- Personality ---

@dataclass
class Personality:
    """Five-dimensional personality for each Sprite.

    All values 0.0-1.0, randomized at creation.
    """
    curiosity: float = 0.5      # How likely to explore new objects
    sociability: float = 0.5     # How strongly it responds to voice
    playfulness: float = 0.5     # How easily it enters play mode
    independence: float = 0.5    # How long before bored without interaction
    affection: float = 0.5       # How much it seeks physical touch

    @classmethod
    def random(cls) -> Personality:
        """Generate a random personality at creation."""
        return cls(
            curiosity=random.uniform(0.2, 0.9),
            sociability=random.uniform(0.2, 0.9),
            playfulness=random.uniform(0.2, 0.9),
            independence=random.uniform(0.2, 0.9),
            affection=random.uniform(0.3, 1.0),
        )

    def describe(self) -> str:
        """Human-readable personality summary."""
        traits = []
        if self.curiosity > 0.7:
            traits.append("very curious")
        elif self.curiosity < 0.3:
            traits.append("cautious")

        if self.sociability > 0.7:
            traits.append("social butterfly")
        elif self.sociability < 0.3:
            traits.append("shy")

        if self.playfulness > 0.7:
            traits.append("playful")
        elif self.playfulness < 0.3:
            traits.append("serious")

        if self.independence > 0.7:
            traits.append("independent")
        elif self.independence < 0.3:
            traits.append("clingy")

        if self.affection > 0.7:
            traits.append("cuddly")
        elif self.affection < 0.3:
            traits.append("aloof")

        if not traits:
            return "balanced and easygoing"
        return ", ".join(traits)


# --- Needs System ---

@dataclass
class Needs:
    """Simulated biological/emotional needs."""
    hunger: float = 0.5        # 0=full, 1=starving
    energy: float = 1.0        # 0=exhausted, 1=fully rested
    social: float = 0.5        # 0=lonely, 1=fulfilled
    curiosity_satisfied: float = 0.5  # 0=bored, 1=stimulated

    def decay(self, dt: float):
        """Needs decay over time (dt in seconds)."""
        # Hunger increases (gets hungrier) over ~4 hours
        self.hunger = min(1.0, self.hunger + dt / (4 * 3600))
        # Energy decreases over ~16 waking hours
        self.energy = max(0.0, self.energy - dt / (16 * 3600))
        # Social need increases (gets lonely) over ~2 hours
        self.social = max(0.0, self.social - dt / (2 * 3600))
        # Curiosity decays over ~1 hour
        self.curiosity_satisfied = max(0.0, self.curiosity_satisfied - dt / 3600)

    def feed(self, amount: float = 1.0):
        self.hunger = max(0.0, self.hunger - 0.7 * amount)

    def rest(self, amount: float = 1.0):
        self.energy = min(1.0, self.energy + 0.5 * amount)

    def socialize(self, amount: float = 1.0):
        self.social = min(1.0, self.social + 0.3 * amount)

    def stimulate(self, amount: float = 1.0):
        self.curiosity_satisfied = min(1.0, self.curiosity_satisfied + 0.4 * amount)

    @property
    def dominant_need(self) -> str:
        scores = {"hunger": self.hunger, "energy": 1 - self.energy,
                  "social": 1 - self.social, "curiosity": 1 - self.curiosity_satisfied}
        return max(scores, key=scores.get)


# --- Pet Engine ---

@dataclass
class Pet:
    """A single Sprite — the full AI pet entity."""

    id: str
    name: str = "Sprite"
    personality: Personality = field(default_factory=Personality.random)
    needs: Needs = field(default_factory=Needs)
    state: PetState = PetState.IDLE
    emotion: Emotion = Emotion.NEUTRAL
    age_seconds: float = 0.0
    last_interaction: float = field(default_factory=time.time)
    position: tuple[float, float] = (0.5, 0.5)  # normalized screen coords
    size: float = 1.0
    expression_params: dict = field(default_factory=dict)

    # Appearance (from generator)
    appearance_prompt: str = ""
    reference_images: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.expression_params = {
            "eye_openness": 1.0,
            "mouth_curve": 0.0,
            "ear_angle": 0.0,
            "tail_wag": 0.0,
            "body_bounce": 0.0,
            "color_saturation": 1.0,
            "glow": 0.0,
        }

    def update(self, dt: float):
        """Update pet state for a time step."""
        self.age_seconds += dt
        self.needs.decay(dt)

        # Time since last interaction
        idle_time = time.time() - self.last_interaction

        # Autonomous state transitions
        new_state = self._evaluate_state(idle_time)
        if new_state != self.state:
            self.state = new_state

        # Update emotion based on state + needs
        self.emotion = self._evaluate_emotion()

        # Update expression parameters
        self._update_expression(dt)

    def _evaluate_state(self, idle_time: float) -> PetState:
        """Determine what state the pet should be in."""
        # Priority: needs-driven
        if self.needs.hunger > 0.7 and self.state not in (PetState.EATING, PetState.SLEEPING):
            return PetState.HUNGRY
        if self.needs.energy < 0.2:
            return PetState.SLEEPING

        # Current state persistence
        if self.state == PetState.EATING:
            return PetState.EATING

        # Idle → boredom → play/curious
        if idle_time > 300 and self.personality.playfulness > 0.5:
            return PetState.PLAYING
        if idle_time > 600 and self.personality.curiosity > 0.5:
            return PetState.CURIOUS

        # Independent pets don't mind being idle
        if idle_time > 1200 and self.personality.independence < 0.3:
            return PetState.FOLLOWING

        return PetState.IDLE

    def _evaluate_emotion(self) -> Emotion:
        if self.state == PetState.SLEEPING:
            return Emotion.SLEEPY
        if self.state == PetState.HUNGRY:
            return Emotion.HUNGRY
        if self.state == PetState.PLAYING:
            return Emotion.EXCITED
        if self.state == PetState.CURIOUS:
            return Emotion.CURIOUS
        if self.state == PetState.SCARED:
            return Emotion.SCARED

        # Idle: blend based on needs
        if self.needs.social < 0.3:
            return Emotion.HUNGRY  # lonely, seeking attention
        if self.needs.hunger > 0.5:
            return Emotion.NEUTRAL

        # Recently interacted → happy/affectionate
        if time.time() - self.last_interaction < 60:
            return Emotion.AFFECTIONATE if self.personality.affection > 0.5 else Emotion.HAPPY

        return Emotion.NEUTRAL

    def _update_expression(self, dt: float):
        """Update visual expression parameters."""
        p = self.expression_params

        # Smooth transitions toward target
        targets = {
            Emotion.HAPPY: {"eye_openness": 0.8, "mouth_curve": 0.5, "tail_wag": 0.6, "body_bounce": 0.4, "glow": 0.3},
            Emotion.EXCITED: {"eye_openness": 1.0, "mouth_curve": 0.7, "tail_wag": 1.0, "body_bounce": 0.8, "glow": 0.6},
            Emotion.CURIOUS: {"eye_openness": 0.9, "mouth_curve": 0.1, "ear_angle": 0.5, "tail_wag": 0.2, "glow": 0.1},
            Emotion.SLEEPY: {"eye_openness": 0.1, "mouth_curve": -0.1, "tail_wag": 0.0, "body_bounce": 0.0, "color_saturation": 0.6, "glow": 0.0},
            Emotion.HUNGRY: {"eye_openness": 0.7, "mouth_curve": -0.3, "tail_wag": 0.3, "glow": 0.0},
            Emotion.SCARED: {"eye_openness": 0.3, "mouth_curve": -0.5, "ear_angle": -0.8, "tail_wag": 0.0, "body_bounce": -0.3, "color_saturation": 0.5, "glow": 0.0},
            Emotion.AFFECTIONATE: {"eye_openness": 0.4, "mouth_curve": 0.4, "tail_wag": 0.4, "body_bounce": 0.1, "glow": 0.5},
            Emotion.NEUTRAL: {"eye_openness": 0.6, "mouth_curve": 0.0, "tail_wag": 0.1, "body_bounce": 0.05, "glow": 0.0},
        }

        target = targets.get(self.emotion, targets[Emotion.NEUTRAL])
        smooth = 3.0 * dt  # smoothing factor

        for key, tval in target.items():
            p[key] += (tval - p[key]) * smooth

    def interact(self, interaction_type: str, intensity: float = 0.5):
        """Handle a user interaction."""
        self.last_interaction = time.time()

        if interaction_type == "poke":
            if random.random() < 0.3:
                self.state = PetState.EXCITED  # some pets get excited by pokes!
            else:
                self.state = PetState.CURIOUS
            self.needs.stimulate(0.3)

        elif interaction_type == "pet":
            self.state = PetState.IDLE
            self.emotion = Emotion.AFFECTIONATE
            self.needs.socialize(1.0)

        elif interaction_type == "feed":
            self.state = PetState.EATING
            self.needs.feed(1.0)
            self.needs.socialize(0.5)

        elif interaction_type == "call":
            self.state = PetState.FOLLOWING
            self.needs.socialize(0.5)

        elif interaction_type == "scare":
            self.state = PetState.SCARED

        elif interaction_type == "praise":
            self.emotion = Emotion.HAPPY
            self.needs.socialize(0.8)
            self.state = PetState.PLAYING if self.personality.playfulness > 0.4 else PetState.IDLE

        elif interaction_type == "greet":
            self.emotion = Emotion.HAPPY
            self.needs.socialize(0.6)

    def get_status_text(self) -> str:
        """Human-readable status for UI."""
        emotion_emoji = {
            Emotion.HAPPY: "😊", Emotion.EXCITED: "🤩", Emotion.CURIOUS: "🤔",
            Emotion.SLEEPY: "😴", Emotion.HUNGRY: "😋", Emotion.SCARED: "😨",
            Emotion.AFFECTIONATE: "🥰", Emotion.NEUTRAL: "😐",
        }
        return f"{emotion_emoji.get(self.emotion, '❓')} {self.emotion.label} — {self.state.value}"

    def get_needs_status(self) -> dict:
        return {
            "hunger": self.needs.hunger,
            "energy": self.needs.energy,
            "social": self.needs.social,
            "curiosity": self.needs.curiosity_satisfied,
            "dominant": self.needs.dominant_need,
        }

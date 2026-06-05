"""Memory Layer for Sprite — Relationship That Grows.

Tracks the evolving relationship between user and pet:
- Daily interaction log
- Learned routines and preferences
- Shared moments (milestones)
- Relationship stage progression
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Moment:
    """A meaningful shared moment."""
    timestamp: float
    event: str
    description: str
    day: int = 0


@dataclass
class DailyLog:
    """One day of interaction data."""
    date: str
    interactions: int = 0
    pokes: int = 0
    pets: int = 0
    feeds: int = 0
    voice_commands: int = 0
    longest_gap_hours: float = 0.0
    mood_summary: str = "neutral"


@dataclass
class PetMemory:
    """Full memory archive for a Sprite."""

    pet_id: str
    pet_name: str = "Sprite"
    created_at: float = field(default_factory=time.time)
    total_interactions: int = 0
    daily_logs: list[DailyLog] = field(default_factory=list)
    moments: list[Moment] = field(default_factory=list)
    tricks_learned: list[str] = field(default_factory=list)
    favorite_spot: str = "center of desk"
    favorite_food: str = "unknown"
    owner_routine: dict = field(default_factory=dict)
    relationship_stage: str = "stranger"  # stranger → acquaintance → friend → companion → family

    def record_interaction(self, interaction_type: str):
        """Record a single interaction."""
        self.total_interactions += 1
        today = self._today_log()
        today.interactions += 1

        if interaction_type == "poke":
            today.pokes += 1
        elif interaction_type == "pet":
            today.pets += 1
        elif interaction_type == "feed":
            today.feeds += 1
        elif interaction_type in ("greet", "praise", "call"):
            today.voice_commands += 1

        self._update_stage()

    def add_moment(self, event: str, description: str):
        """Record a milestone moment."""
        days = int((time.time() - self.created_at) / 86400)
        self.moments.append(Moment(
            timestamp=time.time(),
            event=event,
            description=description,
            day=days,
        ))

    def _today_log(self) -> DailyLog:
        today_str = time.strftime("%Y-%m-%d")
        if not self.daily_logs or self.daily_logs[-1].date != today_str:
            self.daily_logs.append(DailyLog(date=today_str))
        return self.daily_logs[-1]

    def _update_stage(self):
        """Progress through relationship stages based on interaction count."""
        if self.total_interactions >= 1000:
            self.relationship_stage = "family"
        elif self.total_interactions >= 500:
            self.relationship_stage = "companion"
        elif self.total_interactions >= 200:
            self.relationship_stage = "friend"
        elif self.total_interactions >= 50:
            self.relationship_stage = "acquaintance"
        else:
            self.relationship_stage = "stranger"

    @property
    def age_days(self) -> int:
        return int((time.time() - self.created_at) / 86400)

    def get_stage_greeting(self) -> str:
        greetings = {
            "stranger": "A new friend appears...",
            "acquaintance": "Oh, it's you! Hello!",
            "friend": "Hey! I was hoping you'd come back!",
            "companion": "There you are! I missed you!",
            "family": "Welcome home. I've been waiting.",
        }
        return greetings.get(self.relationship_stage, "Hello!")

    def get_stage_behavior(self) -> dict:
        """How the pet behaves at current relationship stage."""
        behaviors = {
            "stranger": {"approach_distance": 0.3, "reaction_speed": 0.3, "trust": 0.1},
            "acquaintance": {"approach_distance": 0.2, "reaction_speed": 0.5, "trust": 0.3},
            "friend": {"approach_distance": 0.1, "reaction_speed": 0.7, "trust": 0.6},
            "companion": {"approach_distance": 0.05, "reaction_speed": 0.9, "trust": 0.85},
            "family": {"approach_distance": 0.0, "reaction_speed": 1.0, "trust": 1.0},
        }
        return behaviors.get(self.relationship_stage, behaviors["stranger"])

    def summary(self) -> str:
        return (
            f"{self.pet_name} — {self.age_days} days old, "
            f"{self.total_interactions} interactions, "
            f"stage: {self.relationship_stage}, "
            f"tricks: {', '.join(self.tricks_learned) if self.tricks_learned else 'none yet'}"
        )

    def to_dict(self) -> dict:
        return {
            "pet_id": self.pet_id,
            "pet_name": self.pet_name,
            "created_at": self.created_at,
            "total_interactions": self.total_interactions,
            "moments": [{"event": m.event, "description": m.description, "day": m.day} for m in self.moments],
            "tricks_learned": self.tricks_learned,
            "favorite_spot": self.favorite_spot,
            "favorite_food": self.favorite_food,
            "relationship_stage": self.relationship_stage,
            "age_days": self.age_days,
        }

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> PetMemory:
        with open(filepath) as f:
            data = json.load(f)
        mem = cls(pet_id=data["pet_id"], pet_name=data["pet_name"])
        mem.created_at = data["created_at"]
        mem.total_interactions = data["total_interactions"]
        mem.tricks_learned = data.get("tricks_learned", [])
        mem.favorite_spot = data.get("favorite_spot", "")
        mem.favorite_food = data.get("favorite_food", "")
        mem.relationship_stage = data.get("relationship_stage", "stranger")
        for m in data.get("moments", []):
            mem.moments.append(Moment(timestamp=0, event=m["event"], description=m["description"], day=m["day"]))
        return mem

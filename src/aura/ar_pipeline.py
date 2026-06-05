"""AR Rendering Pipeline for Aura.

Simulates the real-time AR pipeline:
  Camera → Scene Understanding → Pet State Update → Generative Render → Composite → Display

For MVP in Streamlit: uses a static background + procedurally rendered pet overlay.
In production: real camera feed with depth estimation and lighting matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class Surface:
    """A detected surface in the camera view."""
    x: float        # normalized 0-1
    y: float        # normalized 0-1
    width: float    # normalized
    height: float   # normalized
    type: str       # "desk", "floor", "wall", "hand"
    confidence: float = 0.8


@dataclass
class LightingInfo:
    """Estimated lighting from the camera feed."""
    ambient_r: float = 0.5
    ambient_g: float = 0.5
    ambient_b: float = 0.5
    direction_x: float = 0.0
    direction_y: float = -0.5  # light usually from above
    intensity: float = 1.0


@dataclass
class CameraFrame:
    """A single frame from the camera."""
    width: int = 720
    height: int = 1280
    timestamp: float = 0.0
    surfaces: list[Surface] = None
    lighting: LightingInfo = None

    def __post_init__(self):
        if self.surfaces is None:
            self.surfaces = []
        if self.lighting is None:
            self.lighting = LightingInfo()


class SceneAnalyzer:
    """Analyzes camera frames for AR placement.

    In production: uses ARKit/ARCore or MediaPipe for scene understanding.
    For MVP: simulated surface detection.
    """

    def analyze(self, frame: CameraFrame) -> CameraFrame:
        """Analyze a frame and detect surfaces + lighting."""
        # Simulate surface detection
        frame.surfaces = [
            Surface(0.2, 0.55, 0.6, 0.35, "desk"),      # main desk area
            Surface(0.1, 0.3, 0.3, 0.2, "wall"),          # wall behind
            Surface(0.6, 0.15, 0.2, 0.2, "wall"),         # wall right
        ]
        frame.lighting = LightingInfo(
            ambient_r=0.6, ambient_g=0.55, ambient_b=0.5,
            direction_x=0.2, direction_y=-0.6,
            intensity=0.8,
        )
        return frame

    def find_best_surface(self, frame: CameraFrame,
                          pet_size: float = 0.15) -> Optional[Surface]:
        """Find the best surface to place the pet on."""
        suitable = [s for s in frame.surfaces
                    if s.type in ("desk", "floor")
                    and s.width >= pet_size
                    and s.height >= pet_size]
        if suitable:
            return max(suitable, key=lambda s: s.width * s.height)
        return None

    def get_placement_position(self, surface: Surface,
                               offset: tuple[float, float] = (0.5, 0.3)) -> tuple[float, float]:
        """Get a normalized position on a surface."""
        return (
            surface.x + surface.width * offset[0],
            surface.y + surface.height * offset[1],
        )


class ARCompositor:
    """Composites the generated pet onto the camera frame.

    Handles lighting adaptation, occlusion, and blending.
    For MVP: returns composition parameters for canvas rendering.
    """

    def composite(self, pet_frame: dict, scene: CameraFrame,
                  lighting: Optional[LightingInfo] = None) -> dict:
        """Composite pet onto scene.

        Returns rendering instructions for the frontend.
        """
        if lighting is None:
            lighting = scene.lighting

        # Adapt pet colors to scene lighting
        brightness = (lighting.ambient_r + lighting.ambient_g + lighting.ambient_b) / 3
        pet_frame["lighting_multiplier"] = brightness

        # Shadow under pet
        shadow_opacity = 0.3 * lighting.intensity

        return {
            **pet_frame,
            "shadow_opacity": shadow_opacity,
            "lighting": {
                "r": lighting.ambient_r,
                "g": lighting.ambient_g,
                "b": lighting.ambient_b,
            },
            "composited": True,
        }


class ARPipeline:
    """Full AR pipeline: camera → analyze → render → composite."""

    def __init__(self):
        self.analyzer = SceneAnalyzer()
        self.compositor = ARCompositor()

    def process_frame(self, camera_frame: CameraFrame,
                      pet_frame: dict) -> dict:
        """Process a single frame through the full AR pipeline."""
        # Analyze scene
        frame = self.analyzer.analyze(camera_frame)

        # Find placement surface
        surface = self.analyzer.find_best_surface(frame)
        if surface:
            pos = self.analyzer.get_placement_position(surface)
            pet_frame["x"] = pos[0]
            pet_frame["y"] = pos[1]

        # Composite
        result = self.compositor.composite(pet_frame, frame)

        return result

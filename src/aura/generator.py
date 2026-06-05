"""Generative Model Pipeline for Aura.

Handles:
1. Pet identity creation (image/description → consistent character)
2. Real-time frame generation (pet state → rendered frame)
3. Appearance consistency via prompt engineering + reference images

Production stack:
- Identity: SDXL + LoRA fine-tuning
- Real-time: SD Turbo / LCM-LoRA with identity conditioning
- Compositing: alpha mask + lighting adaptation

For MVP: simulated pipeline with deterministic aura rendering.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuraAppearance:
    """The visual identity of a Aura."""
    base_prompt: str
    seed: int
    body_shape: str          # "round", "elongated", "fluffy", "geometric"
    color_palette: list[str]  # ["#FF6B6B", "#FFE66D", ...]
    eye_style: str            # "big_round", "dots", "almond", "anime"
    size_category: str        # "tiny", "small", "medium"
    special_features: list[str] = field(default_factory=list)
    # ["glowing_antennae", "sparkle_trail", "floating", "wings"]


class IdentityGenerator:
    """Generates unique Aura identities from user input.

    In production, this calls Stable Diffusion / DALL-E API.
    For MVP, we procedurally generate a unique appearance description.
    """

    BODY_SHAPES = ["round and bouncy", "elongated and graceful", "fluffy and cloud-like",
                   "geometric and crystalline", "droplet-shaped", "star-shaped"]
    COLOR_SCHEMES = [
        ["#FF6B6B", "#FFE66D", "#FF8E72"],  # warm sunset
        ["#6BCB77", "#4D96FF", "#FFD93D"],  # fresh meadow
        ["#9B59B6", "#3498DB", "#E8DAEF"],  # twilight
        ["#FF6B9D", "#C44DFF", "#45B7D1"],  # candy
        ["#2ECC71", "#F1C40F", "#E67E22"],  # forest
        ["#E8D5B7", "#C4A882", "#8B7355"],  # warm neutral
        ["#A8E6CF", "#DCEDC1", "#FFD3B6"],  # pastel garden
        ["#FFD3E0", "#E8B4D8", "#C9B8E8"],  # soft lavender
    ]
    EYE_STYLES = ["big round sparkly eyes", "tiny dot eyes", "almond-shaped gentle eyes",
                  "anime-style expressive eyes", "single large cyclops eye",
                  "three small eyes in triangle"]
    SPECIALS = ["glowing antennae that pulse with emotion",
                "a trail of sparkles that follows movement",
                "tiny wings that flutter when excited",
                "floating above surfaces, never touching",
                "color-shifting based on mood",
                "transparent edges that shimmer"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed or random.randint(0, 2**31))

    def generate_from_description(self, description: str) -> AuraAppearance:
        """Generate a unique Aura from a text description.

        Uses the description as a seed for consistent generation.
        In production: SDXL/DALL-E with the description as prompt.
        """
        seed = hash(description) % 2**31
        rng = random.Random(seed)

        # Extract mood/style hints from description
        desc_lower = description.lower()

        if any(w in desc_lower for w in ["fluffy", "furry", "cloud", "soft"]):
            body = "fluffy and cloud-like"
        elif any(w in desc_lower for w in ["crystal", "geometric", "sharp"]):
            body = "geometric and crystalline"
        elif any(w in desc_lower for w in ["slinky", "long", "snake", "worm"]):
            body = "elongated and graceful"
        elif any(w in desc_lower for w in ["star", "sparkle"]):
            body = "star-shaped"
        elif any(w in desc_lower for w in ["bouncy", "ball", "round", "circle"]):
            body = "round and bouncy"
        else:
            body = rng.choice(self.BODY_SHAPES)

        # Pick colors based on description mood
        if any(w in desc_lower for w in ["dark", "night", "shadow", "gothic"]):
            palette = ["#2C3E50", "#8E44AD", "#34495E"]
        elif any(w in desc_lower for w in ["ocean", "sea", "water", "blue"]):
            palette = ["#3498DB", "#1ABC9C", "#2980B9"]
        elif any(w in desc_lower for w in ["fire", "flame", "hot", "red"]):
            palette = ["#E74C3C", "#F39C12", "#D35400"]
        elif any(w in desc_lower for w in ["forest", "green", "nature", "plant"]):
            palette = ["#27AE60", "#2ECC71", "#F1C40F"]
        else:
            palette = rng.choice(self.COLOR_SCHEMES)

        eyes = rng.choice(self.EYE_STYLES)
        specials = rng.sample(self.SPECIALS, k=rng.randint(0, 2))

        size = "small" if any(w in desc_lower for w in ["tiny", "small", "little"]) else "medium"

        prompt = (
            f"A {body} creature with {eyes}. "
            f"Color palette: {', '.join(palette[:2])}. "
            f"{' '.join(specials)}. "
            f"Cute, stylized, 2D illustration style, soft edges, "
            f"consistent character design, full body, transparent background."
        )

        return AuraAppearance(
            base_prompt=prompt,
            seed=seed,
            body_shape=body,
            color_palette=palette,
            eye_style=eyes,
            size_category=size,
            special_features=specials,
        )

    def generate_from_image_reference(self, image_description: str,
                                      style_notes: str = "") -> AuraAppearance:
        """Generate a Aura inspired by an image.

        In production: img2img with ControlNet for pose/shape extraction.
        """
        combined = f"{image_description} {style_notes}".strip()
        return self.generate_from_description(combined)


class FrameRenderer:
    """Generative rendering pipeline for Aura frames.

    Two-tier:
    1. If GPU + StreamDiffusionV2 available: real-time SD Turbo rendering
    2. Otherwise: procedural Canvas rendering (MVP fallback)

    To use real rendering:
        from aura.stream_renderer import StreamRenderer, RendererConfig
        renderer = StreamRenderer(RendererConfig(lora_path="pet.safetensors"))
        renderer.load()
        frame = renderer.render(camera_frame, pose_image, prompt)
    """

    def __init__(self, appearance: AuraAppearance):
        self.appearance = appearance
        self.frame_count = 0
        self._rng = random.Random(appearance.seed)
        self._stream_renderer = None  # Lazy-loaded when GPU available

    def try_load_stream_renderer(self, lora_path: str = None) -> bool:
        """Attempt to load StreamDiffusionV2 for real rendering.
        
        Returns True if successfully loaded, False if GPU not available.
        """
        try:
            from .stream_renderer import StreamRenderer, RendererConfig
            config = RendererConfig(
                model_id="stabilityai/sd-turbo",
                lora_path=lora_path,
                width=512, height=512,
                target_fps=30,
                use_tiny_vae=True,
            )
            self._stream_renderer = StreamRenderer(config)
            self._stream_renderer.load(lora_path=lora_path)
            return True
        except (ImportError, Exception) as e:
            print(f"[Aura] Stream renderer not available: {e}")
            print(f"[Aura] Falling back to procedural rendering.")
            return False

    def has_real_renderer(self) -> bool:
        return self._stream_renderer is not None

    def render_frame(self, expression_params: dict,
                     position: tuple[float, float] = (0.5, 0.5),
                     size: float = 1.0,
                     camera_frame=None,
                     pose_image=None) -> dict:
        """Render a single frame of the Aura.

        Returns a dict with rendering instructions for the canvas.
        In production: returns a PIL Image (generated by SD Turbo).
        """
        self.frame_count += 1

        # If real renderer is available, use it
        if self._stream_renderer:
            prompt = f"{self.appearance.body_shape} creature, looking at camera, cute"
            try:
                from PIL import Image as PILImage
                rendered = self._stream_renderer.render(
                    camera_frame=camera_frame,
                    pose_image=pose_image,
                    prompt=prompt,
                    strength=0.5,
                )
                # Return rendering params pointing to the generated image
                return {
                    "x": position[0], "y": position[1],
                    "size": size,
                    "rendered_image": rendered,
                    "renderer": "streamdiffusion_v2",
                    "frame": self.frame_count,
                }
            except Exception as e:
                print(f"[Aura] Stream render failed, falling back: {e}")

        # Procedural animation based on expression params (fallback)
        bounce = expression_params.get("body_bounce", 0)
        wag = expression_params.get("tail_wag", 0)
        glow = expression_params.get("glow", 0)
        saturation = expression_params.get("color_saturation", 1.0)
        eye_open = expression_params.get("eye_openness", 1.0)
        mouth_curve = expression_params.get("mouth_curve", 0)
        ear_angle = expression_params.get("ear_angle", 0)

        # Animate with time
        t = self.frame_count * 0.1
        bounce_offset = math.sin(t * 3) * bounce * 8
        wag_offset = math.sin(t * 5) * wag * 15

        return {
            "x": position[0],
            "y": position[1],
            "size": size,
            "body_bounce_y": bounce_offset,
            "tail_angle": wag_offset,
            "glow_intensity": glow,
            "color_saturation": saturation,
            "eye_scale_y": max(0.1, eye_open),
            "mouth_curve": mouth_curve,
            "ear_angle": ear_angle * 30,
            "body_shape": self.appearance.body_shape,
            "colors": self.appearance.color_palette,
            "eye_style": self.appearance.eye_style,
            "specials": self.appearance.special_features,
            "frame": self.frame_count,
        }


import math  # for sin/cos in render_frame

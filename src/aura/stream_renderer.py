"""StreamDiffusionV2 Renderer for Aura.

Real-time generative rendering pipeline for AR pet.
Wraps StreamDiffusionV2 with LoRA identity + ControlNet pose control.

Requirements:
  - NVIDIA GPU (A100/H100/4090) with 24GB+ VRAM
  - CUDA 12.1+
  - pip install streamdiffusion diffusers torch controlnet_aux

Performance (A100, 512×512):
  - SD Turbo 1-step: 60+ FPS
  - SDXL Turbo 1-step: 30+ FPS
  - With ControlNet: 25+ FPS
  - With LoRA: no significant overhead

Architecture:
  Camera frame → Pose estimation → ControlNet conditioning
                                    ↓
  Identity LoRA ──→ SD Turbo img2img ──→ Rendered pet frame
                                    ↓
  Alpha mask ──→ Composite with camera ──→ Display
"""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

import numpy as np
from PIL import Image


# --- Configuration ---

@dataclass
class RendererConfig:
    """Configuration for the StreamDiffusionV2 renderer."""

    # Model
    model_id: str = "stabilityai/sd-turbo"
    # Alternative: "stabilityai/sdxl-turbo" for higher quality at lower FPS
    use_sdxl: bool = False

    # Resolution
    width: int = 512
    height: int = 512

    # Performance
    num_inference_steps: int = 1       # SD Turbo works with 1 step
    use_tiny_vae: bool = True          # Faster VAE, slight quality loss
    use_tiny_vae: bool = True

    # Identity
    lora_path: Optional[str] = None    # Path to trained LoRA weights
    lora_scale: float = 0.8            # How strongly to apply identity

    # Pose control
    controlnet_model: str = "lllyasviel/control_v11p_sd15_openpose"
    controlnet_scale: float = 0.7

    # Acceleration
    use_xformers: bool = True          # Memory-efficient attention
    use_sfast: bool = False            # Stable Fast (additional speedup)
    compile_model: bool = True         # torch.compile for speed

    # Streaming
    target_fps: int = 30
    frame_buffer_size: int = 3

    # Device
    device: str = "cuda"
    torch_dtype: str = "float16"


# --- StreamDiffusionV2 Wrapper ---

class StreamRenderer:
    """Real-time generative renderer using StreamDiffusionV2.

    Usage:
        renderer = StreamRenderer(config)
        renderer.load(lora_path="my_pet.safetensors")

        # In render loop:
        for camera_frame in camera_stream:
            pet_frame = renderer.render(
                camera_frame=camera_frame,
                pose=current_pose,
                expression=current_expression,
            )
            display(pet_frame)
    """

    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()
        self._pipe = None
        self._stream = None
        self._lora_loaded = False
        self._controlnet_loaded = False
        self._warmup_done = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=8)
        self._running = False
        self._render_thread: Optional[threading.Thread] = None

        # Frame statistics
        self.frame_count = 0
        self.fps_history: list[float] = []
        self.current_fps = 0.0

    # --- Model Loading ---

    def load(self, lora_path: Optional[str] = None) -> StreamRenderer:
        """Load the SD Turbo model with StreamDiffusionV2 optimizations.

        Args:
            lora_path: Path to .safetensors LoRA file for pet identity.
        """
        import torch
        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            ControlNetModel,
        )
        from streamdiffusion import StreamDiffusion
        from streamdiffusion.acceleration import (
            accelerate_with_xformers,
            accelerate_with_sfast,
        )

        model_id = "stabilityai/sdxl-turbo" if self.config.use_sdxl else "stabilityai/sd-turbo"

        # Load base pipeline
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.config.device)

        # Load ControlNet if configured
        if self.config.controlnet_model:
            controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model,
                torch_dtype=getattr(torch, self.config.torch_dtype),
            ).to(self.config.device)
            self._pipe.controlnet = controlnet
            self._controlnet_loaded = True

        # Acceleration
        if self.config.use_xformers:
            accelerate_with_xformers(self._pipe)
        if self.config.use_sfast:
            accelerate_with_sfast(self._pipe)

        # torch.compile for speed
        if self.config.compile_model:
            self._pipe.unet = torch.compile(self._pipe.unet)

        # Load LoRA
        if lora_path or self.config.lora_path:
            self._load_lora(lora_path or self.config.lora_path)

        # Initialize StreamDiffusion engine
        self._stream = StreamDiffusion(
            pipe=self._pipe,
            width=self.config.width,
            height=self.config.height,
            use_tiny_vae=self.config.use_tiny_vae,
            torch_dtype=getattr(torch, self.config.torch_dtype),
        )

        # Prepare for streaming
        self._stream.prepare(
            prompt="",  # Will be set per-frame
            negative_prompt="blurry, distorted, deformed, ugly, bad anatomy",
            num_inference_steps=self.config.num_inference_steps,
        )

        print(f"[Aura] StreamDiffusionV2 loaded. Model: {model_id}")
        print(f"[Aura] LoRA: {'loaded' if self._lora_loaded else 'none'}")
        print(f"[Aura] ControlNet: {'loaded' if self._controlnet_loaded else 'none'}")

        return self

    def _load_lora(self, lora_path: str):
        """Load LoRA weights for pet identity."""
        from diffusers.utils import load_image

        if not Path(lora_path).exists():
            print(f"[Aura] WARNING: LoRA file not found: {lora_path}")
            return

        self._pipe.load_lora_weights(lora_path)
        self._pipe.fuse_lora(lora_scale=self.config.lora_scale)
        self._lora_loaded = True
        print(f"[Aura] Identity LoRA loaded: {lora_path}")

    # --- Real-Time Rendering ---

    def render(
        self,
        camera_frame: Optional[Image.Image] = None,
        pose_image: Optional[Image.Image] = None,   # ControlNet pose input
        prompt: str = "",
        strength: float = 0.6,                       # img2img strength
        seed: int = -1,
    ) -> Image.Image:
        """Render a single frame of the Aura.

        Args:
            camera_frame: Current camera frame (for inpainting/compositing context).
                         If None, generates standalone pet frame.
            pose_image: OpenPose skeleton or edge map for ControlNet.
            prompt: Text prompt describing desired pose/expression.
            strength: How much to deviate from input (0=keep, 1=full regenerate).
            seed: Random seed (-1 for random).

        Returns:
            PIL Image of the rendered pet frame (RGBA with alpha mask).
        """
        import torch

        # Build full prompt with identity prefix
        full_prompt = self._build_prompt(prompt)

        # Use camera frame as init image for img2img
        init_image = camera_frame

        # Set prompt for this frame
        self._stream.pipe.set_progress_bar_config(disable=True)

        # Generate
        with torch.inference_mode():
            output = self._stream(
                prompt=full_prompt,
                negative_prompt="blurry, distorted, deformed, ugly, bad anatomy, extra limbs",
                image=init_image,
                control_image=pose_image,
                strength=strength,
                num_inference_steps=self.config.num_inference_steps,
                generator=torch.Generator(device=self.config.device).manual_seed(seed) if seed >= 0 else None,
            )

        self.frame_count += 1

        # Extract alpha mask (simple: white background removal via threshold)
        output_rgba = self._add_alpha_mask(output)

        return output_rgba

    def render_batch(
        self,
        frames: list[Image.Image],
        poses: Optional[list[Image.Image]] = None,
        prompts: Optional[list[str]] = None,
        strength: float = 0.6,
    ) -> list[Image.Image]:
        """Render multiple frames in batch for better throughput."""
        results = []
        for i, frame in enumerate(frames):
            pose = poses[i] if poses else None
            prompt = prompts[i] if prompts else ""
            results.append(self.render(frame, pose, prompt, strength))
        return results

    # --- Streaming Mode ---

    def start_streaming(
        self,
        frame_source: Callable[[], Image.Image],
        pose_source: Optional[Callable[[], Image.Image]] = None,
        prompt_fn: Optional[Callable[[], str]] = None,
        on_frame: Optional[Callable[[Image.Image], None]] = None,
    ):
        """Start continuous rendering in background thread.

        Args:
            frame_source: Callable that returns next camera frame.
            pose_source: Callable that returns next ControlNet pose image.
            prompt_fn: Callable that returns next text prompt.
            on_frame: Callback receiving each rendered frame.
        """
        self._running = True
        self._render_thread = threading.Thread(
            target=self._streaming_loop,
            args=(frame_source, pose_source, prompt_fn, on_frame),
            daemon=True,
        )
        self._render_thread.start()
        print(f"[Aura] Streaming started at target {self.config.target_fps} FPS")

    def stop_streaming(self):
        """Stop the rendering thread."""
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=5.0)
        print(f"[Aura] Streaming stopped. Total frames: {self.frame_count}")

    def _streaming_loop(
        self,
        frame_source: Callable[[], Image.Image],
        pose_source: Optional[Callable[[], Image.Image]],
        prompt_fn: Optional[Callable[[], str]],
        on_frame: Optional[Callable[[Image.Image], None]],
    ):
        """Main rendering loop running in background thread."""
        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = time.time()

        while self._running:
            loop_start = time.time()

            try:
                # Get inputs
                camera = frame_source()
                pose = pose_source() if pose_source else None
                prompt = prompt_fn() if prompt_fn else ""

                # Render
                pet_frame = self.render(camera, pose, prompt)

                # Track FPS
                render_time = time.time() - loop_start
                self.fps_history.append(1.0 / max(render_time, 0.001))
                if len(self.fps_history) > 100:
                    self.fps_history = self.fps_history[-100:]
                self.current_fps = sum(self.fps_history[-30:]) / min(len(self.fps_history), 30)

                # Deliver frame
                if on_frame:
                    on_frame(pet_frame)

                # Frame rate limiting
                elapsed = time.time() - last_frame_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                last_frame_time = time.time()

            except Exception as e:
                print(f"[Aura] Render error: {e}")
                time.sleep(0.1)

    # --- Helpers ---

    def _build_prompt(self, action_prompt: str) -> str:
        """Build the full generation prompt with identity anchoring."""
        identity_anchor = (
            "exact same creature as reference, consistent character, "
            "same colors, same shape, same eyes, same fur pattern, "
            "same art style, identical design, "
        )
        quality = "high quality, sharp focus, smooth edges, soft lighting"

        return f"{identity_anchor} {action_prompt} {quality}"

    def _add_alpha_mask(self, image: Image.Image) -> Image.Image:
        """Extract alpha mask by removing white/light background."""
        import numpy as np

        img_array = np.array(image.convert("RGBA"))

        # Simple threshold: make near-white pixels transparent
        r, g, b, a = img_array[..., 0], img_array[..., 1], img_array[..., 2], img_array[..., 3]

        # Pixels where all channels are bright → transparent
        brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
        mask = brightness < 240  # Keep non-white pixels

        # Edge feathering
        from scipy import ndimage
        mask = ndimage.binary_erosion(mask, iterations=1).astype(np.uint8) * 255

        img_array[..., 3] = mask

        return Image.fromarray(img_array, "RGBA")

    def get_stats(self) -> dict:
        """Return current rendering statistics."""
        return {
            "fps": self.current_fps,
            "frames_rendered": self.frame_count,
            "lora_loaded": self._lora_loaded,
            "controlnet_loaded": self._controlnet_loaded,
            "resolution": f"{self.config.width}×{self.config.height}",
            "model": self.config.model_id,
        }

    def unload(self):
        """Free GPU memory."""
        if self._running:
            self.stop_streaming()
        del self._pipe
        del self._stream
        import torch
        torch.cuda.empty_cache()
        print("[Aura] Renderer unloaded, GPU memory freed.")


# --- Pose Estimator (for ControlNet input) ---

class PoseEstimator:
    """Estimates pet pose from interaction state for ControlNet conditioning.

    In production, this would use OpenPose to extract human skeleton
    and map it to pet skeleton. For MVP, we generate synthetic pose maps.
    """

    # Keypoint definitions for a simple creature skeleton
    KEYPOINTS = {
        "head": 0,
        "neck": 1,
        "body_center": 2,
        "tail_base": 3,
        "tail_tip": 4,
        "front_left_leg": 5,
        "front_right_leg": 6,
        "back_left_leg": 7,
        "back_right_leg": 8,
        "left_ear": 9,
        "right_ear": 10,
    }

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

    def generate_pose_image(
        self,
        body_position: tuple[float, float] = (0.5, 0.5),
        pose: str = "sitting",
        expression_params: Optional[dict] = None,
    ) -> Image.Image:
        """Generate a synthetic pose skeleton image for ControlNet.

        Args:
            body_position: Normalized (x, y) center of pet.
            pose: "sitting", "standing", "lying", "walking", "jumping", "curled".
            expression_params: From Pet.expression_params.

        Returns:
            PIL Image with OpenPose-style skeleton rendering.
        """
        import numpy as np
        from PIL import Image, ImageDraw

        # Create blank canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)

        cx = int(body_position[0] * self.width)
        cy = int(body_position[1] * self.height)

        # Define skeleton based on pose
        if pose == "sitting":
            keypoints = self._sitting_pose(cx, cy)
        elif pose == "standing":
            keypoints = self._standing_pose(cx, cy)
        elif pose == "lying":
            keypoints = self._lying_pose(cx, cy)
        elif pose == "walking":
            keypoints = self._walking_pose(cx, cy)
        elif pose == "jumping":
            keypoints = self._jumping_pose(cx, cy)
        elif pose == "curled":
            keypoints = self._curled_pose(cx, cy)
        else:
            keypoints = self._sitting_pose(cx, cy)

        # Apply expression modifiers
        if expression_params:
            ear_angle = expression_params.get("ear_angle", 0)
            body_bounce = expression_params.get("body_bounce", 0)
            tail_wag = expression_params.get("tail_wag", 0)

            # Modify ear positions
            if "left_ear" in keypoints and "right_ear" in keypoints:
                keypoints["left_ear"] = (
                    keypoints["left_ear"][0] + int(ear_angle * 20),
                    keypoints["left_ear"][1],
                )
                keypoints["right_ear"] = (
                    keypoints["right_ear"][0] - int(ear_angle * 20),
                    keypoints["right_ear"][1],
                )

            # Body bounce
            for k in keypoints:
                keypoints[k] = (keypoints[k][0], keypoints[k][1] + int(body_bounce * 15))

            # Tail wag
            if "tail_tip" in keypoints:
                keypoints["tail_tip"] = (
                    keypoints["tail_tip"][0] + int(tail_wag * 30),
                    keypoints["tail_tip"][1],
                )

        # Draw keypoints
        for name, (kx, ky) in keypoints.items():
            draw.ellipse([kx - 4, ky - 4, kx + 4, ky + 4], fill=(255, 255, 255))

        # Draw connections (limbs)
        connections = [
            ("head", "neck"), ("neck", "body_center"),
            ("body_center", "tail_base"), ("tail_base", "tail_tip"),
            ("body_center", "front_left_leg"), ("body_center", "front_right_leg"),
            ("body_center", "back_left_leg"), ("body_center", "back_right_leg"),
            ("head", "left_ear"), ("head", "right_ear"),
        ]
        for a, b in connections:
            if a in keypoints and b in keypoints:
                draw.line([keypoints[a], keypoints[b]], fill=(200, 200, 200), width=2)

        return img

    # --- Pose Definitions ---

    def _sitting_pose(self, cx: int, cy: int) -> dict:
        h = self.height
        return {
            "head": (cx, cy - 80),
            "neck": (cx, cy - 50),
            "body_center": (cx, cy),
            "tail_base": (cx, cy + 30),
            "tail_tip": (cx + 60, cy + 10),
            "front_left_leg": (cx - 25, cy + 40),
            "front_right_leg": (cx + 25, cy + 40),
            "back_left_leg": (cx - 30, cy + 60),
            "back_right_leg": (cx + 30, cy + 60),
            "left_ear": (cx - 20, cy - 110),
            "right_ear": (cx + 20, cy - 110),
        }

    def _standing_pose(self, cx: int, cy: int) -> dict:
        return {
            "head": (cx, cy - 100),
            "neck": (cx, cy - 70),
            "body_center": (cx, cy - 20),
            "tail_base": (cx, cy + 10),
            "tail_tip": (cx + 50, cy - 30),
            "front_left_leg": (cx - 20, cy + 50),
            "front_right_leg": (cx + 20, cy + 50),
            "back_left_leg": (cx - 20, cy + 80),
            "back_right_leg": (cx + 20, cy + 80),
            "left_ear": (cx - 20, cy - 120),
            "right_ear": (cx + 20, cy - 120),
        }

    def _lying_pose(self, cx: int, cy: int) -> dict:
        return {
            "head": (cx - 80, cy),
            "neck": (cx - 50, cy),
            "body_center": (cx, cy),
            "tail_base": (cx + 40, cy),
            "tail_tip": (cx + 90, cy - 10),
            "front_left_leg": (cx - 30, cy + 20),
            "front_right_leg": (cx - 30, cy - 20),
            "back_left_leg": (cx + 20, cy + 20),
            "back_right_leg": (cx + 20, cy - 20),
            "left_ear": (cx - 100, cy - 25),
            "right_ear": (cx - 100, cy + 25),
        }

    def _walking_pose(self, cx: int, cy: int) -> dict:
        return {
            "head": (cx, cy - 90),
            "neck": (cx, cy - 60),
            "body_center": (cx, cy - 10),
            "tail_base": (cx, cy + 20),
            "tail_tip": (cx + 50, cy),
            "front_left_leg": (cx - 15, cy + 40),
            "front_right_leg": (cx + 25, cy + 60),
            "back_left_leg": (cx - 25, cy + 80),
            "back_right_leg": (cx + 15, cy + 60),
            "left_ear": (cx - 20, cy - 115),
            "right_ear": (cx + 20, cy - 115),
        }

    def _jumping_pose(self, cx: int, cy: int) -> dict:
        return {
            "head": (cx, cy - 120),
            "neck": (cx, cy - 90),
            "body_center": (cx, cy - 40),
            "tail_base": (cx, cy - 10),
            "tail_tip": (cx + 40, cy - 50),
            "front_left_leg": (cx - 20, cy - 10),
            "front_right_leg": (cx + 20, cy - 10),
            "back_left_leg": (cx - 15, cy + 20),
            "back_right_leg": (cx + 15, cy + 20),
            "left_ear": (cx - 25, cy - 140),
            "right_ear": (cx + 25, cy - 140),
        }

    def _curled_pose(self, cx: int, cy: int) -> dict:
        return {
            "head": (cx + 30, cy - 20),
            "neck": (cx + 10, cy),
            "body_center": (cx, cy + 20),
            "tail_base": (cx - 20, cy + 20),
            "tail_tip": (cx - 50, cy),
            "front_left_leg": (cx + 10, cy + 30),
            "front_right_leg": (cx - 10, cy + 30),
            "back_left_leg": (cx - 5, cy + 40),
            "back_right_leg": (cx - 15, cy + 40),
            "left_ear": (cx + 20, cy - 35),
            "right_ear": (cx + 40, cy - 20),
        }

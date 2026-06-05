"""WebSocket Streaming Server for Sprite.

Streams rendered AR pet frames from GPU server to mobile clients.
Architecture:
  Mobile App ←→ WebSocket ←→ StreamRenderer (GPU)

Protocol (JSON + binary frames):
  Client → Server:
    {"type": "init", "lora_id": "buddy_v1"}
    {"type": "camera_frame", "data": "<base64>"}
    {"type": "interaction", "action": "poke", "intensity": 0.8}
    {"type": "pose_update", "pose": "sitting", "position": [0.5, 0.4]}

  Server → Client:
    {"type": "frame", "data": "<base64 png>", "fps": 31.2, "timestamp": 1234567890}
    {"type": "stats", "fps": 30.5, "frames_rendered": 5421}
    {"type": "error", "message": "GPU OOM"}

Usage:
    python -m sprite.server --port 8765 --lora ./loras/buddy.safetensors
"""

from __future__ import annotations

import asyncio
import json
import base64
import time
import io
import traceback
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from PIL import Image


# --- Protocol ---

@dataclass
class ClientState:
    """Per-client rendering state."""
    client_id: str
    connected_at: float = field(default_factory=time.time)
    renderer: Optional["StreamRenderer"] = None  # noqa: F821
    pose_estimator: Optional["PoseEstimator"] = None  # noqa: F821
    last_camera_frame: Optional[Image.Image] = None
    current_pose: str = "sitting"
    current_position: tuple[float, float] = (0.5, 0.4)
    expression_params: dict = field(default_factory=dict)
    frame_count: int = 0
    active: bool = True


class SpriteServer:
    """WebSocket server that streams rendered Sprite frames.

    Designed to run on a cloud GPU instance (e.g., AWS g5.xlarge, Lambda Labs A100).
    Multiple clients can connect simultaneously — each gets their own render pipeline.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        lora_path: Optional[str] = None,
        model_id: str = "stabilityai/sd-turbo",
        max_clients: int = 4,
    ):
        self.host = host
        self.port = port
        self.lora_path = lora_path
        self.model_id = model_id
        self.max_clients = max_clients

        self.clients: dict[str, ClientState] = {}
        self._server = None
        self._running = False

    # --- Lifecycle ---

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            print("[Sprite] Install websockets: pip install websockets")
            raise

        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,  # 10MB max message
        )
        print(f"[Sprite] Server started: ws://{self.host}:{self.port}")
        print(f"[Sprite] LoRA: {self.lora_path or 'none'}")
        print(f"[Sprite] Model: {self.model_id}")
        print(f"[Sprite] Max clients: {self.max_clients}")

        await self._server.wait_closed()

    async def stop(self):
        """Stop the server and clean up."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Unload all renderers
        for client in self.clients.values():
            if client.renderer:
                client.renderer.unload()

        print("[Sprite] Server stopped.")

    # --- Client Handling ---

    async def _handle_client(self, websocket, path=""):
        """Handle a single WebSocket client connection."""
        client_id = f"client_{len(self.clients):03d}"

        # Check capacity
        if len(self.clients) >= self.max_clients:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Server full ({self.max_clients} max). Try again later.",
            }))
            await websocket.close()
            return

        # Initialize client state
        client = ClientState(client_id=client_id)
        self.clients[client_id] = client

        remote = websocket.remote_address
        print(f"[Sprite] Client connected: {client_id} from {remote}")

        try:
            # Initialize renderer for this client
            await self._init_client_renderer(client)

            # Message loop
            async for message in websocket:
                try:
                    await self._handle_message(websocket, client, message)
                except Exception as e:
                    print(f"[Sprite] Error handling message from {client_id}: {e}")
                    traceback.print_exc()
                    await self._send_error(websocket, str(e))

        except Exception as e:
            print(f"[Sprite] Client {client_id} error: {e}")
        finally:
            # Cleanup
            if client.renderer:
                client.renderer.unload()
            self.clients.pop(client_id, None)
            print(f"[Sprite] Client disconnected: {client_id}")

    async def _init_client_renderer(self, client: ClientState):
        """Initialize the StreamRenderer for a client."""
        from .stream_renderer import StreamRenderer, RendererConfig, PoseEstimator

        config = RendererConfig(
            model_id=self.model_id,
            lora_path=self.lora_path,
            width=512,
            height=512,
            target_fps=30,
            use_tiny_vae=True,
            use_xformers=True,
            compile_model=False,  # First frame is slow with compile
        )

        client.renderer = StreamRenderer(config)
        client.renderer.load(lora_path=self.lora_path)
        client.pose_estimator = PoseEstimator(512, 512)

        print(f"[Sprite] Renderer initialized for {client.client_id}")

    async def _handle_message(self, websocket, client: ClientState, raw_message: str):
        """Process an incoming WebSocket message."""
        msg = json.loads(raw_message)
        msg_type = msg.get("type", "")

        if msg_type == "camera_frame":
            await self._handle_camera_frame(websocket, client, msg)

        elif msg_type == "interaction":
            await self._handle_interaction(websocket, client, msg)

        elif msg_type == "pose_update":
            client.current_pose = msg.get("pose", "sitting")
            client.current_position = tuple(msg.get("position", [0.5, 0.4]))
            if "expression" in msg:
                client.expression_params = msg["expression"]

        elif msg_type == "init":
            # Re-initialize with different LoRA
            lora_id = msg.get("lora_id")
            if lora_id:
                lora_path = f"./loras/{lora_id}.safetensors"
                if Path(lora_path).exists():
                    client.renderer.unload()
                    client.renderer.load(lora_path=lora_path)

        elif msg_type == "stats_request":
            stats = client.renderer.get_stats() if client.renderer else {}
            await websocket.send(json.dumps({
                "type": "stats",
                **stats,
                "client_id": client.client_id,
                "clients_connected": len(self.clients),
            }))

    async def _handle_camera_frame(self, websocket, client: ClientState, msg: dict):
        """Process a camera frame from the mobile client."""
        # Decode frame
        frame_b64 = msg.get("data", "")
        if not frame_b64:
            return

        try:
            frame_bytes = base64.b64decode(frame_b64)
            camera_frame = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            camera_frame = camera_frame.resize((512, 512), Image.LANCZOS)
        except Exception as e:
            await self._send_error(websocket, f"Invalid frame data: {e}")
            return

        client.last_camera_frame = camera_frame

        # Generate pose image from current state
        pose_image = None
        if client.pose_estimator:
            pose_image = client.pose_estimator.generate_pose_image(
                body_position=client.current_position,
                pose=client.current_pose,
                expression_params=client.expression_params,
            )

        # Build prompt from current state
        prompt = self._build_frame_prompt(client)

        # Render
        t_start = time.time()
        pet_frame = client.renderer.render(
            camera_frame=camera_frame,
            pose_image=pose_image,
            prompt=prompt,
            strength=0.5,
        )
        render_ms = (time.time() - t_start) * 1000

        client.frame_count += 1

        # Encode and send
        buf = io.BytesIO()
        pet_frame.save(buf, format="PNG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        fps = client.renderer.current_fps

        await websocket.send(json.dumps({
            "type": "frame",
            "data": frame_b64,
            "fps": round(fps, 1),
            "render_ms": round(render_ms, 1),
            "frame_id": client.frame_count,
            "timestamp": time.time(),
        }))

    async def _handle_interaction(self, websocket, client: ClientState, msg: dict):
        """Process an interaction event from the mobile client."""
        action = msg.get("action", "poke")
        intensity = msg.get("intensity", 0.5)

        # Update pet state based on interaction
        if action == "poke":
            client.current_pose = "jumping"
            client.expression_params["body_bounce"] = 0.8
        elif action == "pet":
            client.current_pose = "sitting"
            client.expression_params["eye_openness"] = 0.4
            client.expression_params["tail_wag"] = 0.6
        elif action == "feed":
            client.current_pose = "sitting"
            client.expression_params["mouth_curve"] = 0.5
        elif action == "call":
            client.current_pose = "standing"
            client.expression_params["ear_angle"] = 0.5
        elif action == "scare":
            client.current_pose = "curled"
            client.expression_params["eye_openness"] = 0.3

        # Acknowledge
        await websocket.send(json.dumps({
            "type": "interaction_ack",
            "action": action,
        }))

    def _build_frame_prompt(self, client: ClientState) -> str:
        """Build the text prompt for the current frame."""
        pose_prompts = {
            "sitting": "sitting calmly, looking at camera",
            "standing": "standing on all fours, alert",
            "lying": "lying down, relaxed",
            "walking": "walking gently, in motion",
            "jumping": "mid-jump, playful, bouncy",
            "curled": "curled up in a ball, sleepy",
        }
        pose_desc = pose_prompts.get(client.current_pose, "sitting")

        # Add expression context
        eye_open = client.expression_params.get("eye_openness", 0.8)
        mouth = client.expression_params.get("mouth_curve", 0)
        tail = client.expression_params.get("tail_wag", 0)

        if eye_open < 0.3:
            pose_desc += ", eyes half closed, sleepy"
        if mouth > 0.4:
            pose_desc += ", happy expression, slight smile"
        if tail > 0.5:
            pose_desc += ", tail wagging enthusiastically"

        return pose_desc

    async def _send_error(self, websocket, message: str):
        """Send an error message to the client."""
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "message": message,
            }))
        except Exception:
            pass


# --- CLI Entry Point ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sprite WebSocket streaming server."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lora", default=None, help="Path to LoRA .safetensors")
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument("--max-clients", type=int, default=4)

    args = parser.parse_args()

    server = SpriteServer(
        host=args.host,
        port=args.port,
        lora_path=args.lora,
        model_id=args.model,
        max_clients=args.max_clients,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[Sprite] Shutting down...")
        asyncio.run(server.stop())


if __name__ == "__main__":
    main()

# Sprite — Product Plan

> **Tagline**: A pet that lives in your world. Drawn by AI. Alive in your camera.
> **Status**: Concept / MVP (pre-seed)
> **Last updated**: 2026-06-05

---

## 1. Thesis

AR pets exist. Tamagotchi existed. AI companions exist. But none of them combine all three: **generative, real-time, and physically present**.

**Sprite** is an AI pet that lives in your phone's camera. Every Sprite is unique — generated from a single image or description, never seen before, never to be replicated. It sits on your desk. It hides behind your coffee mug. It recognizes your voice, reacts to your touch, and learns your routines.

This is not a 3D model with ARKit. This is a generative model rendering in real time, frame by frame, composited onto your camera feed. The pet is drawn into existence — and drawn differently each time, but with consistent identity.

---

## 2. Core Experience

```
[Open Sprite app] → [Camera viewfinder]
                          ↓
[Generate your Sprite: upload a drawing, a photo, or describe it]
                          ↓
[A unique creature appears in your camera — sitting on your desk]
                          ↓
┌─────────────────────────────────────────────────┐
│  👆 Tap to poke → it jumps                       │
│  👆 Swipe to pet → it leans in                   │
│  🗣️ "Hello!" → it tilts its head                 │
│  🗣️ "Are you hungry?" → it nods                  │
│  📱 Tilt phone → it slides across the surface    │
│  ⏰ 7PM → it appears at the edge, waiting         │
│  🌙 Midnight → it's asleep on your nightstand    │
└─────────────────────────────────────────────────┘
```

---

## 3. Generative Identity

Every Sprite is born from a single seed image or text prompt.

### Appearance Generation (once, at creation)
```
User provides: [photo of their cat] OR [drawing] OR ["a small round creature with rabbit ears and fox tail"]
        ↓
Image generation model (Stable Diffusion / DALL-E / Midjourney API)
        ↓
Base appearance: 4-8 reference frames (front, side, back, expressions)
        ↓
LoRA fine-tune: lock the identity into a consistent character
        ↓
Sprite identity is born
```

### Real-Time Rendering (every frame) — Powered by StreamDiffusionV2

```
Camera frame (720p, 30fps)
        ↓
Scene understanding: detect surfaces, lighting, occluders
        ↓
Pet state update: position, pose, expression, reaction
        ↓
Pose Estimator: generate ControlNet skeleton from pet state
        ↓
StreamDiffusionV2: SD Turbo img2img + Identity LoRA + ControlNet
        ↓  (1 denoising step, ~16ms on A100, ~5ms on H100)
Alpha mask extraction + composite with camera frame
        ↓
Display at 30-60 FPS
```

### Frame Rate Benchmarks (StreamDiffusionV2, Nov 2025)

| GPU | Model | Steps | FPS | Resolution |
|-----|-------|-------|-----|------------|
| A100 | SD Turbo 1.3B | 1 | **61.6** | 512×512 |
| A100 | SDXL Turbo 14B | 1 | **31.6** | 512×512 |
| H100 | SD Turbo 1.3B | 1 | **90+** | 512×512 |
| RTX 4090 | SD Turbo 1.3B | 1 | **45** | 512×512 |

With LoRA: no significant overhead (<2% FPS impact).
With ControlNet: ~15% FPS reduction.

### Identity Consistency Strategy

```
Pet photos (5-15) ──→ LoRA fine-tuning (~15 min on A100)
                            ↓
                  Identity LoRA (.safetensors)
                            ↓
[Every frame] ──→ SD Turbo img2img + LoRA + ControlNet ──→ Consistent pet
```

- **LoRA rank 16**: Captures face, fur pattern, body shape, color palette
- **IP-Adapter** (optional): Additional identity anchoring for edge cases
- **ControlNet OpenPose**: Controls pet body position and posture per frame
- **Prompt anchoring**: Identity description prepended to every frame prompt

---

## 4. AR Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     MOBILE DEVICE                         │
│                                                          │
│  Camera ──→ Frame Grab ──→ Scene Understanding           │
│                               │                          │
│                               ├→ Surface Detection       │
│                               ├→ Lighting Estimation     │
│                               └→ Occlusion Mask          │
│                                    │                     │
│  ┌─────────────────────────────────┼──────────────────┐  │
│  │  Interaction                  │                    │  │
│  │  ├→ Touch (poke/pet/drag)     │                    │  │
│  │  └→ Voice (ASR → intent)     │                    │  │
│  └───────────────────────────────┼────────────────────┘  │
│                                    │                     │
│  ┌─────────────────────────────────┼──────────────────┐  │
│  │  Pet Engine                    │                    │  │
│  │  ├→ State machine (idle/play/  │                    │  │
│  │  │   hungry/sleeping/curious)  │                    │  │
│  │  ├→ Personality weights        │                    │  │
│  │  └→ Memory (routines, likes)  │                    │  │
│  └───────────────────────────────┼────────────────────┘  │
│                                    │                     │
│  ┌─────────────────────────────────┼──────────────────┐  │
│  │  Generative Renderer           │                    │  │
│  │  ├→ Identity LoRA (persistent) │                    │  │
│  │  ├→ Pose+Expression control    │                    │  │
│  │  └→ Lighting-adaptive shading  │                    │  │
│  └───────────────────────────────┼────────────────────┘  │
│                                    │                     │
│  Composite ──→ Display             │                     │
└────────────────────────────────────┼─────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │         CLOUD (generation)        │
                    │  ┌────────────────────────────┐  │
                    │  │ Identity Generation         │  │
                    │  │ (one-time, on creation)     │  │
                    │  └────────────────────────────┘  │
                    │  ┌────────────────────────────┐  │
                    │  │ LoRA Training               │  │
                    │  │ (fine-tune identity)        │  │
                    │  └────────────────────────────┘  │
                    └────────────────────────────────────┘
```

### Generation APIs (2026 landscape)

| API | Use Case | Latency | Cost |
|-----|----------|---------|------|
| **Stability AI (SDXL + LoRA)** | Identity generation + fine-tuning | 3-8s | ~$0.04/image |
| **DALL-E 3 API** | Base appearance generation | 5-15s | ~$0.04/image |
| **Runway Gen-3** | Video-consistent pet animation | 2-5s/video | ~$0.05/sec |
| **ComfyUI + LCM-LoRA** | Real-time frame rendering (self-host) | 0.05-0.2s | GPU cost |
| **SD Turbo / SDXL Turbo** | 1-step diffusion for interactive fps | 0.05-0.1s | GPU cost |

**Recommended production stack**: Self-hosted ComfyUI with SDXL Turbo + identity LoRA on a cloud GPU, serving frames to the mobile app via WebSocket.

---

## 5. Pet Behavior Engine

### State Machine
```
                    ┌──────────┐
          ┌────────→│ SLEEPING │←────────┐
          │         └────┬─────┘         │
          │              │ wake up       │ time + no interaction
          │         ┌────↓─────┐         │
          │  hungry │  IDLE    │ bored   │
          │         └──┬───┬───┘         │
          │            │   │             │
     ┌────↓─────┐      │   └──────┐      │
     │ HUNGRY   │←─────┘          │      │
     └────┬─────┘                 │      │
          │ feed             ┌────↓─────┐│
     ┌────↓─────┐            │ PLAYING  ││
     │ EATING   │            └────┬─────┘│
     └────┬─────┘                 │      │
          │ done             tired│      │
          └────────→ IDLE ←───────┘      │
                       │                 │
                       └─────────────────┘
```

### Personality Dimensions (0-1, randomized at creation)
- **Curiosity**: How likely to explore new objects in the camera
- **Sociability**: How strongly it responds to voice and attention
- **Playfulness**: How easily it enters play mode
- **Independence**: How long before it gets bored without interaction
- **Affection**: How much it seeks physical touch

### Emotional State (dynamic, updated every interaction)
| State | Trigger | Expression |
|-------|---------|------------|
| Happy | Petted, fed, greeted | Bouncing, tail wagging, bright colors |
| Curious | New object in view | Tilting head, approaching slowly |
| Sleepy | Night time, inactivity | Yawning, curling up, dimmed colors |
| Hungry | 4+ hours since last feed | Following user, making sounds |
| Excited | User returns after long absence | Running in circles, jumping |
| Scared | Loud noise, sudden movement | Hiding, shrinking, dark colors |

---

## 6. Interaction Model

### Touch
| Gesture | Action | Pet Response |
|---------|--------|-------------|
| Single tap | Poke | Jump/surprised, then curious |
| Double tap | Call | Comes to tapped location |
| Swipe (on pet) | Pet | Leans in, happy, closes eyes |
| Long press | Pick up | Lifted, wiggling |
| Drag | Move | Follows finger to new surface |
| Shake phone | Surprise | Startled, hides, then peeks out |

### Voice (on-device ASR → intent classification)
| User Says | Intent | Response |
|-----------|--------|----------|
| "Hello!" / "Hi [name]" | Greet | Approaches, tilts head, wags |
| "Come here" | Summon | Moves toward camera center |
| "Are you hungry?" | Query | Nods or shakes head |
| "Good boy/girl/[name]" | Praise | Happy dance, glowing |
| "Sit" / "Stay" | Command | Obeys (obedience × personality) |
| "What's that?" | Point attention | Looks where camera is pointing |
| "Goodnight" | Dismiss | Curls up, dims, enters sleep |
| [Name] | Attention | Looks at camera, ears perk up |

---

## 7. Memory Layer

Like 念念, Sprite remembers. But here it's about the relationship:

```
Day 1: Sprite is shy. Stays at the edges. Watches.
Day 3: Approaches when you say its name.
Day 7: Sits on your desk while you work. Knows your morning routine.
Day 30: Appears at the door when you come home. Has a favorite spot.
Day 100: Knows your mood from your voice. Brings you "gifts" (generated objects).
```

### Memory Data Model
```
Sprite.memory = {
  "name": "given by user",
  "birthday": "creation date",
  "favorite_spot": "top-right of desk",
  "favorite_food": "learned from feeding interactions",
  "owner_routine": {
    "wake_up": "07:30",
    "leave": "08:45",
    "return": "18:30",
    "sleep": "23:00"
  },
  "tricks_learned": ["sit", "come", "fetch"],
  "mood_history": [...],
  "shared_moments": [
    {date: "2026-06-05", event: "first time it sat on my hand"},
    {date: "2026-06-10", event: "brought me a generated flower"}
  ]
}
```

---

## 8. MVP Scope (This Repository)

### a) Concept Design
→ PLAN.md + bilingual READMEs

### b) Engine Prototype
```
src/sprite/
├── pet_engine.py      # Behavior state machine, personality, emotions
├── generator.py       # Generative model pipeline (image-to-pet + rendering)
├── ar_pipeline.py     # AR rendering simulation (camera→gen→composite)
├── interaction.py     # Touch gesture + voice command models
├── memory.py          # Pet memory & relationship tracking
```

### c) Interactive Prototype (Streamlit)
```
app/
└── streamlit_app.py   # Simulated AR pet experience:
                        - Pet creation (describe → generate appearance)
                        - Live camera simulation (upload photo → pet composited)
                        - Touch interaction (click/pet/drag the sprite)
                        - Voice input (type to simulate speech)
                        - Memory timeline
```

---

## 9. Why Now?

1. **Real-time generation is here**: SD Turbo, LCM, and distilled models can generate images in 50-100ms. Fast enough for interactive AR.
2. **On-device ML is viable**: iPhone 15 Pro's Neural Engine + Core ML can run distilled diffusion models. Apple's MLX framework enables local fine-tuning.
3. **AR adoption is plateauing**: Pokémon GO was 2016. ARKit/Meta Quest haven't delivered the killer consumer app. Generative AR is the next wave.
4. **Tamagotchi nostalgia + AI novelty**: Perfect cultural moment — 90s kids who loved Tamagotchi now have disposable income and AI curiosity.
5. **No competitor**: There is no generative AR pet app. The gap is wide open.

---

## 10. Inspirations
- **Tamagotchi** — proved digital pets have mass emotional appeal
- **Pokémon GO** — proved AR + creatures = cultural phenomenon
- **Replika** — proved AI companions can form deep bonds
- **Genie/Gen** — proved image→interactive world is possible
- **SD Turbo / LCM** — proved real-time image generation is viable
- **Nintendogs** — proved touch + voice pet interaction is magical

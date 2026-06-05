> 🌐 [中文](README.zh-CN.md) | **English**

# Sprite ✨📱

**Generative AR Pet** — A pet that lives in your camera. Drawn by AI. Every one is unique.

## What Is Sprite?

Sprite is an AI pet that lives in your phone's camera viewfinder. It's not a 3D model — it's drawn in real time by a generative model, composited onto your camera feed. Every Sprite is born unique from a single image or description. No two are alike.

- 🎨 **Generative Identity** — Upload a drawing or describe it. Your Sprite is one of a kind.
- 📱 **Lives in Your Camera** — Sits on your desk, hides behind your coffee, follows you around.
- 👆 **Touch** — Tap to poke, swipe to pet, drag to move.
- 🗣️ **Voice** — Talk to it. It tilts its head. It learns your voice.
- 🧠 **Memory** — Day 1 it's shy. Day 30 it knows your routine.

## Quick Start

```bash
git clone https://github.com/SpencerRaw/sprite.git
cd sprite
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project Structure

```
sprite/
├── PLAN.md                   # Product plan & architecture
├── README.md / README.zh-CN.md
├── requirements.txt
├── app/
│   └── streamlit_app.py      # Interactive AR pet demo
├── src/sprite/
│   ├── pet_engine.py         # Behavior, personality, emotions
│   ├── generator.py          # Generative model pipeline
│   ├── ar_pipeline.py        # AR rendering simulation
│   ├── interaction.py        # Touch + voice interaction
│   └── memory.py             # Pet memory & relationship
└── data/
```

## License

MIT — see [LICENSE](LICENSE)

---

*A pet that lives in your world. Drawn by AI. Alive in your camera.*

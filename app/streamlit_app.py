"""Aura — Generative AR Pet Prototype.

Streamlit app simulating the AR pet experience.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import streamlit.components.v1 as components
import time
import random

from aura.pet_engine import Pet, PetState, Emotion, Personality, Needs
from aura.generator import IdentityGenerator, FrameRenderer, AuraAppearance
from aura.ar_pipeline import ARPipeline, CameraFrame, SceneAnalyzer
from aura.interaction import TouchRecognizer, VoiceRecognizer, Gesture, VoiceIntent
from aura.memory import PetMemory


st.set_page_config(
    page_title="Aura — Your AR Pet",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS ---
st.markdown("""
<style>
    .aura-title { font-size: 2.5rem; font-weight: 600; letter-spacing: -0.02em; color: #2d2d2d; }
    .aura-subtitle { font-size: 1.1rem; color: #888; }
    .pet-canvas-container { background: #1a1a2e; border-radius: 16px; padding: 10px;
                            text-align: center; border: 2px solid #333; }
    .status-bar { background: #f5f0e8; padding: 0.8rem 1.2rem; border-radius: 12px;
                  font-size: 0.95rem; margin: 0.5rem 0; }
    .interaction-btn { font-size: 1.5rem !important; min-height: 60px !important; }
    .memory-card { background: #faf6ef; padding: 1rem; border-radius: 12px;
                   border-left: 3px solid #c4a882; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# --- Init Session ---
if "pet" not in st.session_state:
    st.session_state.pet = None
if "appearance" not in st.session_state:
    st.session_state.appearance = None
if "renderer" not in st.session_state:
    st.session_state.renderer = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "ar_pipeline" not in st.session_state:
    st.session_state.ar_pipeline = ARPipeline()
if "touch" not in st.session_state:
    st.session_state.touch = TouchRecognizer()
if "voice" not in st.session_state:
    st.session_state.voice = VoiceRecognizer()
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ✨ Aura")
    st.caption("Your Generative AR Pet")

    page = st.radio("Navigate", ["🎨 Create Aura", "📱 AR View", "📖 Memory"],
                    label_visibility="collapsed")

    st.divider()
    if st.session_state.pet:
        pet = st.session_state.pet
        st.markdown(f"**{pet.name}** — {pet.age_seconds/3600:.1f}h old")
        st.caption(pet.get_status_text())

        needs = pet.get_needs_status()
        st.progress(1 - needs["hunger"], text="🍖 Hunger")
        st.progress(needs["energy"], text="⚡ Energy")
        st.progress(needs["social"], text="💕 Social")
        st.progress(needs["curiosity"], text="🔍 Curiosity")

    st.divider()
    st.caption("[GitHub](https://github.com/SpencerRaw/aura)")


# ============================================================
# PAGE 1: CREATE SPRITE
# ============================================================
if page == "🎨 Create Aura":
    st.markdown('<p class="aura-title">🎨 Create Your Aura</p>', unsafe_allow_html=True)
    st.markdown('<p class="aura-subtitle">Describe it. Draw it. Every Aura is one of a kind.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Describe Your Aura")
        desc = st.text_area(
            "What does it look like?",
            placeholder="e.g., A small fluffy creature with big round eyes, pastel purple fur, tiny wings, and a sparkly tail...",
            height=120,
            label_visibility="collapsed",
        )
        name = st.text_input("Give it a name", value="Aura", label_visibility="collapsed")

        if st.button("✨ Birth a Aura", type="primary", use_container_width=True):
            if desc.strip():
                gen = IdentityGenerator()
                appearance = gen.generate_from_description(desc)
                pet = Pet(id=f"aura_{random.randint(1000,9999)}", name=name or "Aura")
                pet.appearance_prompt = appearance.base_prompt
                renderer = FrameRenderer(appearance)
                memory = PetMemory(pet_id=pet.id, pet_name=pet.name)

                st.session_state.pet = pet
                st.session_state.appearance = appearance
                st.session_state.renderer = renderer
                st.session_state.memory = memory
                st.session_state.voice = VoiceRecognizer(pet.name)
                st.session_state.messages = []

                memory.add_moment("birth", f"{name} was born from your description")
                st.success(f"{name} is born! Switch to AR View to meet them.")
                st.rerun()
            else:
                st.warning("Describe your Aura first!")

    with col2:
        st.markdown("### Preview")
        if st.session_state.appearance:
            app = st.session_state.appearance
            colors = " ".join(f'<span style="display:inline-block;width:20px;height:20px;background:{c};border-radius:4px;"></span>' for c in app.color_palette)
            st.markdown(f"""
            <div class="memory-card">
                <strong>Body:</strong> {app.body_shape}<br>
                <strong>Eyes:</strong> {app.eye_style}<br>
                <strong>Colors:</strong> {colors}<br>
                <strong>Size:</strong> {app.size_category}<br>
                <strong>Special:</strong> {', '.join(app.special_features) if app.special_features else 'none'}<br>
                <strong>Prompt:</strong> <small>{app.base_prompt[:200]}...</small>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.pet:
                pers = st.session_state.pet.personality
                st.markdown(f"""
                <div class="memory-card">
                    <strong>Personality:</strong> {pers.describe()}<br>
                    <small>Curiosity: {pers.curiosity:.0%} · Sociability: {pers.sociability:.0%} · Playfulness: {pers.playfulness:.0%} · Independence: {pers.independence:.0%} · Affection: {pers.affection:.0%}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Describe your Aura on the left to see a preview.")


# ============================================================
# PAGE 2: AR VIEW
# ============================================================
elif page == "📱 AR View":
    if not st.session_state.pet:
        st.warning("Create your Aura first!")
        st.stop()

    pet = st.session_state.pet
    renderer = st.session_state.renderer
    memory = st.session_state.memory
    ar = st.session_state.ar_pipeline
    voice = st.session_state.voice

    st.markdown(f'<p class="aura-title">📱 {pet.name}</p>', unsafe_allow_html=True)

    # Update pet state
    pet.update(0.5)  # 500ms step
    st.session_state.frame_count += 1

    # Render current frame
    frame_data = renderer.render_frame(
        pet.expression_params,
        position=pet.position,
        size=pet.size,
    )

    # AR composite
    camera = CameraFrame()
    composited = ar.process_frame(camera, frame_data)

    # --- Pet Canvas ---
    canvas_html = render_pet_canvas(composited, pet, memory)
    components.html(canvas_html, height=520)

    # Status bar
    st.markdown(f"""
    <div class="status-bar">
        {pet.get_status_text()} &nbsp;|&nbsp;
        Needs: 🍖{1-pet.needs.hunger:.0%} ⚡{pet.needs.energy:.0%} 💕{pet.needs.social:.0%} 🔍{pet.needs.curiosity_satisfied:.0%}
        &nbsp;|&nbsp; {memory.relationship_stage.title()}
    </div>
    """, unsafe_allow_html=True)

    # --- Interactions ---
    st.markdown("### 👆 Interact")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("👆 Poke", use_container_width=True, key="poke"):
            pet.interact("poke")
            memory.record_interaction("poke")
            st.session_state.messages.append(f"👆 You poked {pet.name}!")
            st.rerun()
    with col2:
        if st.button("🤚 Pet", use_container_width=True, key="pet_btn"):
            pet.interact("pet")
            memory.record_interaction("pet")
            st.session_state.messages.append(f"🤚 You petted {pet.name}...")
            if memory.relationship_stage == "stranger" and memory.total_interactions >= 10:
                memory.add_moment("first_pet", f"{pet.name} let you pet them for the first time")
            st.rerun()
    with col3:
        if st.button("🍖 Feed", use_container_width=True, key="feed"):
            pet.interact("feed")
            memory.record_interaction("feed")
            st.session_state.messages.append(f"🍖 You fed {pet.name}!")
            if pet.needs.hunger < 0.3:
                st.session_state.messages.append(f"😋 {pet.name} looks satisfied!")
            st.rerun()
    with col4:
        if st.button("📢 Call", use_container_width=True, key="call"):
            pet.interact("call")
            memory.record_interaction("call")
            st.session_state.messages.append(f"📢 You called {pet.name}!")
            st.rerun()

    # Voice input
    st.markdown("### 🗣️ Talk to Your Aura")
    voice_text = st.text_input("Say something...", placeholder='"Hello!" or "Good boy!" or "Are you hungry?"',
                               label_visibility="collapsed")
    if voice_text:
        result = voice.recognize(voice_text)
        if result.intent and result.confidence > 0.3:
            action = voice.get_intent_action(result.intent)
            pet.interact(action)
            memory.record_interaction(action)
            st.session_state.messages.append(f"🗣️ You: \"{voice_text}\" → {pet.name} reacts!")
            st.rerun()
        else:
            st.session_state.messages.append(f"🗣️ You: \"{voice_text}\" → {pet.name} tilts their head...")

    # Message log
    if st.session_state.messages:
        with st.expander(f"💬 Log ({len(st.session_state.messages)} messages)", expanded=False):
            for msg in reversed(st.session_state.messages[-20:]):
                st.caption(msg)


# ============================================================
# PAGE 3: MEMORY
# ============================================================
elif page == "📖 Memory":
    if not st.session_state.memory:
        st.warning("Create your Aura first!")
        st.stop()

    memory = st.session_state.memory
    pet = st.session_state.pet

    st.markdown(f'<p class="aura-title">📖 {pet.name}\'s Memory</p>', unsafe_allow_html=True)

    # Relationship stage
    stage_emoji = {"stranger": "🌱", "acquaintance": "🌿", "friend": "🪴",
                   "companion": "🌳", "family": "🏡"}
    emoji = stage_emoji.get(memory.relationship_stage, "❓")
    st.markdown(f"### {emoji} {memory.relationship_stage.title()}")

    stage_info = {
        "stranger": "They're still getting to know you. Be patient, be gentle.",
        "acquaintance": "They recognize you now. They're starting to trust.",
        "friend": "They're happy to see you. Playtime is the best time.",
        "companion": "You're their person. They wait for you.",
        "family": "Unbreakable bond. They know your heart.",
    }
    st.caption(stage_info.get(memory.relationship_stage, ""))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Age", f"{memory.age_days} days")
    with c2:
        st.metric("Interactions", memory.total_interactions)
    with c3:
        st.metric("Milestones", len(memory.moments))

    # Timeline
    st.markdown("### 🕰️ Milestone Timeline")
    if memory.moments:
        for m in reversed(memory.moments):
            st.markdown(f"""
            <div class="memory-card">
                <strong>Day {m.day}</strong> — {m.event}<br>
                <small>{m.description}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No milestones yet. Start interacting with your Aura!")

    # Behavior by stage
    st.markdown("### 🧠 Behavior Profile")
    behavior = memory.get_stage_behavior()
    b_cols = st.columns(3)
    with b_cols[0]:
        st.metric("Trust", f"{behavior['trust']:.0%}")
    with b_cols[1]:
        st.metric("Approach", f"{1-behavior['approach_distance']:.0%}")
    with b_cols[2]:
        st.metric("Reaction", f"{behavior['reaction_speed']:.0%}")


# ============================================================
# PET CANVAS RENDERER (HTML/JS)
# ============================================================

def render_pet_canvas(frame: dict, pet: Pet, memory: PetMemory) -> str:
    """Render the Aura on a canvas simulating the AR view."""
    colors = frame.get("colors", ["#FF6B6B", "#FFE66D", "#FF8E72"])
    eye_open = frame.get("eye_scale_y", 0.8)
    mouth = frame.get("mouth_curve", 0)
    bounce = frame.get("body_bounce_y", 0)
    glow = frame.get("glow_intensity", 0)
    body_shape = frame.get("body_shape", "round and bouncy")
    saturation = frame.get("color_saturation", 1.0)

    # Color manipulation for mood
    c1, c2 = colors[0], colors[1] if len(colors) > 1 else colors[0]

    greeting = memory.get_stage_greeting()

    return f"""
    <div class="pet-canvas-container">
        <canvas id="auraCanvas" width="600" height="460"
                style="cursor:pointer; border-radius:12px;"></canvas>
        <div style="color:#ccc; margin-top:8px; font-style:italic;">{greeting}</div>
    </div>
    <script>
    const canvas = document.getElementById('auraCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W/2, cy = H/2 + {bounce};

    // Background (simulated desk surface)
    const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
    bgGrad.addColorStop(0, '#2c2c3e');
    bgGrad.addColorStop(0.55, '#3a3a4e');
    bgGrad.addColorStop(0.55, '#5a4a3a');
    bgGrad.addColorStop(1, '#4a3a2a');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    // Shadow under pet
    ctx.fillStyle = `rgba(0,0,0,${{0.2 + {frame.get('shadow_opacity', 0.3) * 0.3}}})`;
    ctx.beginPath();
    ctx.ellipse(cx, cy + 100, 80, 20, 0, 0, Math.PI * 2);
    ctx.fill();

    // Glow aura
    if ({glow} > 0.1) {{
        const glowGrad = ctx.createRadialGradient(cx, cy, 40, cx, cy, 140);
        glowGrad.addColorStop(0, `rgba(${{parseInt({glow}*120)}}, ${{parseInt({glow}*120)}}, 255, ${{glow*0.4}})`);
        glowGrad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = glowGrad;
        ctx.fillRect(cx - 140, cy - 140, 280, 280);
    }}

    // Body
    const bodyColor = '{c1}';
    const bodyGrad = ctx.createRadialGradient(cx - 15, cy - 15, 10, cx, cy, 70);
    bodyGrad.addColorStop(0, lightenColor(bodyColor, 0.3));
    bodyGrad.addColorStop(1, bodyColor);
    ctx.fillStyle = bodyGrad;

    // Different body shapes
    if ('{body_shape}'.includes('round')) {{
        ctx.beginPath();
        ctx.arc(cx, cy, 65, 0, Math.PI * 2);
        ctx.fill();
    }} else if ('{body_shape}'.includes('fluffy')) {{
        // Fluffy: multiple overlapping circles
        for (let i = 0; i < 5; i++) {{
            const angle = (i / 5) * Math.PI * 2;
            const rx = cx + Math.cos(angle) * 25;
            const ry = cy + Math.sin(angle) * 25;
            ctx.beginPath();
            ctx.arc(rx, ry, 35 + Math.random() * 10, 0, Math.PI * 2);
            ctx.fill();
        }}
        ctx.beginPath();
        ctx.arc(cx, cy, 50, 0, Math.PI * 2);
        ctx.fill();
    }} else {{
        ctx.beginPath();
        ctx.arc(cx, cy, 60, 0, Math.PI * 2);
        ctx.fill();
    }}

    // Accent color patches
    const accentColor = '{c2}';
    ctx.fillStyle = accentColor;
    ctx.beginPath();
    ctx.arc(cx - 35, cy - 25, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + 35, cy + 20, 15, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    const eyeY = cy - 10;
    const eyeScaleY = {eye_open};

    // Left eye
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.ellipse(cx - 22, eyeY, 16, 16 * eyeScaleY, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#111';
    ctx.beginPath();
    ctx.arc(cx - 20, eyeY, 7, 0, Math.PI * 2);
    ctx.fill();
    // Eye shine
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(cx - 23, eyeY - 5, 3, 0, Math.PI * 2);
    ctx.fill();

    // Right eye
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.ellipse(cx + 22, eyeY, 16, 16 * eyeScaleY, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#111';
    ctx.beginPath();
    ctx.arc(cx + 24, eyeY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(cx + 21, eyeY - 5, 3, 0, Math.PI * 2);
    ctx.fill();

    // Mouth
    const mouthY = cy + 25;
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const mouthCurve = {mouth};
    if (mouthCurve > 0.2) {{
        // Happy mouth
        ctx.arc(cx, mouthY - 5, 15, 0.1, Math.PI - 0.1);
    }} else if (mouthCurve < -0.2) {{
        // Sad mouth
        ctx.arc(cx, mouthY + 15, 15, Math.PI + 0.3, -0.3);
    }} else {{
        // Neutral
        ctx.moveTo(cx - 12, mouthY);
        ctx.lineTo(cx + 12, mouthY);
    }}
    ctx.stroke();

    // Ears
    const earAngle = {frame.get('ear_angle', 0)} * Math.PI / 180;
    ctx.fillStyle = bodyColor;
    // Left ear
    ctx.beginPath();
    ctx.save();
    ctx.translate(cx - 45, cy - 50);
    ctx.rotate(-0.3 + earAngle);
    ctx.fillRect(-8, -30, 16, 30);
    ctx.restore();
    // Right ear
    ctx.beginPath();
    ctx.save();
    ctx.translate(cx + 45, cy - 50);
    ctx.rotate(0.3 - earAngle);
    ctx.fillRect(-8, -30, 16, 30);
    ctx.restore();

    // Tail
    const tailAngle = {frame.get('tail_angle', 0)} * Math.PI / 180;
    ctx.strokeStyle = bodyColor;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(cx + 60, cy + 10);
    ctx.quadraticCurveTo(cx + 100 + Math.sin(tailAngle) * 20, cy - 30, cx + 110, cy - 60 + Math.cos(tailAngle) * 10);
    ctx.stroke();
    // Tail tip
    ctx.fillStyle = accentColor;
    ctx.beginPath();
    ctx.arc(cx + 110, cy - 60 + Math.cos(tailAngle) * 10, 8, 0, Math.PI * 2);
    ctx.fill();

    // Legs
    ctx.fillStyle = bodyColor;
    [cx - 25, cx + 25].forEach(lx => {{
        ctx.beginPath();
        ctx.ellipse(lx, cy + 70, 14, 10, 0, 0, Math.PI * 2);
        ctx.fill();
    }});

    function lightenColor(hex, amount) {{
        const r = parseInt(hex.slice(1,3), 16);
        const g = parseInt(hex.slice(3,5), 16);
        const b = parseInt(hex.slice(5,7), 16);
        const lr = Math.min(255, r + (255-r) * amount);
        const lg = Math.min(255, g + (255-g) * amount);
        const lb = Math.min(255, b + (255-b) * amount);
        return `rgb(${{Math.floor(lr)}},${{Math.floor(lg)}},${{Math.floor(lb)}})`;
    }}

    // Click to poke
    canvas.addEventListener('click', (e) => {{
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
        if (dist < 100) {{
            // Poke animation
            ctx.save();
            ctx.translate(cx, cy);
            const shakeX = (Math.random() - 0.5) * 10;
            const shakeY = (Math.random() - 0.5) * 10;
            // Brief flash
            ctx.fillStyle = 'rgba(255,255,200,0.3)';
            ctx.beginPath();
            ctx.arc(0, 0, 80, 0, Math.PI * 2);
            ctx.fill();
        }}
    }});
    </script>
    """

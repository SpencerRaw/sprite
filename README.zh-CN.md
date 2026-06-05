> 🌐 [English](README.md) | **中文**

# Aura 精灵 ✨📱

**生成式AR虚拟宠物**——活在摄像头里的AI宠物。每一只都独一无二。

## 这是什么？

Aura 是一只活在手机摄像头里的AI宠物。它不是3D模型——它由生成模型实时绘制，叠加在你的摄像头画面上。每只 Aura 从一张图或一句话中诞生，独一无二。

- 🎨 **生成式身份** — 上传一张画或描述它。你的精灵举世无双。
- 📱 **活在摄像头里** — 蹲在你桌上，躲在杯子后面，跟着你走。
- 👆 **触控** — 点一下戳它，划一下摸它，拖着它走。
- 🗣️ **语音** — 跟它说话。它歪头。它认得你的声音。
- 🧠 **记忆** — 第1天它害羞。第30天它懂你的日常。

## 快速开始

```bash
git clone https://github.com/SpencerRaw/aura.git
cd aura
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 项目结构

```
aura/
├── PLAN.md                   # 产品计划与架构
├── README.md / README.zh-CN.md
├── requirements.txt
├── app/
│   └── streamlit_app.py      # 交互式AR宠物演示
├── src/aura/
│   ├── pet_engine.py         # 行为状态机、性格、情绪
│   ├── generator.py          # 生成模型管线
│   ├── ar_pipeline.py        # AR渲染模拟
│   ├── interaction.py        # 触控 + 语音交互
│   └── memory.py             # 宠物记忆与关系
└── data/
```

## 许可证

MIT — 详见 [LICENSE](LICENSE)

---

*活在你世界里的宠物。AI画的。活在你摄像头里。*

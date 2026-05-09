# 🎬 Promo Video Pipeline

> 把任意 GitHub 项目 → 30 秒 AI 宣传视频。5 个 Phase，6 个 Agent，全链路 trace。
> 结合 [Open Design](https://github.com/nexu-io/open-design) 的 `hyperframes` skill 做 motion graphics，[Remotion](https://www.remotion.dev/) 做剪辑合成，MusicGen 出 BGM，edge-tts 出双语配音。

<p align="center">
  <a href="README.md"><b>简体中文</b></a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a>
</p>

---

## ✨ 核心特点

- **5 Phase 流水线**：分析 → 视觉资产 + 录屏（并行）→ Remotion 合成 → BGM → 配音
- **6 个 LLM Agent**：每个阶段一个独立 Agent，基于 Anthropic SDK
- **OpenDesign × HyperFrames**：自然语言 → motion graphics .mp4（HTML+GSAP+帧捕获）
- **双语配音**：edge-tts 中英 voiceover + ducking + amix
- **全链路 Observability**：events.jsonl + 每 agent 独立 JSONL + Phoenix UI 看 LLM span
- **人在回路**：5 个人工拍板节点，user 只讨论 + 审阅，不写代码

---

## 🏗 架构

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ─────────────────────┐
│   git clone → 读 README/源码 → project_brief.md          │
│   ⏸ 人工介入 #1：审 brief                               │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · 并行段（双路）──────────────────────────────────┐
│  路 A · Agent 6 OpenDesigner                              │
│       brief → LLM 选 hyperframes skill                    │
│       → 多轮 (User × Agent × OpenDesign)                  │
│       → motion film .mp4                                   │
│       ├─ 采纳为 Hero → run_dir/hero/intro.mp4             │
│       └─ 采纳为 Final → 跳过 Phase 3-5                    │
│                                                           │
│  路 B · Agent 2 SetupRunner                               │
│       检测项目类型 → 装依赖 → 起服务 → 录屏                  │
│       → recordings/test.mp4                                │
│   ⏸ 人工介入 #2：审视觉资产 + 录屏窗口                       │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────┐
│   M3a · cutting_plan.json                                 │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4 │
│   ⏸ 人工介入 #3：剪辑思路 + vN 反馈                          │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────┐
│   M4a · numpy 节拍脚手架（kicks 对齐 scene 切割）           │
│   M4b · MusicGen-small（CUDA fp16，~40s 推理）              │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                      │
│   ⏸ 人工介入 #4：听 BGM 反馈                                │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver（4 步串行）─────────────────┐
│   Step1 · LLM 写 voiceover_script.json (中英双语)         │
│   Step2 · edge-tts 逐句合成 mp3                           │
│   Step3 · voice_timeline 拼成 voice_full.wav              │
│   Step4 · BGM ducking + amix mux → final_zh-CN/en-US.mp4  │
│   ⏸ 人工介入 #5：审脚本                                     │
└──────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🚀 Quick Start

### 1. 环境准备
```bash
# Python venv + 依赖
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# 或
.venv/bin/pip install -r requirements.txt        # *nix

# 必备外部
- ffmpeg (PATH)
- Node.js 24 + pnpm 10 (Phase 3 Remotion + Phase 2A OpenDesign)
- CUDA GPU (可选，Phase 4b MusicGen 加速)
```

### 2. OpenDesign daemon
```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm --filter @open-design/daemon build

# 启 daemon (NODE_TLS_REJECT_UNAUTHORIZED=0 绕开 corp VPN 证书校验)
NODE_TLS_REJECT_UNAUTHORIZED=0 pnpm tools-dev run web
# → Web: http://127.0.0.1:<port>/  Daemon: http://127.0.0.1:<port>/
```

### 3. 配 OpenCode 接火山方舟（或其他 LLM provider）
`~/.config/opencode/opencode.json`：
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "volcark": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://ark.cn-beijing.volces.com/api/coding/v3",
        "apiKey": "<YOUR_KEY>"
      },
      "models": {
        "doubao-seed-code": { "name": "Doubao Seed Code" }
      }
    }
  },
  "model": "volcark/doubao-seed-code"
}
```

### 4. 配 .env
```ini
ANTHROPIC_BASE_URL=<your provider compatible endpoint>
ANTHROPIC_API_KEY=<key>
# Or use ARK_KEY_1
```

### 5. 跑流水线
```bash
# CLI 启 Phase 1
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

# 启 Web UI
.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

---

## 📊 Observability

每条 run 在 `workspace/<project>/runs/<run_id>/` 下持久化：

```
events.jsonl                        全 lifecycle 事件
logs/
  pipeline.jsonl                    全 agent 行级日志
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 session 状态
opendesign_artifacts/               OpenDesign archive 备份
```

**Phoenix UI**: http://localhost:6006/ — 全部 LLM call + verify span 持久化（SQLite at `~/.phoenix/`）

每个 Agent 入口都用 `@traced_agent("Agent N · 子步骤", phase=N)` 装饰，自动 emit `agent_start` / `agent_done` 事件 + 创建 OTEL span，全链路追溯。

---

## 🛠 Tech Stack

| 层 | 技术 |
|---|---|
| Agent runtime | Python 3.13 + Anthropic SDK + 火山方舟 Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| Observability | Phoenix (Arize) + loguru + OTEL |
| 视觉合成 | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| 视频剪辑 | Remotion (React + TSX → mp4) |
| BGM | numpy 节拍脚手架 + facebook/musicgen-small (PyTorch CUDA) |
| 配音 | edge-tts (Microsoft Azure Neural TTS) + ffmpeg ducking |

---

## 📁 项目结构

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2
│   ├── remotion_composer.py   Agent 3
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/
│   ├── bgm_scaffold.py        Agent 4 M4a
│   ├── bgm_musicgen.py        Agent 4 M4b
│   ├── bgm_mux.py             Agent 4 M4c
│   ├── tts_edge.py            Agent 5 Step2
│   ├── voice_timeline.py      Agent 5 Step3
│   ├── bgm_duck_mux.py        Agent 5 Step4
│   ├── opendesign_client.py   Agent 6 daemon HTTP client
│   ├── opendesign_lifecycle.py daemon 启停
│   ├── plan_executor.py       Agent 2 plan 执行
│   ├── recorder.py            Agent 2 ffmpeg 录窗口
│   ├── web_recorder.py        Agent 2 playwright 录 URL
│   ├── remotion_codegen.py    Agent 3 cutting_plan → TSX
│   └── remotion_render.py     Agent 3 npx remotion render
├── observability/
│   ├── tracer.py              Phoenix OTEL setup
│   ├── logger.py              loguru + per-agent JSONL
│   ├── audit.py               @traced_agent decorator + run_context
│   └── events.py              EventBus (events.jsonl)
├── verify/                    ffprobe 验产物
├── web/                       FastAPI + 模板
├── pipeline.py                Pipeline class + 状态机
└── cli.py                     viedo CLI
```

---

## 🤝 致谢

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames 集成
- [remotion](https://github.com/remotion-dev/remotion) · React-based 视频合成
- [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) · 文本驱动音乐生成
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS 包装
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) · LLM observability

---

## 📝 License

Apache-2.0

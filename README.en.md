# 🎬 Promo Video Pipeline

> Turn any GitHub repo into a 30-second AI promo video. 5 phases, 6 agents, full-stack tracing.
> Powered by [Open Design](https://github.com/nexu-io/open-design)'s `hyperframes` skill for motion graphics, [Remotion](https://www.remotion.dev/) for composition, MusicGen for BGM, and edge-tts for bilingual voiceover.

<p align="center">
  <a href="README.md">简体中文</a> ·
  <a href="README.en.md"><b>English</b></a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a>
</p>

<p align="center">
  <img src="docs/workflow.svg" alt="Workflow diagram" width="100%"/>
</p>

---

## ✨ Highlights

- **5-phase pipeline**: analysis → visual asset + recording (parallel) → Remotion composition → BGM → voiceover
- **6 LLM agents**: one per phase, all built on the Anthropic SDK
- **OpenDesign × HyperFrames**: natural-language → motion-graphics .mp4 (HTML+GSAP+frame capture)
- **Bilingual voiceover**: edge-tts (Chinese + English) + ducking + amix
- **Full observability**: events.jsonl + per-agent JSONL + Langfuse UI for LLM spans
- **Human-in-the-loop**: 5 explicit review gates — the user discusses & approves; never writes code

---

## 🏗 Architecture

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ─────────────────────┐
│   git clone → read README/source → project_brief.md      │
│   ⏸ Gate #1: User reviews brief                          │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · Parallel paths ───────────────────────────────┐
│  Path A · Agent 6 OpenDesigner                            │
│       brief → LLM picks `hyperframes` skill               │
│       → multi-turn (User × Agent × OpenDesign)            │
│       → motion film .mp4                                   │
│       ├─ Adopt as Hero  → run_dir/hero/intro.mp4          │
│       └─ Adopt as Final → skip Phase 3-5                  │
│                                                           │
│  Path B · Agent 2 SetupRunner                             │
│       detect project → install → spin services → record   │
│       → recordings/test.mp4                                │
│   ⏸ Gate #2: User reviews visual asset + recorded window  │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────┐
│   M3a · cutting_plan.json                                 │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4 │
│   ⏸ Gate #3: User reviews edit plan + each vN              │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────┐
│   M4a · numpy beat scaffold (kicks aligned to scene cuts)  │
│   M4b · MusicGen-small (CUDA fp16, ~40s inference)          │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                      │
│   ⏸ Gate #4: User reviews BGM                              │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver (4 sequential steps) ───────┐
│   Step1 · LLM writes voiceover_script.json (zh-CN/en-US)  │
│   Step2 · edge-tts synthesizes per-cue mp3                │
│   Step3 · voice_timeline assembles voice_full.wav         │
│   Step4 · BGM ducking + amix → final_zh-CN/en-US.mp4      │
│   ⏸ Gate #5: User reviews script                           │
└──────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🚀 Quick Start

### 1. Environment
```bash
# Python venv + deps
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# or
.venv/bin/pip install -r requirements.txt        # *nix

# External requirements:
# - ffmpeg on PATH
# - Node.js 24 + pnpm 10 (Phase 3 Remotion + Phase 2A OpenDesign)
# - CUDA GPU (optional, accelerates Phase 4b MusicGen)
```

### 2. OpenDesign daemon
```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm --filter @open-design/daemon build

# Start daemon (NODE_TLS_REJECT_UNAUTHORIZED=0 bypasses corp-VPN cert MITM)
NODE_TLS_REJECT_UNAUTHORIZED=0 pnpm tools-dev run web
# → Web: http://127.0.0.1:<port>/  Daemon: http://127.0.0.1:<port>/
```

### 3. Configure OpenCode for your LLM provider
`~/.config/opencode/opencode.json`:
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

### 4. .env
```ini
ANTHROPIC_BASE_URL=<your provider compatible endpoint>
ANTHROPIC_API_KEY=<key>
# Or set ARK_KEY_1
```

### 5. Run
```bash
# CLI: kick off Phase 1
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

# Web UI
.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

---

## 📊 Observability

Each run persists everything under `workspace/<project>/runs/<run_id>/`:

```
events.jsonl                        all lifecycle events
logs/
  pipeline.jsonl                    full per-line agent logs
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 session state
opendesign_artifacts/               OpenDesign archive backup
```

**Langfuse UI**: http://localhost:3000/ — every LLM call + traced_step span + verify event, persisted to docker volumes (postgres + clickhouse + minio + redis), survives reboots.

Each agent entrypoint is decorated with `@traced_agent("Agent N · sub-step", phase=N)`, which auto-emits `agent_start`/`agent_done` events and opens an OTEL span — full causal chain across all 5 phases.

---

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| Agent runtime | Python 3.13 + Anthropic SDK + Volcengine Ark Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| Observability | Langfuse (self-hosted via docker) + loguru + OpenTelemetry |
| Visual composition | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| Video editing | Remotion (React + TSX → mp4) |
| BGM | numpy beat scaffold + facebook/musicgen-small (PyTorch CUDA) |
| Voiceover | edge-tts (Microsoft Azure Neural TTS) + ffmpeg ducking |

---

## 📁 Repo layout

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2
│   ├── remotion_composer.py   Agent 3
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/                     per-agent helpers (bgm_*, voice_*, opendesign_*, recorder, etc.)
├── observability/             Langfuse OTLP tracing + loguru logging + EventBus
├── verify/                    ffprobe-based artifact verification
├── web/                       FastAPI + Jinja templates
├── pipeline.py                Pipeline class + state machine
└── cli.py                     viedo CLI
```

---

## 🤝 Acknowledgements

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames integration
- [remotion](https://github.com/remotion-dev/remotion) · React-based video composition
- [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) · Text-conditioned music generation
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS wrapper
- [Langfuse](https://github.com/langfuse/langfuse) · LLM observability (self-hosted)

---

## 📝 License

Apache-2.0

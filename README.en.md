# 🎬 Promo Video Pipeline

> Turn any GitHub repo into a 30-second AI promo video. **An autonomous agent reads the source code, drives the actual project as a real user would, and decides how to cut the result.**
>
> 5 phases, 6 agents, full-stack tracing into self-hosted Langfuse.
> Visual assets via [Open Design](https://github.com/nexu-io/open-design)'s `hyperframes` and `static_hero` skills;
> demo recording via playwright (web projects) / pyte+PIL (CLI projects);
> editing via [Remotion](https://www.remotion.dev/), BGM via MusicGen, voiceover via edge-tts.

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

- **Demo Driver = autonomous demonstrator agent**: reads project source → decides what to demo → operates the running app via playwright/pty → judges for itself when the demo is meaningfully complete. **No duration cap** — if a project takes a million years to demonstrate end-to-end, record for a million years.
- **Three-track mixing**: the editor agent gets (a) the Demo Driver's real screen recording + (b) OpenDesigner hyperframe motion clips + (c) static_hero designed HTML pages, and decides the mix on its own — no preset ratios.
- **User-in-the-loop, end-to-end**: 5 explicit review gates plus, while the Demo Driver is running, the user can type "skip the login flow / focus on feature X / wrap it up" at any time and the agent picks up the message on its next turn.
- **Full observability**: events.jsonl + per-agent JSONL + self-hosted Langfuse UI; every LLM call and every internal step (`MusicGen.load_model`, `http.GET /api/...`, `pty.send`, `browser.click`) nests under its parent agent span.
- **6 LLM agents** built on the Anthropic SDK in tool-use loops, with user feedback gates — **no hand-rolled state machines for agent decisions**.

---

## 🏗 Architecture

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ──────────────────────────────┐
│   git clone → list_dir / read_file source → project_brief.md     │
│   ⏸ Gate #1: user reviews brief (positioning / audience / tone   │
│              — NOT a checklist of features to demo)              │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · parallel paths ───────────────────────────────────────┐
│  Path A · Agent 6 OpenDesigner (visual assets)                    │
│      brief → pick skill (hyperframes / static_hero)               │
│      → multi-turn (user × agent × OpenCode CLI × Open Design)     │
│      ├─ hyperframes  → motion film .mp4   (→ hyperframes/)        │
│      └─ static_hero  → designed HTML page (→ html_asset/)         │
│                                                                  │
│  Path B1 · Agent 2 SetupRunner (boot the project)                 │
│      reads source + README → setup_plan.json                      │
│      → install / seed / start services (user approves plan)       │
│                                                                  │
│  Path B2 · Demo Driver Agent (demonstrate the project)            │
│      reads source itself to decide what's worth showing           │
│      ├─ web project: playwright BrowserSession                    │
│      │   (goto / click / fill / wait / screenshot…)               │
│      └─ CLI project: PtySession (pty_send / wait_for / screen)    │
│      → records the entire session (no duration cap)               │
│      → recordings/demo.mp4 + demo_captions.jsonl                  │
│   ⏸ Gate #2: user can chat with the driver mid-flight, then       │
│              reviews the recording when it finishes               │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────────────┐
│   M3a · cutting_plan.json — agent autonomously mixes:             │
│         • recording (Demo Driver capture)                         │
│         • hyperframe (OpenDesigner motion_film .mp4)              │
│         • html (OpenDesigner static_hero page + scroll/zoom)      │
│         + caption track from demo_captions.jsonl                  │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4        │
│   ⏸ Gate #3: user reviews mix + each vN                           │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────────────┐
│   M4a · numpy beat scaffold (kicks aligned to scene cuts)         │
│   M4b · MusicGen-small/melody (CUDA fp16, ~40s inference)         │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                             │
│   ⏸ Gate #4: user reviews BGM                                     │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver (4 sequential steps) ───────────────┐
│   Step1 · LLM writes voiceover_script.json (zh-CN/en-US)          │
│   Step2 · edge-tts synthesises per-cue mp3                        │
│   Step3 · voice_timeline assembles voice_full.wav                 │
│   Step4 · BGM ducking + amix → final_zh-CN/en-US.mp4              │
│   ⏸ Gate #5: user reviews script                                  │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🤖 Demo Driver — the heart of the project

This is **not** a fixed-duration screen recorder. The Demo Driver is an autonomous agent with four tool families:

### 1. Source-reading tools (decide *what* to demo)
`list_dir` / `read_file` / `find_files` / `grep` — the driver reads route handlers, main loops, CLI commands, feature modules. **What to demo is decided by the code, not by the brief's "selling points" list** — marketing copy describes positioning, the source code describes reality.

### 2. Project-operation tools (decide *how* to demo)

**Web mode** (whenever `setup_plan.json` declares a `health_url`):
```
browser_goto / browser_click / browser_fill / browser_press / browser_scroll
browser_hover / browser_wait_for
browser_screenshot     → returns PNG inline so the agent sees it (vision)
browser_visible_text   → document.body.innerText
browser_a11y_snapshot  → CDP Accessibility.getFullAXTree
browser_interactables  → catalog of clickables/inputs with stable selectors
```
Backed by `tools/browser_session.py`'s `BrowserSession`: playwright chromium with `record_video_dir` so the entire session is captured natively, transcoded webm → mp4 on stop.

**CLI mode** (no services, or service.command is a CLI program):
```
pty_send (write to stdin)        pty_wait_for (regex on screen)
pty_screen (the full pyte grid)  pty_read_recent (tail n lines)
pty_is_alive
```
Backed by `tools/pty_session.py`'s `PtySession`: subprocess + pyte terminal emulator + a background sampler thread that renders one PNG per 1/fps tick, then ffmpeg-stitches into mp4 at stop time. **No 30-second deadline** — the driver decides when to call stop.

### 3. Demo-control tools
- `mark_caption(zh, en, importance)` — tags the *current* recording timestamp with a bilingual caption, written to `demo_captions.jsonl`. Phase 3 picks these up as the caption track instead of having an LLM invent text.
- `ask_user(question)` — blocks until the user replies (the user types in the web UI → `live_feedback.jsonl` is appended → driver reads it).
- `finish_demo(summary, completeness)` — driver decides the demo is done, ends the loop, stops the session, finalises the mp4.
- `log_thought(text)` — write to the agent log without affecting captions or video.

### 4. User-in-the-loop (a control channel, not a tool)
The user types into a textarea in the web UI at any time, e.g.:
```
skip the auth, go straight to the main feature
that step was too fast — wait_for next time
ok that's enough, wrap it up
```
Each entry is appended to `live_feedback.jsonl`. **Before every LLM round**, the driver reads new entries and splices them into the conversation:
```
[USER LIVE FEEDBACK]: skip the auth, go straight to the main feature
```
The LLM sees this on its next turn and adjusts.

---

## 🎨 Three-track mixing (Phase 3)

`cutting_plan.json` allows five `background.type` values: `color` · `gradient` · `recording` · `hyperframe` · `html`.

| Type | Source | Use for |
|---|---|---|
| `recording` | Demo Driver capture `recordings/test.mp4` | Authenticity — actual project running |
| `hyperframe` | OpenDesigner motion_film outputs `hyperframes/*.mp4` | Polish — designer animations |
| `html` | OpenDesigner static_hero output `html_asset/index.html` | Hero / intro / outro — live `<iframe>` with scroll + zoom |
| `color` / `gradient` | Solid / gradient colour | Legibility for text-dense scenes |

**No preset ratio.** The RemotionComposer agent receives an `available_assets` map and chooses the mix itself. The system prompt only specifies hard rules:
- **R3** (head/tail skip): recording must skip first 5s and last 5s (unstable frames). Hyperframe / html have no such rule (OpenDesign output is clean).
- **R4** (legibility): title-style (large + short text) must use recording/hyperframe with darken 0.65–0.85, or html. Body-style (small or long text) must use color/gradient/html, or recording/hyperframe with darken ≥ 0.7.
- **R5**: consecutive scenes default to a 15-frame crossfade.
- **R6**: every `source_path` must exist in `available_assets`.

`tools/remotion_codegen.py` translates the plan into TSX:
- `recording` / `hyperframe` → `<Video>` / `<OffthreadVideo src={staticFile(...)} startFrom={...}>`
- `html` → custom `<HtmlBg>` component: a real `<iframe>` + `onLoad` triggers `contentWindow.scrollTo(0, scrollMax * pct)` + CSS `transform: scale(zoom)`

---

## 🚀 Quick Start

### 1. Environment
```bash
# Python venv + deps
python -m venv .venv
.venv/Scripts/pip install -e .                  # Windows
# or
.venv/bin/pip install -e .                       # *nix

# Playwright Chromium (required for Demo Driver web mode)
.venv/Scripts/python -m playwright install chromium

# External requirements:
# - ffmpeg on PATH
# - Node.js 24 + pnpm 10 (Phase 3 Remotion + Phase 2A OpenDesign)
# - Docker Desktop (for self-hosted Langfuse)
# - CUDA GPU (optional, accelerates Phase 4b MusicGen)
```

### 2. Bring up self-hosted Langfuse
```bash
mkdir langfuse-stack && cd langfuse-stack
curl -O https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml

cat > .env << 'EOF'
SALT=langfuse-local-salt-XXXX
ENCRYPTION_KEY=<openssl rand -hex 32>
NEXTAUTH_SECRET=<openssl rand -hex 32>
LANGFUSE_INIT_ORG_ID=local-org
LANGFUSE_INIT_ORG_NAME=Local
LANGFUSE_INIT_PROJECT_ID=video-pipeline
LANGFUSE_INIT_PROJECT_NAME=video-pipeline
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-local-XXXX
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-local-XXXX
LANGFUSE_INIT_USER_EMAIL=local@example.com
LANGFUSE_INIT_USER_NAME=Local
LANGFUSE_INIT_USER_PASSWORD=langfuse-local-pw
EOF

# Brings up postgres + clickhouse + redis + minio + langfuse-web + langfuse-worker
docker compose --env-file .env up -d
# Wait ~30s, then visit http://localhost:3000
# (login: local@example.com / langfuse-local-pw)
```

> **China network notes**: if `cgr.dev/chainguard/minio` won't pull, swap that line in
> `docker-compose.yml` to `docker.io/minio/minio`. If port 6379 conflicts with a
> local Redis, change the redis port mapping to `127.0.0.1:6380:6379`.

### 3. Project `.env`
```ini
# LLM provider (Volcengine Ark Coding Plan is the default;
# any Anthropic-API-compatible endpoint also works)
ANTHROPIC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding
ANTHROPIC_API_KEY=<your key>
ARK_BASE_URL_OPENAI=https://ark.cn-beijing.volces.com/api/coding/v3
ARK_KEY_1=<your key>
LLM_REASONING=claude-sonnet-4-20250514
LLM_FAST=deepseek-v3.2

# Langfuse target (defaults match the docker-compose stack above)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-XXXX     # match langfuse-stack/.env
LANGFUSE_SECRET_KEY=sk-lf-local-XXXX
```

### 4. OpenDesign daemon
```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm --filter @open-design/daemon build

# Start daemon (NODE_TLS_REJECT_UNAUTHORIZED=0 bypasses corporate VPN cert MITM)
NODE_TLS_REJECT_UNAUTHORIZED=0 pnpm tools-dev run web
# → Web: http://127.0.0.1:<port>/  Daemon: http://127.0.0.1:<port>/
```

OpenDesign drives the OpenCode CLI; configure your provider in
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

### 5. Run
```bash
# CLI: kick off Phase 1
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

# Web UI
.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

The web UI hosts every gate, the live Demo Driver panel (progress / captions / two-way chat), and the Phase 3-5 run/accept routes.

---

## 📊 Observability

Each run persists everything under `workspace/<project>/runs/<run_id>/`:

```
project_brief.md                    Phase 1 output
setup_plan.json                     Phase 2B SetupRunner output
recordings/
  demo.mp4                          Demo Driver recording
  test.mp4                          accepted Phase 2 recording (= demo.mp4)
demo_captions.jsonl                 bilingual caption track
demo_summary.md                     driver's wrap-up summary
demo_driver_progress.json           live progress (web UI polls this)
live_feedback.jsonl                 user → driver chat channel
hyperframes/*.mp4                   OpenDesigner motion_film output
html_asset/                         OpenDesigner static_hero output
cutting_plan.json                   Phase 3a Composer output
remotion/                           Phase 3b generated Remotion project
outputs/v1.mp4                      Phase 3 render
outputs/v1_bgm_final.mp4            Phase 4 with BGM
outputs/final_zh-CN.mp4             Phase 5 final with voiceover
events.jsonl                        all lifecycle events
logs/
  pipeline.jsonl                    full per-line agent logs
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B plan / exec
  demo_driver.jsonl                 Phase 2C
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 session state
```

**Langfuse UI**: http://localhost:3000/ — every LLM call + traced_step span + verify event, persisted to docker volumes (postgres + clickhouse + minio + redis), survives reboots.

Each agent entry is decorated with `@traced_agent("Agent N · sub-step", phase=N)`, which auto-emits `agent_start`/`agent_done` events and opens an OTEL parent span. Inner steps use the `traced_step("MusicGen.load_model", ...)` context manager so you can see (e.g.) Agent 4 BGM → MusicGen.load_model → MusicGen.tokenize_inputs → MusicGen.generate → MusicGen.write_wav as one nested tree. Anthropic SDK calls are auto-instrumented via `openinference-instrumentation-anthropic` — prompts, responses, tool_use, tool_result, token usage all captured.

---

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| Agent runtime | Python 3.13 + Anthropic SDK + Volcengine Ark Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| Observability | Langfuse (self-hosted via docker compose) + loguru + OTEL HTTP exporter + openinference-instrumentation-anthropic |
| Demo Driver / web | playwright (chromium headless) + record_video_dir + CDP a11y |
| Demo Driver / CLI | subprocess + pyte (terminal emulator) + PIL + ffmpeg image-seq |
| Visual assets | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| Video editing | Remotion (React + TSX → mp4), with OffthreadVideo + custom HtmlBg iframe component |
| BGM | numpy beat scaffold + facebook/musicgen-{small,melody} (PyTorch CUDA fp16) |
| Voiceover | edge-tts (Microsoft Azure Neural TTS) + ffmpeg sidechaincompress ducking |

---

## 📁 Repo layout

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2 SetupRunner (plan-only, host executes)
│   ├── demo_driver.py         Demo Driver Agent (Phase 2C, autonomous) ★
│   ├── remotion_composer.py   Agent 3 cutting_plan + three-track mixing
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/
│   ├── pty_session.py         PtySession (CLI demo + recording) ★
│   ├── browser_session.py     BrowserSession (web demo + recording) ★
│   ├── bgm_scaffold.py        Agent 4 M4a numpy beats
│   ├── bgm_musicgen.py        Agent 4 M4b MusicGen
│   ├── bgm_mux.py             Agent 4 M4c ffmpeg
│   ├── tts_edge.py            Agent 5 Step2
│   ├── voice_timeline.py      Agent 5 Step3
│   ├── bgm_duck_mux.py        Agent 5 Step4 ducking + amix
│   ├── opendesign_client.py   Agent 6 daemon HTTP client
│   ├── opendesign_lifecycle.py daemon start/stop
│   ├── plan_executor.py       Agent 2 plan execution
│   ├── service_manager.py     long-lived service process tracker
│   ├── recorder.py            ffmpeg gdigrab fallback recorder
│   ├── remotion_codegen.py    Agent 3 cutting_plan → TSX (incl. HtmlBg)
│   └── remotion_render.py     npx remotion render
├── observability/
│   ├── tracer.py              Langfuse OTLP HTTP exporter setup
│   ├── logger.py              loguru + per-agent JSONL
│   ├── audit.py               @traced_agent + traced_step + run_context
│   └── events.py              EventBus (events.jsonl)
├── verify/                    ffprobe-based artifact verification
├── web/
│   ├── main.py                FastAPI routes + templates
│   └── templates/             Jinja2 + HTMX partials
├── pipeline.py                Pipeline class + state machine
└── cli.py                     viedo CLI

★ = key new components for the Demo Driver path
```

---

## 📚 Further reading

- [WORKFLOW.md](WORKFLOW.md) — 5-phase state machine, gate definitions, observability triplet spec
- `src/agents/demo_driver.py` top docstring — driver design philosophy (agent loop + user-in-loop, anti-state-machine)
- Langfuse self-hosting docs: https://langfuse.com/self-hosting

---

## 🤝 Acknowledgements

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames
- [remotion](https://github.com/remotion-dev/remotion) · React-based video composition
- [microsoft/playwright](https://github.com/microsoft/playwright-python) · Browser automation + video recording
- [selectel/pyte](https://github.com/selectel/pyte) · Pythonic terminal emulator
- [facebook/musicgen](https://huggingface.co/facebook/musicgen-melody) · Text-conditioned music generation
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS wrapper
- [Langfuse](https://github.com/langfuse/langfuse) · LLM observability (self-hosted)

---

## 📝 License

Apache-2.0

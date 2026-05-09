# 🎬 Promo Video Pipeline

> 모든 GitHub 프로젝트 → 30 초 AI 프로모 비디오. **자율 에이전트가 소스 코드를 읽고, 실제 사용자처럼 프로젝트를 조작하여 데모하고, 편집 방침을 스스로 결정한다.**
>
> 5 페이즈, 6 에이전트, 풀스택 트레이싱을 self-hosted Langfuse로.
> 시각 자산은 [Open Design](https://github.com/nexu-io/open-design)의 `hyperframes` / `static_hero` skill;
> 데모 녹화는 playwright(web) / pyte+PIL(CLI);
> 편집은 [Remotion](https://www.remotion.dev/), BGM은 MusicGen, 보이스오버는 edge-tts.

<p align="center">
  <a href="README.md">简体中文</a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md"><b>한국어</b></a>
</p>

<p align="center">
  <img src="docs/workflow.svg" alt="Workflow diagram" width="100%"/>
</p>

---

## ✨ 특징

- **Demo Driver = 자율 데모 에이전트**: 프로젝트 소스 읽고 → 무엇을 데모할지 판단 → playwright/pty로 실제 조작 → 「의미 있는 데모가 완료되었다」고 스스로 판단할 때까지 진행. **시간 제한 없음**. 1000만 년 걸려야 끝나면 1000만 년 녹화한다.
- **3 트랙 믹싱**: 에디터 에이전트는 (a) Demo Driver의 실제 화면 녹화 + (b) OpenDesigner hyperframe 모션 + (c) static_hero 디자인 HTML 페이지 3 종을 받아 믹스를 스스로 결정. 사전 비율 없음.
- **User-in-the-loop (전 과정)**: 5개 리뷰 게이트 + Demo Driver 실행 중에도 사용자가 「로그인 건너뛰고 / 기능 X에 집중 / 됐어 마무리」라고 텍스트 입력 가능. 다음 턴에서 에이전트가 인지.
- **풀스택 옵저버빌리티**: events.jsonl + 에이전트별 JSONL + self-hosted Langfuse UI. 모든 LLM 호출 + 내부 step (`MusicGen.load_model`, `http.GET /api/...`, `pty.send`, `browser.click`)이 부모 span 아래 중첩.
- **6 LLM 에이전트** Anthropic SDK + tool-use loop + 사용자 피드백 게이트. **수기 상태 머신 사용 안 함**.

---

## 🏗 아키텍처

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ──────────────────────────────┐
│   git clone → list_dir / read_file 소스 → project_brief.md      │
│   ⏸ Gate #1: 사용자가 brief 검토 (포지셔닝/오디언스/톤만 — 데모  │
│              기능 체크리스트가 아님)                              │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · 병렬 경로 ────────────────────────────────────────────┐
│  Path A · Agent 6 OpenDesigner (시각 자산)                        │
│      brief → skill 선택 (hyperframes / static_hero)               │
│      → 멀티턴 (user × agent × OpenCode CLI × OpenDesign)          │
│      ├─ hyperframes → motion film .mp4   (→ hyperframes/)         │
│      └─ static_hero → 디자인 HTML page (→ html_asset/)             │
│                                                                  │
│  Path B1 · Agent 2 SetupRunner (프로젝트 기동)                     │
│      소스 + README → setup_plan.json                              │
│      → install / seed / start services (사용자가 plan 승인)        │
│                                                                  │
│  Path B2 · Demo Driver Agent (프로젝트 데모)                       │
│      소스 읽고 무엇을 데모할지 스스로 판단                          │
│      ├─ web: playwright BrowserSession                            │
│      │   (goto / click / fill / wait / screenshot…)               │
│      └─ CLI: PtySession (pty_send / wait_for / screen)            │
│      → 세션 전체 녹화 (시간 제한 없음)                              │
│      → recordings/demo.mp4 + demo_captions.jsonl                  │
│   ⏸ Gate #2: 실행 중 사용자 개입 가능, 종료 후 녹화 검토            │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────────────┐
│   M3a · cutting_plan.json — 에이전트가 자율적으로 믹스:            │
│         • recording (Demo Driver 실제 녹화)                       │
│         • hyperframe (OpenDesigner motion_film .mp4)              │
│         • html (OpenDesigner static_hero 페이지 + scroll/zoom)     │
│         + 캡션 트랙은 demo_captions.jsonl에서 가져옴                │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4        │
│   ⏸ Gate #3: 사용자가 편집 방향 + 각 vN 검토                       │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────────────┐
│   M4a · numpy 비트 스캐폴드 (kicks를 씬 컷에 정렬)                 │
│   M4b · MusicGen-small/melody (CUDA fp16, 추론 ~40초)              │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                             │
│   ⏸ Gate #4: 사용자가 BGM 검토                                     │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver (4 단계 직렬) ──────────────────────┐
│   Step1 · LLM이 voiceover_script.json 작성 (zh-CN/en-US)          │
│   Step2 · edge-tts로 큐별 mp3 합성                                │
│   Step3 · voice_timeline로 voice_full.wav 조립                    │
│   Step4 · BGM ducking + amix → final_zh-CN/en-US.mp4              │
│   ⏸ Gate #5: 사용자가 스크립트 검토                                │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🤖 Demo Driver — 프로젝트의 핵심

고정 시간 화면 녹화기가 아니다. Demo Driver는 4개 도구군을 가진 자율 에이전트:

### 1. 소스 읽기 도구 (*무엇을* 데모할지 결정)
`list_dir` / `read_file` / `find_files` / `grep` — 라우트 핸들러, 메인 루프, CLI 명령, 기능 모듈을 읽는다. **무엇을 데모할지는 코드가 결정한다. brief의 「셀링 포인트」 리스트가 아니다** — 마케팅 카피는 포지셔닝 이야기, 소스 코드는 현실 이야기.

### 2. 프로젝트 조작 도구 (*어떻게* 데모할지 결정)

**Web 모드** (`setup_plan.json`에 `health_url`이 있을 때):
```
browser_goto / browser_click / browser_fill / browser_press / browser_scroll
browser_hover / browser_wait_for
browser_screenshot     → PNG을 인라인 반환 (vision)
browser_visible_text   → document.body.innerText
browser_a11y_snapshot  → CDP Accessibility.getFullAXTree
browser_interactables  → 안정 셀렉터 포함 클릭/입력 가능 요소 목록
```
백엔드는 `tools/browser_session.py`의 `BrowserSession`: playwright chromium + `record_video_dir`로 세션 전체를 네이티브 녹화, stop 시 webm → mp4 트랜스코드.

**CLI 모드** (services 없거나 service.command가 CLI 프로그램):
```
pty_send (stdin 쓰기)            pty_wait_for (화면에서 정규식)
pty_screen (pyte 그리드 전체)     pty_read_recent (마지막 n 줄)
pty_is_alive
```
백엔드는 `tools/pty_session.py`의 `PtySession`: subprocess + pyte 터미널 에뮬레이터 + 백그라운드 샘플링 스레드가 1/fps마다 PNG 렌더, stop 시 ffmpeg로 mp4 결합. **30초 데드라인 없음** — driver가 stop 호출할 때까지 계속.

### 3. 데모 제어 도구
- `mark_caption(zh, en, importance)` — *현재* 녹화 타임스탬프에 이중언어 자막 태그. `demo_captions.jsonl`에 작성. Phase 3가 캡션 트랙으로 사용. LLM이 문구를 발명하지 않는다.
- `ask_user(question)` — 사용자 응답까지 블록 (web UI에서 사용자 입력 → `live_feedback.jsonl` 추가 → driver가 읽음).
- `finish_demo(summary, completeness)` — driver 자신이 데모 완료 판단, 루프 종료, 세션 정지, mp4 확정.
- `log_thought(text)` — 에이전트 로그만 작성 (비디오에 영향 없음).

### 4. User-in-the-loop (도구 아닌 제어 채널)
사용자는 web UI 텍스트 영역에 언제든 입력 가능, 예:
```
인증 건너뛰고 메인 기능 바로 보여줘
이 단계 너무 빨라 — 다음엔 wait_for 사용해
이제 충분, 마무리
```
각 항목은 `live_feedback.jsonl`에 추가. **LLM 호출마다 직전**에 driver가 새 항목을 읽고 conversation에 splice:
```
[USER LIVE FEEDBACK]: 인증 건너뛰고 메인 기능 바로 보여줘
```
LLM은 다음 턴에서 보고 조정.

---

## 🎨 3 트랙 믹싱 (Phase 3)

`cutting_plan.json`의 `background.type` 5 값: `color` · `gradient` · `recording` · `hyperframe` · `html`.

| Type | 출처 | 용도 |
|---|---|---|
| `recording` | Demo Driver 녹화 `recordings/test.mp4` | 진정성 — 실제 프로젝트 동작 |
| `hyperframe` | OpenDesigner motion_film 출력 `hyperframes/*.mp4` | 세련됨 — 디자이너 애니메이션 |
| `html` | OpenDesigner static_hero 출력 `html_asset/index.html` | hero / intro / outro — 실제 `<iframe>` + scroll + zoom |
| `color` / `gradient` | 단색 / 그라데이션 | 텍스트 밀집 씬의 가독성 |

**사전 비율 없음.** RemotionComposer 에이전트는 `available_assets`를 받아 믹스를 스스로 결정. 시스템 프롬프트는 하드 룰만:
- **R3** 머리/꼬리 스킵: recording은 처음/끝 5초 스킵 필수 (불안정 프레임). hyperframe / html은 불필요 (OpenDesign 출력 깨끗).
- **R4** 가독성: 크고 짧은 타이틀 텍스트 → recording/hyperframe + darken 0.65–0.85 또는 html. 작거나 긴 본문 → color/gradient/html, 또는 recording/hyperframe + darken ≥ 0.7.
- **R5**: 연속 씬은 기본 15 프레임 크로스페이드.
- **R6**: 모든 `source_path`는 `available_assets`에 존재 필수.

`tools/remotion_codegen.py`가 plan을 TSX로 번역:
- `recording` / `hyperframe` → `<Video>` / `<OffthreadVideo src={staticFile(...)} startFrom={...}>`
- `html` → 커스텀 `<HtmlBg>` 컴포넌트: 실제 `<iframe>` + `onLoad`로 `contentWindow.scrollTo(0, scrollMax * pct)` + CSS `transform: scale(zoom)`

---

## 🚀 Quick Start

### 1. 환경
```bash
python -m venv .venv
.venv/Scripts/pip install -e .                  # Windows
# 또는
.venv/bin/pip install -e .                       # *nix

# Playwright Chromium (Demo Driver web 모드 필수)
.venv/Scripts/python -m playwright install chromium

# 외부 요구사항:
# - ffmpeg를 PATH에
# - Node.js 24 + pnpm 10 (Phase 3 Remotion + Phase 2A OpenDesign)
# - Docker Desktop (self-hosted Langfuse용)
# - CUDA GPU (선택, Phase 4b MusicGen 가속)
```

### 2. self-hosted Langfuse 기동
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

# 6 서비스 기동: postgres + clickhouse + redis + minio + langfuse-web + langfuse-worker
docker compose --env-file .env up -d
# 약 30초 대기 후 http://localhost:3000으로
```

> **중국 본토 네트워크**: `cgr.dev/chainguard/minio` pull 실패 시
> `docker.io/minio/minio`로 교체. Redis 6379가 로컬과 충돌하면
> `127.0.0.1:6380:6379`로 변경.

### 3. 프로젝트 `.env`
```ini
ANTHROPIC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding
ANTHROPIC_API_KEY=<your key>
ARK_BASE_URL_OPENAI=https://ark.cn-beijing.volces.com/api/coding/v3
ARK_KEY_1=<your key>
LLM_REASONING=claude-sonnet-4-20250514
LLM_FAST=deepseek-v3.2

LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-XXXX     # langfuse-stack/.env와 일치
LANGFUSE_SECRET_KEY=sk-lf-local-XXXX
```

### 4. OpenDesign daemon
```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm --filter @open-design/daemon build

NODE_TLS_REJECT_UNAUTHORIZED=0 pnpm tools-dev run web
# → Web: http://127.0.0.1:<port>/  Daemon: http://127.0.0.1:<port>/
```

OpenCode를 `~/.config/opencode/opencode.json`로 설정:
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

### 5. 파이프라인 실행
```bash
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

---

## 📊 옵저버빌리티

각 run은 `workspace/<project>/runs/<run_id>/`에 영속:

```
project_brief.md                    Phase 1 출력
setup_plan.json                     Phase 2B SetupRunner 출력
recordings/
  demo.mp4                          Demo Driver 녹화
  test.mp4                          채택된 Phase 2 녹화 (= demo.mp4)
demo_captions.jsonl                 이중언어 자막 트랙
demo_summary.md                     driver 마무리 요약
demo_driver_progress.json           실시간 진행 상황 (web UI 폴링)
live_feedback.jsonl                 사용자 → driver 채팅 채널
hyperframes/*.mp4                   OpenDesigner motion_film 출력
html_asset/                         OpenDesigner static_hero 출력
cutting_plan.json                   Phase 3a Composer 출력
remotion/                           Phase 3b 생성 Remotion 프로젝트
outputs/v1.mp4                      Phase 3 렌더
outputs/v1_bgm_final.mp4            Phase 4 BGM 추가 후
outputs/final_zh-CN.mp4             Phase 5 보이스오버 추가 후
events.jsonl                        모든 라이프사이클 이벤트
logs/
  pipeline.jsonl                    전 에이전트 라인 로그
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B plan / exec
  demo_driver.jsonl                 Phase 2C
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 세션 상태
```

**Langfuse UI**: http://localhost:3000/ — 모든 LLM 호출 + traced_step span + verify 이벤트를 docker volumes (postgres + clickhouse + minio + redis)에 저장 (재부팅 후에도 유지).

각 에이전트 진입점에 `@traced_agent("Agent N · 서브 스텝", phase=N)` 데코레이터 부착, `agent_start`/`agent_done` 이벤트 자동 emit + OTEL 부모 span 생성. 내부 step은 `traced_step("MusicGen.load_model", ...)` 컨텍스트 매니저로 중첩 — Langfuse에서 Agent 4 BGM → MusicGen.load_model → MusicGen.tokenize_inputs → MusicGen.generate → MusicGen.write_wav가 한 트리로 보임. Anthropic SDK 호출은 `openinference-instrumentation-anthropic`로 자동 계측, prompt / response / tool_use / tool_result / token usage 모두 캡처.

---

## 🛠 기술 스택

| 레이어 | 기술 |
|---|---|
| Agent runtime | Python 3.13 + Anthropic SDK + Volcengine Ark Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| 옵저버빌리티 | Langfuse (docker self-hosted) + loguru + OpenTelemetry HTTP exporter + openinference-instrumentation-anthropic |
| Demo Driver / web | playwright (chromium headless) + record_video_dir + CDP a11y |
| Demo Driver / CLI | subprocess + pyte (터미널 에뮬레이터) + PIL + ffmpeg image-seq |
| 시각 자산 | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| 비디오 편집 | Remotion (React + TSX → mp4), OffthreadVideo + 커스텀 HtmlBg iframe 컴포넌트 |
| BGM | numpy 비트 + facebook/musicgen-{small,melody} (PyTorch CUDA fp16) |
| 보이스오버 | edge-tts (Microsoft Azure Neural TTS) + ffmpeg sidechaincompress ducking |

---

## 📁 리포지토리 구조

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2 SetupRunner (plan만, host가 실행)
│   ├── demo_driver.py         Demo Driver Agent (Phase 2C 자율 데모) ★
│   ├── remotion_composer.py   Agent 3 cutting_plan + 3 트랙 믹스
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/
│   ├── pty_session.py         PtySession (CLI 데모 + 녹화) ★
│   ├── browser_session.py     BrowserSession (web 데모 + 녹화) ★
│   ├── bgm_*.py               Agent 4 BGM 각 단계
│   ├── tts_edge.py / voice_timeline.py / bgm_duck_mux.py  Agent 5
│   ├── opendesign_*.py        Agent 6 daemon 클라이언트 / 라이프사이클
│   ├── remotion_codegen.py    Agent 3 cutting_plan → TSX (HtmlBg 포함)
│   └── remotion_render.py     npx remotion render
├── observability/             Langfuse OTLP tracing + loguru + EventBus
├── verify/                    ffprobe 아티팩트 검증
├── web/                       FastAPI + Jinja2 템플릿
├── pipeline.py                Pipeline 클래스 + 상태 머신
└── cli.py                     viedo CLI

★ = Demo Driver 경로의 주요 신규 컴포넌트
```

---

## 📚 추가 문서

- [WORKFLOW.md](WORKFLOW.md) — 5 페이즈 상태 머신, 게이트 정의, 옵저버빌리티 3종 세트 사양
- `src/agents/demo_driver.py` 상단 docstring — driver 설계 철학 (agent loop + user-in-loop, anti-state-machine)
- Langfuse self-hosting 공식 문서: https://langfuse.com/self-hosting

---

## 🤝 감사의 말

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames
- [remotion](https://github.com/remotion-dev/remotion) · React 기반 비디오 합성
- [microsoft/playwright](https://github.com/microsoft/playwright-python) · 브라우저 자동화 + 비디오 녹화
- [selectel/pyte](https://github.com/selectel/pyte) · Pythonic 터미널 에뮬레이터
- [facebook/musicgen](https://huggingface.co/facebook/musicgen-melody) · 텍스트 기반 음악 생성
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS 래퍼
- [Langfuse](https://github.com/langfuse/langfuse) · LLM 옵저버빌리티 (self-hosted)

---

## 📝 라이선스

Apache-2.0

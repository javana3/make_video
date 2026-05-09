# 🎬 Promo Video Pipeline

> 어떤 GitHub 프로젝트도 30초 AI 프로모 비디오로 변환. 5 페이즈, 6 에이전트, 풀스택 트레이싱.
> [Open Design](https://github.com/nexu-io/open-design)의 `hyperframes` skill로 모션 그래픽, [Remotion](https://www.remotion.dev/)으로 합성, MusicGen으로 BGM, edge-tts로 이중언어 보이스오버.

<p align="center">
  <a href="README.md">简体中文</a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md"><b>한국어</b></a>
</p>

---

## ✨ 주요 기능

- **5 페이즈 파이프라인**: 분석 → 비주얼 자산 + 녹화 (병렬) → Remotion 합성 → BGM → 보이스오버
- **6 개 LLM 에이전트**: 페이즈마다 전용, 모두 Anthropic SDK 기반
- **OpenDesign × HyperFrames**: 자연어 → 모션 그래픽 .mp4 (HTML+GSAP+프레임 캡처)
- **이중언어 보이스오버**: edge-tts (중국어 + 영어) + ducking + amix
- **풀스택 옵저버빌리티**: events.jsonl + 에이전트별 JSONL + Phoenix UI에서 LLM span
- **휴먼 인 더 루프**: 5 개의 명시적 검토 게이트 — 사용자는 논의와 승인만, 코드는 작성하지 않음

---

## 🏗 아키텍처

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ─────────────────────┐
│   git clone → README/소스 → project_brief.md             │
│   ⏸ 게이트 #1: 사용자가 brief 검토                          │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · 병렬 경로 ─────────────────────────────────────┐
│  경로 A · Agent 6 OpenDesigner                            │
│       brief → LLM이 `hyperframes` skill 선택              │
│       → 다중 턴 (User × Agent × OpenDesign)              │
│       → 모션 필름 .mp4                                     │
│       ├─ Hero로 채택 → run_dir/hero/intro.mp4             │
│       └─ Final로 채택 → Phase 3-5 건너뛰기                 │
│                                                           │
│  경로 B · Agent 2 SetupRunner                             │
│       프로젝트 감지 → 의존성 설치 → 서비스 시작 → 녹화         │
│       → recordings/test.mp4                                │
│   ⏸ 게이트 #2: 비주얼 자산 + 녹화 윈도우 검토                  │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────┐
│   M3a · cutting_plan.json                                 │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4 │
│   ⏸ 게이트 #3: 편집 방향 + 각 vN 검토                        │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────┐
│   M4a · numpy 비트 스캐폴드 (kick을 씬 컷에 정렬)            │
│   M4b · MusicGen-small (CUDA fp16, 추론 ~40초)              │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                      │
│   ⏸ 게이트 #4: BGM 검토                                     │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver (4 단계 직렬) ──────────────┐
│   Step1 · LLM이 voiceover_script.json 작성 (zh-CN/en-US) │
│   Step2 · edge-tts로 큐별 mp3 합성                         │
│   Step3 · voice_timeline으로 voice_full.wav 결합            │
│   Step4 · BGM ducking + amix → final_zh-CN/en-US.mp4      │
│   ⏸ 게이트 #5: 스크립트 검토                                  │
└──────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# Python venv + 의존성
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# 또는
.venv/bin/pip install -r requirements.txt        # *nix

# 외부 도구:
# - ffmpeg (PATH)
# - Node.js 24 + pnpm 10 (Phase 3 Remotion + Phase 2A OpenDesign)
# - CUDA GPU (선택, Phase 4b MusicGen 가속)
```

### 2. OpenDesign daemon
```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm --filter @open-design/daemon build

# daemon 시작 (NODE_TLS_REJECT_UNAUTHORIZED=0으로 회사 VPN 인증서 MITM 우회)
NODE_TLS_REJECT_UNAUTHORIZED=0 pnpm tools-dev run web
# → Web: http://127.0.0.1:<port>/  Daemon: http://127.0.0.1:<port>/
```

### 3. OpenCode를 LLM 제공자에 연결
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
# 또는 ARK_KEY_1 설정
```

### 5. 실행
```bash
# CLI: Phase 1 시작
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

# Web UI
.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

---

## 📊 옵저버빌리티

각 run은 `workspace/<project>/runs/<run_id>/` 아래에 모두 영구 저장됩니다:

```
events.jsonl                        모든 라이프사이클 이벤트
logs/
  pipeline.jsonl                    모든 에이전트의 라인별 로그
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 세션 상태
opendesign_artifacts/               OpenDesign 아카이브 백업
```

**Phoenix UI**: http://localhost:6006/ — 모든 LLM 호출 + verify span을 `~/.phoenix/` SQLite에 저장 (재부팅 후에도 유지).

각 에이전트 엔트리는 `@traced_agent("Agent N · 하위 단계", phase=N)`으로 데코레이트되며, 자동으로 `agent_start`/`agent_done` 이벤트 발행 + OTEL span 시작 — 5 페이즈 전체에 걸친 인과 체인 추적 가능.

---

## 🛠 기술 스택

| 레이어 | 기술 |
|---|---|
| 에이전트 런타임 | Python 3.13 + Anthropic SDK + Volcengine Ark Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| 옵저버빌리티 | Phoenix (Arize) + loguru + OpenTelemetry |
| 비주얼 합성 | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| 비디오 편집 | Remotion (React + TSX → mp4) |
| BGM | numpy 비트 스캐폴드 + facebook/musicgen-small (PyTorch CUDA) |
| 보이스오버 | edge-tts (Microsoft Azure Neural TTS) + ffmpeg ducking |

---

## 📁 저장소 구조

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2
│   ├── remotion_composer.py   Agent 3
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/                     에이전트별 헬퍼
├── observability/             Phoenix tracing + loguru logging + EventBus
├── verify/                    ffprobe 기반 산출물 검증
├── web/                       FastAPI + Jinja 템플릿
├── pipeline.py                Pipeline 클래스 + 상태 머신
└── cli.py                     viedo CLI
```

---

## 🤝 감사의 말

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames 통합
- [remotion](https://github.com/remotion-dev/remotion) · React 기반 비디오 합성
- [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) · 텍스트 조건부 음악 생성
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS 래퍼
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) · LLM 옵저버빌리티

---

## 📝 라이선스

Apache-2.0

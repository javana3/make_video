# 🎬 Promo Video Pipeline

> 任意の GitHub リポジトリを 30 秒の AI プロモ動画へ。5 フェーズ、6 エージェント、フルスタック・トレース対応。
> [Open Design](https://github.com/nexu-io/open-design) の `hyperframes` skill によるモーショングラフィックス、[Remotion](https://www.remotion.dev/) による合成、MusicGen による BGM、edge-tts によるバイリンガルナレーション。

<p align="center">
  <a href="README.md">简体中文</a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md"><b>日本語</b></a> ·
  <a href="README.ko.md">한국어</a>
</p>

<p align="center">
  <img src="docs/workflow.svg" alt="Workflow diagram" width="100%"/>
</p>

---

## ✨ 特徴

- **5 フェーズパイプライン**: 解析 → ビジュアル資産 + 録画（並列）→ Remotion 合成 → BGM → ナレーション
- **6 つの LLM エージェント**: 各フェーズ専用、Anthropic SDK ベース
- **OpenDesign × HyperFrames**: 自然言語 → モーショングラフィックス .mp4（HTML+GSAP+フレームキャプチャ）
- **バイリンガル音声**: edge-tts（中国語＋英語）+ ducking + amix
- **完全な可観測性**: events.jsonl + エージェント別 JSONL + Langfuse UI で LLM スパン
- **ヒューマン・イン・ザ・ループ**: 5 つの明示的な承認ゲート — ユーザーは議論と承認のみ、コードは書かない

---

## 🏗 アーキテクチャ

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ─────────────────────┐
│   git clone → README/ソース → project_brief.md            │
│   ⏸ ゲート #1: ユーザーが brief をレビュー                    │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · 並列ルート ────────────────────────────────────┐
│  ルート A · Agent 6 OpenDesigner                          │
│       brief → LLM が `hyperframes` skill を選択           │
│       → 多ターン (User × Agent × OpenDesign)             │
│       → モーション映画 .mp4                                │
│       ├─ Hero として採用 → run_dir/hero/intro.mp4         │
│       └─ Final として採用 → Phase 3-5 をスキップ            │
│                                                           │
│  ルート B · Agent 2 SetupRunner                           │
│       プロジェクト検出 → 依存導入 → サービス起動 → 録画       │
│       → recordings/test.mp4                                │
│   ⏸ ゲート #2: ビジュアル資産 + 録画ウィンドウを承認            │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────┐
│   M3a · cutting_plan.json                                 │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4 │
│   ⏸ ゲート #3: 編集方針 + 各 vN をレビュー                    │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────┐
│   M4a · numpy ビートスキャフォールド（kick をシーン切替に整合）│
│   M4b · MusicGen-small (CUDA fp16, 推論 ~40 秒)             │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                      │
│   ⏸ ゲート #4: BGM をレビュー                                │
└──────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver（4 ステップ直列）─────────────┐
│   Step1 · LLM が voiceover_script.json を作成 (zh-CN/en-US)│
│   Step2 · edge-tts で各キューの mp3 合成                    │
│   Step3 · voice_timeline で voice_full.wav を結合           │
│   Step4 · BGM ducking + amix → final_zh-CN/en-US.mp4      │
│   ⏸ ゲート #5: 台本をレビュー                                 │
└──────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🚀 クイックスタート

### 1. 環境構築
```bash
# Python venv + 依存
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# または
.venv/bin/pip install -r requirements.txt        # *nix

# 外部ツール:
# - ffmpeg (PATH 上)
# - Node.js 24 + pnpm 10 (Phase 3 Remotion + Phase 2A OpenDesign)
# - CUDA GPU (任意、Phase 4b MusicGen を高速化)
```

### 2. OpenDesign daemon
```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
pnpm install
pnpm --filter @open-design/daemon build

# daemon 起動 (NODE_TLS_REJECT_UNAUTHORIZED=0 で社内 VPN の証明書 MITM を回避)
NODE_TLS_REJECT_UNAUTHORIZED=0 pnpm tools-dev run web
# → Web: http://127.0.0.1:<port>/  Daemon: http://127.0.0.1:<port>/
```

### 3. OpenCode を LLM プロバイダーに接続
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
# または ARK_KEY_1 をセット
```

### 5. 実行
```bash
# CLI: Phase 1 を起動
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

# Web UI
.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

---

## 📊 可観測性

各 run は `workspace/<project>/runs/<run_id>/` 以下にすべて永続化されます：

```
events.jsonl                        全ライフサイクルイベント
logs/
  pipeline.jsonl                    全エージェントの行単位ログ
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 セッション状態
opendesign_artifacts/               OpenDesign アーカイブのバックアップ
```

**Langfuse UI**: http://localhost:3000/ — 全 LLM 呼び出し + traced_step スパン + verify イベントを docker volumes（postgres + clickhouse + minio + redis）に保存（再起動しても残る）。

各エージェントエントリは `@traced_agent("Agent N · サブステップ", phase=N)` で装飾され、`agent_start`/`agent_done` イベントを自動発行し OTEL スパンを開始 — 5 フェーズ全体の因果連鎖が完全に追跡可能。

---

## 🛠 技術スタック

| レイヤ | 技術 |
|---|---|
| エージェントランタイム | Python 3.13 + Anthropic SDK + Volcengine Ark Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| 可観測性 | Langfuse（docker self-hosted）+ loguru + OpenTelemetry |
| ビジュアル合成 | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| 動画編集 | Remotion (React + TSX → mp4) |
| BGM | numpy ビートスキャフォールド + facebook/musicgen-small (PyTorch CUDA) |
| ナレーション | edge-tts (Microsoft Azure Neural TTS) + ffmpeg ducking |

---

## 📁 リポジトリ構成

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2
│   ├── remotion_composer.py   Agent 3
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/                     エージェント別ヘルパー
├── observability/             Langfuse OTLP tracing + loguru logging + EventBus
├── verify/                    ffprobe ベースの成果物検証
├── web/                       FastAPI + Jinja テンプレート
├── pipeline.py                Pipeline クラス + ステートマシン
└── cli.py                     viedo CLI
```

---

## 🤝 謝辞

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames 統合
- [remotion](https://github.com/remotion-dev/remotion) · React ベースの動画合成
- [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) · テキスト条件付き音楽生成
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS ラッパー
- [Langfuse](https://github.com/langfuse/langfuse) · LLM 可観測性（self-hosted）

---

## 📝 ライセンス

Apache-2.0

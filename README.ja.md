# 🎬 Promo Video Pipeline

> 任意の GitHub プロジェクト → 30 秒の AI プロモビデオ。**自律エージェントがソースコードを読み、実ユーザーのようにプロジェクトを操作してデモし、編集方針を自分で決める。**
>
> 5 フェーズ、6 エージェント、フルスタック・トレーシングをセルフホスト Langfuse へ。
> 視覚アセットは [Open Design](https://github.com/nexu-io/open-design) の `hyperframes` / `static_hero` skill；
> デモ収録は playwright（web）/ pyte+PIL（CLI）；
> 編集は [Remotion](https://www.remotion.dev/)、BGM は MusicGen、ナレーションは edge-tts。

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

- **Demo Driver = 自律デモンストレーター・エージェント**: ソースコードを読み、何をデモするかを判断 → playwright/pty で実際に操作 → 「意味あるデモが完了した」と判断するまで継続。**時間制限なし**。1000 万年かかるなら 1000 万年録画する。
- **3 トラック・ミキシング**: エディタ・エージェントは (a) Demo Driver の実画面録画 + (b) OpenDesigner の hyperframe モーション + (c) static_hero デザイン HTML ページの 3 種類を受け取り、ミックスを自分で決める。事前比率なし。
- **User-in-the-loop（全工程）**: 5 つのレビュー・ゲートに加え、Demo Driver 実行中もユーザーが「ログインを飛ばして / 機能 X に集中して / 終わってよし」とテキスト入力で介入可能。次のターンでエージェントが受け取る。
- **フルスタック可観測性**: events.jsonl + エージェント別 JSONL + セルフホスト Langfuse UI。各 LLM 呼び出しと内部ステップ（`MusicGen.load_model`、`http.GET /api/...`、`pty.send`、`browser.click`）が親スパン下にネストされる。
- **6 LLM エージェント** Anthropic SDK + ツール使用ループ + ユーザー・フィードバック・ゲート。**手書きステートマシンは使わない**。

---

## 🏗 アーキテクチャ

```
[GitHub URL]
     │
     ▼
┌─ Phase 1 · Agent 1 ProjectAnalyzer ──────────────────────────────┐
│   git clone → list_dir / read_file ソース → project_brief.md     │
│   ⏸ Gate #1: ユーザーが brief をレビュー（ポジショニング/オーディエンス│
│              /トーンのみ — デモ機能のチェックリストではない）       │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 2 · 並列パス ──────────────────────────────────────────────┐
│  Path A · Agent 6 OpenDesigner（視覚アセット）                      │
│      brief → skill 選択（hyperframes / static_hero）               │
│      → マルチターン（user × agent × OpenCode CLI × OpenDesign）    │
│      ├─ hyperframes → motion film .mp4   (→ hyperframes/)          │
│      └─ static_hero → デザイン HTML page (→ html_asset/)            │
│                                                                    │
│  Path B1 · Agent 2 SetupRunner（プロジェクト起動）                   │
│      ソース + README → setup_plan.json                             │
│      → install / seed / start services（user が plan を承認）       │
│                                                                    │
│  Path B2 · Demo Driver Agent（プロジェクトのデモ）                   │
│      ソースを読んで何をデモすべきか自分で判断                          │
│      ├─ web: playwright BrowserSession                             │
│      │   (goto / click / fill / wait / screenshot…)                │
│      └─ CLI: PtySession (pty_send / wait_for / screen)             │
│      → セッション全体を録画（時間制限なし）                            │
│      → recordings/demo.mp4 + demo_captions.jsonl                   │
│   ⏸ Gate #2: 実行中ユーザーが介入可、終了後に録画をレビュー            │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 3 · Agent 3 RemotionComposer ─────────────────────────────┐
│   M3a · cutting_plan.json — エージェントが自律的にミックス：         │
│         • recording (Demo Driver の実録画)                         │
│         • hyperframe (OpenDesigner motion_film .mp4)               │
│         • html (OpenDesigner static_hero ページ + scroll/zoom)      │
│         + キャプション・トラックを demo_captions.jsonl から取得       │
│   M3b · cutting_plan → TSX → npm install → render → v1.mp4         │
│   ⏸ Gate #3: ユーザーが編集方針 + 各 vN をレビュー                    │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 4 · Agent 4 BGMComposer ──────────────────────────────────┐
│   M4a · numpy ビート・スキャフォールド（kicks をシーン区切りに整列）   │
│   M4b · MusicGen-small/melody（CUDA fp16、推論約 40 秒）             │
│   M4c · ffmpeg mux → v1_bgm_final.mp4                              │
│   ⏸ Gate #4: ユーザーが BGM をレビュー                                │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─ Phase 5 · Agent 5 VoiceOver（4 ステップ直列）─────────────────────┐
│   Step1 · LLM が voiceover_script.json を作成（zh-CN/en-US）        │
│   Step2 · edge-tts でキューごとに mp3 合成                          │
│   Step3 · voice_timeline で voice_full.wav を組み立て                │
│   Step4 · BGM ducking + amix → final_zh-CN/en-US.mp4                │
│   ⏸ Gate #5: ユーザーがスクリプトをレビュー                            │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
[final_zh-CN.mp4 / final_en-US.mp4]
```

---

## 🤖 Demo Driver — プロジェクトの中核

固定時間のスクリーンレコーダーではない。Demo Driver は 4 つのツール群を持つ自律エージェント:

### 1. ソース読取ツール（*何を*デモするかを決定）
`list_dir` / `read_file` / `find_files` / `grep` — ルートハンドラ、メインループ、CLI コマンド、機能モジュールを読む。**何をデモするかはコードが決める。brief の「セールスポイント」リストではない** — マーケコピーは位置付けの話、ソースコードは現実の話。

### 2. プロジェクト操作ツール（*どう*デモするかを決定）

**Web モード**（`setup_plan.json` に `health_url` がある場合）:
```
browser_goto / browser_click / browser_fill / browser_press / browser_scroll
browser_hover / browser_wait_for
browser_screenshot     → PNG をインラインで返す（vision）
browser_visible_text   → document.body.innerText
browser_a11y_snapshot  → CDP Accessibility.getFullAXTree
browser_interactables  → 安定セレクタ付きのクリック/入力可能要素一覧
```
裏側は `tools/browser_session.py` の `BrowserSession`：playwright chromium + `record_video_dir` でセッション全体をネイティブ収録、stop 時に webm → mp4 トランスコード。

**CLI モード**（services なし、または service.command が CLI プログラム）:
```
pty_send (stdin 書き込み)        pty_wait_for (画面で正規表現)
pty_screen (pyte グリッド全体)    pty_read_recent (末尾 n 行)
pty_is_alive
```
裏側は `tools/pty_session.py` の `PtySession`：subprocess + pyte 端末エミュレータ + バックグラウンド・サンプリング・スレッドが 1/fps ごとに PNG をレンダ、stop 時に ffmpeg で mp4 に結合。**30 秒のデッドラインなし** — driver が stop 呼ぶまで継続。

### 3. デモ制御ツール
- `mark_caption(zh, en, importance)` — *現在の*録画タイムスタンプにバイリンガル字幕タグ。`demo_captions.jsonl` に書き出し、Phase 3 がキャプション・トラックとして利用。LLM に文言を発明させない。
- `ask_user(question)` — ユーザー返信までブロック（web UI でユーザーが入力 → `live_feedback.jsonl` に追記 → driver が読む）。
- `finish_demo(summary, completeness)` — driver 自身がデモ完了と判断、ループ終了、セッション停止、mp4 確定。
- `log_thought(text)` — エージェントログのみ書き込み（動画には影響なし）。

### 4. User-in-the-loop（ツールではなく制御チャネル）
ユーザーは web UI のテキスト欄にいつでも入力可能、例:
```
認証を飛ばしてメイン機能を直接見せて
このステップ早すぎ — 次は wait_for を使って
もう十分、まとめて
```
各エントリは `live_feedback.jsonl` に追記。**LLM 呼び出しごとの直前**に driver が新規エントリを読み、conversation にスプライス:
```
[USER LIVE FEEDBACK]: 認証を飛ばしてメイン機能を直接見せて
```
LLM は次のターンで読んで調整する。

---

## 🎨 3 トラック・ミキシング（Phase 3）

`cutting_plan.json` の `background.type` は 5 値: `color` · `gradient` · `recording` · `hyperframe` · `html`。

| Type | ソース | 用途 |
|---|---|---|
| `recording` | Demo Driver 録画 `recordings/test.mp4` | 真実味 — 実際のプロジェクト動作 |
| `hyperframe` | OpenDesigner motion_film 出力 `hyperframes/*.mp4` | 洗練 — デザイナーアニメ |
| `html` | OpenDesigner static_hero 出力 `html_asset/index.html` | hero / intro / outro — 実 `<iframe>` + scroll + zoom |
| `color` / `gradient` | 単色 / グラデ | テキスト密集シーンの可読性 |

**事前比率なし。** RemotionComposer エージェントは `available_assets` を受け取り、ミックスを自分で決める。システムプロンプトはハードルールのみ:
- **R3** 頭尾スキップ: recording は先頭/末尾 5 秒スキップ必須（不安定フレーム）。hyperframe / html は不要（OpenDesign 出力はクリーン）。
- **R4** 可読性: 大きく短いタイトル文字 → recording/hyperframe + darken 0.65–0.85 または html。小さい/長い本文 → color/gradient/html、または recording/hyperframe + darken ≥ 0.7。
- **R5**: 連続シーンはデフォルトで 15 フレームのクロスフェード。
- **R6**: 全 `source_path` は `available_assets` に存在必須。

`tools/remotion_codegen.py` が plan を TSX に翻訳:
- `recording` / `hyperframe` → `<Video>` / `<OffthreadVideo src={staticFile(...)} startFrom={...}>`
- `html` → カスタム `<HtmlBg>` コンポーネント: 実 `<iframe>` + `onLoad` で `contentWindow.scrollTo(0, scrollMax * pct)` + CSS `transform: scale(zoom)`

---

## 🚀 Quick Start

### 1. 環境
```bash
python -m venv .venv
.venv/Scripts/pip install -e .                  # Windows
# または
.venv/bin/pip install -e .                       # *nix

# Playwright Chromium（Demo Driver web モードで必須）
.venv/Scripts/python -m playwright install chromium

# 外部要件:
# - ffmpeg を PATH に
# - Node.js 24 + pnpm 10（Phase 3 Remotion + Phase 2A OpenDesign）
# - Docker Desktop（セルフホスト Langfuse 用）
# - CUDA GPU（オプション、Phase 4b MusicGen を高速化）
```

### 2. セルフホスト Langfuse を起動
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

# 6 サービス起動: postgres + clickhouse + redis + minio + langfuse-web + langfuse-worker
docker compose --env-file .env up -d
# 約 30 秒待って http://localhost:3000 へ
```

> **中国大陸ネットワーク**: `cgr.dev/chainguard/minio` が pull できない場合
> `docker.io/minio/minio` に置換。Redis 6379 がローカルと衝突する場合は
> `127.0.0.1:6380:6379` に。

### 3. プロジェクト `.env`
```ini
ANTHROPIC_BASE_URL=https://ark.cn-beijing.volces.com/api/coding
ANTHROPIC_API_KEY=<your key>
ARK_BASE_URL_OPENAI=https://ark.cn-beijing.volces.com/api/coding/v3
ARK_KEY_1=<your key>
LLM_REASONING=claude-sonnet-4-20250514
LLM_FAST=deepseek-v3.2

LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-XXXX     # langfuse-stack/.env と一致
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

OpenCode を `~/.config/opencode/opencode.json` で設定:
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

### 5. パイプライン実行
```bash
.venv/Scripts/python -m src.cli analyze https://github.com/<user>/<repo>

.venv/Scripts/python -m src.cli serve --port 7860
# → http://127.0.0.1:7860/
```

---

## 📊 可観測性

各 run は `workspace/<project>/runs/<run_id>/` に永続化:

```
project_brief.md                    Phase 1 出力
setup_plan.json                     Phase 2B SetupRunner 出力
recordings/
  demo.mp4                          Demo Driver 録画
  test.mp4                          採用済み Phase 2 録画 (= demo.mp4)
demo_captions.jsonl                 バイリンガル字幕トラック
demo_summary.md                     driver の総括
demo_driver_progress.json           実行時進捗（web UI がポーリング）
live_feedback.jsonl                 ユーザー → driver チャットチャネル
hyperframes/*.mp4                   OpenDesigner motion_film 出力
html_asset/                         OpenDesigner static_hero 出力
cutting_plan.json                   Phase 3a Composer 出力
remotion/                           Phase 3b 生成 Remotion プロジェクト
outputs/v1.mp4                      Phase 3 レンダ
outputs/v1_bgm_final.mp4            Phase 4 BGM 付き
outputs/final_zh-CN.mp4             Phase 5 ナレーション付き最終
events.jsonl                        全ライフサイクルイベント
logs/
  pipeline.jsonl                    全エージェント行ログ
  agent1_analyzer.jsonl             Phase 1
  agent2_setup.jsonl                Phase 2B plan / exec
  demo_driver.jsonl                 Phase 2C
  agent3_remotion.jsonl             Phase 3
  agent4_bgm.jsonl                  Phase 4
  agent5_voice.jsonl                Phase 5
  agent6_opendesigner.jsonl         Phase 2A
opendesign/state.json               Agent 6 セッション状態
```

**Langfuse UI**: http://localhost:3000/ — 全 LLM 呼び出し + traced_step スパン + verify イベントを docker volumes（postgres + clickhouse + minio + redis）に保存（再起動しても残る）。

各エージェント入口に `@traced_agent("Agent N · 子ステップ", phase=N)` デコレータが付与され、`agent_start`/`agent_done` イベントを自動 emit + OTEL 親スパンを生成。内部ステップは `traced_step("MusicGen.load_model", ...)` コンテキストマネージャでネスト — Langfuse 上では Agent 4 BGM → MusicGen.load_model → MusicGen.tokenize_inputs → MusicGen.generate → MusicGen.write_wav が 1 本のツリーで見える。Anthropic SDK 呼び出しは `openinference-instrumentation-anthropic` で自動計装、prompt / response / tool_use / tool_result / token usage が捕捉される。

---

## 🛠 技術スタック

| レイヤ | 技術 |
|---|---|
| Agent runtime | Python 3.13 + Anthropic SDK + Volcengine Ark Coding Plan |
| Web UI | FastAPI + Jinja2 + HTMX + SSE + Tailwind CSS |
| 可観測性 | Langfuse（docker self-hosted）+ loguru + OpenTelemetry HTTP exporter + openinference-instrumentation-anthropic |
| Demo Driver / web | playwright (chromium headless) + record_video_dir + CDP a11y |
| Demo Driver / CLI | subprocess + pyte（端末エミュレータ）+ PIL + ffmpeg image-seq |
| 視覚アセット | OpenDesign daemon + OpenCode CLI + HyperFrames (HTML→MP4 + GSAP) |
| 動画編集 | Remotion (React + TSX → mp4)、OffthreadVideo + カスタム HtmlBg iframe コンポ |
| BGM | numpy ビート + facebook/musicgen-{small,melody} (PyTorch CUDA fp16) |
| ナレーション | edge-tts (Microsoft Azure Neural TTS) + ffmpeg sidechaincompress ducking |

---

## 📁 リポジトリ構成

```
src/
├── agents/
│   ├── project_analyzer.py    Agent 1
│   ├── setup_runner.py        Agent 2 SetupRunner（plan のみ、host が実行）
│   ├── demo_driver.py         Demo Driver Agent（Phase 2C 自律デモ）★
│   ├── remotion_composer.py   Agent 3 cutting_plan + 3 トラックミックス
│   ├── voice_over.py          Agent 5
│   └── opendesigner.py        Agent 6
├── tools/
│   ├── pty_session.py         PtySession (CLI デモ + 録画) ★
│   ├── browser_session.py     BrowserSession (web デモ + 録画) ★
│   ├── bgm_*.py               Agent 4 BGM 各ステップ
│   ├── tts_edge.py / voice_timeline.py / bgm_duck_mux.py  Agent 5
│   ├── opendesign_*.py        Agent 6 daemon クライアント / ライフサイクル
│   ├── remotion_codegen.py    Agent 3 cutting_plan → TSX (HtmlBg を含む)
│   └── remotion_render.py     npx remotion render
├── observability/             Langfuse OTLP tracing + loguru + EventBus
├── verify/                    ffprobe アーティファクト検証
├── web/                       FastAPI + Jinja2 テンプレート
├── pipeline.py                Pipeline クラス + ステートマシン
└── cli.py                     viedo CLI

★ = Demo Driver パスの主要新コンポーネント
```

---

## 📚 関連ドキュメント

- [WORKFLOW.md](WORKFLOW.md) — 5 フェーズ・ステートマシン、ゲート定義、可観測性 3 点セット仕様
- `src/agents/demo_driver.py` 冒頭 docstring — driver 設計思想（agent loop + user-in-loop、anti-state-machine）
- Langfuse セルフホスト公式: https://langfuse.com/self-hosting

---

## 🤝 謝辞

- [nexu-io/open-design](https://github.com/nexu-io/open-design) · Open Design daemon + skills + HyperFrames
- [remotion](https://github.com/remotion-dev/remotion) · React ベース動画合成
- [microsoft/playwright](https://github.com/microsoft/playwright-python) · ブラウザ自動化 + 動画録画
- [selectel/pyte](https://github.com/selectel/pyte) · Pythonic 端末エミュレータ
- [facebook/musicgen](https://huggingface.co/facebook/musicgen-melody) · テキスト駆動音楽生成
- [edge-tts](https://github.com/rany2/edge-tts) · Microsoft Azure Neural TTS ラッパー
- [Langfuse](https://github.com/langfuse/langfuse) · LLM 可観測性（self-hosted）

---

## 📝 ライセンス

Apache-2.0

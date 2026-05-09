# 宣发视频制作 · Agent Workflow 开发文档

> **目标**：给定一个 GitHub 仓库地址，通过线性 Agent 工作流 + 关键节点用户拍板，自动产出一支高质量宣发视频。
> **受众**：开发该 Agent workflow 系统的工程师。
> **实现语言**：**Python**（明确指定，不要 TypeScript / Node）。
> **来源**：基于 `Football Match Simulator` 完整实战提炼，所有约束和坑均已在生产中踩过。

---

## 0 · 角色定义

本工作流是**线性 Agent 流水线**——每个阶段由一个 Agent 全权负责，前一阶段产物验收通过后才进入下一阶段。**不存在**"主 Agent 调度子 Agent"的多层结构。

| 角色 | 职责 |
|---|---|
| **User** | 提供 GitHub 地址、各阶段拍板、与 Agent 讨论方案 |
| **Agent 1 · ProjectAnalyzer** | clone 仓库、读 README/核心代码，输出项目特点/作用/目的报告，与 User 迭代到满意 |
| **Agent 2 · SetupRunner** | 配置环境、装依赖、启动项目服务、录屏（**录屏时长由 User 决定**）——三件事一个 Agent 干完 |
| **Agent 3 · RemotionComposer** | 与 User **讨论** "HTML + 录屏如何组合"思路；用 Remotion 写完整视频工程；迭代出 vN |
| **Agent 4 · BGMComposer** | 节拍脚手架对齐切点 → MusicGen 升级音色 → ffmpeg mux 进视频 |
| **Agent 5 · VoiceOver** | 写脚本（与 User 讨论）→ edge-tts 逐句 → Timeline 拼合 → BGM ducking mux |
| **OpenDesign** | 外部 SaaS，**不是 Agent**。User 自己用 OpenDesign 出 HTML 视觉概念稿 |

**核心原则**：
- 严格线性，下一个 Agent 在上一个 Agent 产物验收通过后启动
- 唯一的"并行"是阶段二：**OpenDesign（User 在 SaaS 上操作）** 与 **Agent 2（本地配置/录屏）** 在同一时段进行，互不依赖
- Agent 报告"完成"不算数，必须主动验收产物（见第 6 节）
- 每个 Agent 都是 Python LLM Agent（基于 Anthropic SDK），可直接调用 shell（git / ffmpeg / npm / edge-tts / python）

---

## 1 · 阶段一 · 项目分析（Agent 1）

```
[User 给出 GitHub URL]
        │
        ▼
[git clone 到 ASCII 路径]
        │
        ▼
[Agent 1 · ProjectAnalyzer]
  ├── 读 README / 核心代码 / prompt 文件 / 配置文件
  ├── 总结产品定位、作用、目的、目标人群、核心功能、独特卖点
  ├── 提炼视觉关键词（极简 / B&W / 金色 / 电影感 …）
  └─→ project_brief.md
        │
        ▼
[User 审阅 → 与 Agent 1 多轮迭代直到满意]   ← 人工介入点 #1
        │
        ▼
   project_brief.md ✅
```

**project_brief.md 必须包含**：
- 产品一句话定位
- 作用与目的（一段话）
- 目标受众
- 3–5 个独特卖点
- 视觉关键词
- 不超过 3 个竞品参考

**约束**：
- 报告先于 OpenDesign 协作产出——不能让 User 从零向 OpenDesign 描述
- Clone 路径必须 ASCII（见 R1）

---

## 2 · 阶段二 · HTML 视觉稿 + 项目录屏（同时段）

阶段二是全流程**唯一**的并行段。两路独立推进，都通过后才进入阶段三。

### 2.1 路 A · OpenDesign HTML 协作（User 自己操作）

```
[project_brief.md]
        │
        ▼
[User 把 brief 喂给 OpenDesign，多轮协作]
        │
        ▼
[<project>-video.zip 解包到 html_asset/]
        │
        ▼
[本地加载 index.html 验收]
        │
        ▼
   html_asset/ ✅
```

OpenDesign 不是 Agent，没有 Python 代码驱动它。User 自己操作 SaaS，把产物 zip 放回工作目录，工作流脚本负责解包验收。

**约束**：
- HTML zip 解包后必须本地可打开（字体/资源不能是外链 CDN）
- HTML 定的是**视觉调性**，不是产品本体的截图

### 2.2 路 B · Agent 2 SetupRunner（配置 + 运行 + 录屏，同一 Agent）

User 在交互中说"配一下、跑起来、录屏 X 分钟"，Agent 2 一次性把三件事做完：

```
[Agent 2 · SetupRunner]
  ├── 检测项目类型（package.json / requirements.txt / Cargo.toml / pyproject.toml / ...）
  ├── 安装依赖（pip install -r / npm install / ...）
  ├── seed 数据库 / 配置 env vars（必要时与 User 确认）
  ├── 启动服务（前端 + 后端，必要时并发多进程）
  ├── 先录 1min 测试 → 让 User 确认捕获区域只包含项目窗口（不是整桌面）
  └── 正式录屏（仅项目窗口）
        └─→ <name>_<resolution>_<timestamp>.webm  ✅
```

**录屏规格**：
- 分辨率：1080p 最低，1440p 优先
- **时长：由 User 决定**（建议 ≥ 3 分钟以留剪辑空间，但硬下限不强制）
- 格式：`.webm`（OBS 默认）或 `.mp4`
- 命名示例：`football_match_TEST5min_1440p_20260506_134203.webm`

**⚠️ 关键约束**：
- **头尾各 5s 不可用**（UI 加载中 / 收尾未完成）→ 后续所有 `startFrom` 必须 ≥ 5s，结束时间必须 ≤ 总时长 − 5s
- **必须先录 1min 测试**确认捕获的是项目窗口而非整桌面——之前有过录了 1 小时整桌面、作废重录的前车之鉴
- **路径必须 ASCII**：FFmpeg 在含中文字符的路径会报 `0xC0000142` DLL init 崩溃，Windows 上用 `mklink /J promo-link C:\ascii\path` 做 junction

### 2.3 阶段二 · 收尾验收

两路都 ready 才进阶段三：
- HTML：`html_asset/index.html` 用 Playwright/headless Chrome 加载，无控制台报错
- 录屏：`ffprobe` 验时长 ≥ User 指定值、分辨率达标

---

## 3 · 阶段三 · Remotion 视频合成（Agent 3）

```
[html_asset/] ──┐
                ├──→ [Agent 3 · RemotionComposer] ──→ vN.mp4
[recording.webm]┘             ↑
                       [User 讨论方案 + 反馈]   ← 人工介入点 #3
```

**核心特征**：Agent 3 不能独自决定剪辑思路。Agent 3 启动后**必须先与 User 讨论**：
- HTML 哪些片段做开场/转场/标语？
- 录屏哪些片段嵌入？哪些时间点是高光？
- 整体节奏（30s / 45s / 60s）？
- 哪些场景配大字标语？哪些放小字说明？

讨论清楚 → Agent 3 写 Remotion 工程 → 渲染 v1 → User 看 → User 给反馈（EditOp）→ Agent 3 局部改 → v2 → ……直到 User 拍板。**User 只讨论和拍板，不动手改代码**。

### 3.1 合成规则（Agent 3 必须遵守）

**背景选择规则**：
| 场景类型 | 背景 |
|---|---|
| 大字标语（字体 > 80px，字数 ≤ 10） | 录屏当背景 + darken overlay |
| 小字 / 多元素（气泡、列表、说明文字） | 干净纯色/渐变背景（PitchBackground） |
| 纯录屏展示片段 | 录屏全屏，无叠加 |

> 违反此规则的后果：小字压在录屏上完全看不清，v3 S5 的 22 个气泡就是这么翻车的。

**过渡规则**：
- 相邻 `<Sequence>` 重叠 **15 帧**（@30fps = 0.5s）做 crossfade
- 禁止每个场景末尾单独 fade-out——会导致黑屏闪烁

**录屏嵌入规则**：
- `startFrom` 换算到帧：`round(t_seconds * fps)`
- 所有 startFrom 必须 ≥ `5 * fps`（跳过头部 5s）
- 所有结束时间必须 ≤ `(recording_duration - 5) * fps`（跳过尾部 5s）
- darken overlay 推荐值：`0.7–0.85`（实测 0.82 在 S5 小字场景可用）

### 3.2 迭代约定

User 的调整指令格式化为 `EditOp`（见第 7 节），Agent 3 接收后局部修改，不重渲整个工程。
每轮产出独立文件：`v1.mp4` → `v2.mp4` → `v3.mp4`，**不覆盖**。

### 3.3 Gate：视频视觉通过

User 看完 vN 拍板"过了"才继续。Agent 3 自己不能判定。

---

## 4 · 阶段四 · BGM（Agent 4）

```
[视频 vN.mp4]
    │
    └─→ [Agent 4 · BGMComposer]
            ├── Step 1: 节拍脚手架（Python + numpy + wave）
            │       ├── 读视频时间轴，提取场景切点时间戳
            │       ├── 按切点排 kick/snare/hat/sub-bass/lead
            │       └─→ bgm_scaffold.wav
            │
            ├── Step 2: MusicGen-melody 升级
            │       ├── 输入：bgm_scaffold.wav（melody 引导）
            │       ├── 输入：文字 prompt（风格/BPM/乐器/情绪）
            │       └─→ bgm_final.wav
            │
            └── Step 3: Mux（视频不重渲）
                    ffmpeg -i video.mp4 -i bgm_final.wav \
                           -c:v copy -c:a aac -b:a 192k \
                           -map 0:v:0 -map 1:a:0 -shortest \
                           promo_vN_bgm.mp4
```

### 4.1 节拍脚手架详解

脚手架的作用：让节奏点**精确对齐**视频场景切点。MusicGen 出来的节奏是随机的，不会自动对齐画面，所以必须先做脚手架定锚点，再用 MusicGen 升级音色。

```python
import numpy as np, wave, struct

SAMPLE_RATE = 44100
BPM = 140
BEAT = 60 / BPM  # 0.4286s per beat

# 场景切点（从视频时间轴读取，单位秒）
CUT_POINTS = [0, 2.5, 5.0, 9.0, 14.0, 23.5, 28.0, 33.0, 36.0]

# 生成 kick：在每个切点落鼓
def make_kick(t, duration=0.15, freq=60):
    ...

# 23.5s 是 GOAL drop，提前 0.5s 放 impact
```

BPM 参考：
- 激情/运动/hype：130–150 BPM
- 科技/未来感：120–135 BPM
- 平静/品牌：80–100 BPM

### 4.2 MusicGen prompt 参考

```
"wild exuberant hard-hitting hype trap, 140 BPM, 808 sub bass, snare rolls,
 melodic lead synth, stadium atmosphere, no vocals, professional mix"
```

**硬件要求**：
- GPU fp16：≥ 6GB VRAM（4GB 临界，可能 OOM 退 CPU）
- CPU fallback：10–15 分钟，可用

### 4.3 Gate：BGM 审听通过

User 听 BGM 候选后给反馈。**人耳判官优于客观指标**，预留迭代空间。
通过后进阶段五。

---

## 5 · 阶段五 · 配音（Agent 5）

```
[BGM 视频 vN.mp4]
    │
    └─→ [Agent 5 · VoiceOver]（4 步串行）
            │
            ├── Step 1: 与 User 讨论文案 → voiceover_script.json
            │
            ├── [User 审稿]  ← 人工介入点 #5
            │
            ├── Step 2: TTS 生成（edge-tts 逐句）
            │
            ├── Step 3: Timeline 拼合 voice_full.wav
            │
            └── Step 4: BGM ducking + amix mux
                    └─→ final_vN.mp4
```

### 5.1 Step 1：脚本时间轴

Agent 5 读 Remotion 工程的场景定义，与 User 讨论文案后输出：

```json
[
  { "id": "S1", "t_start": 0.0,  "t_end": 4.5,  "text": "Twenty-two AI agents.", "lang": "en" },
  { "id": "S2", "t_start": 5.0,  "t_end": 9.0,  "text": "One match.",             "lang": "en" },
  { "id": "S4", "t_start": 14.0, "t_end": 19.0, "text": "Every decision, computed.", "lang": "en" },
  { "id": "outro", "t_start": 31.5, "t_end": 34.8, "text": "Twenty-two hearts. One game. Twenty-twenty-six.", "lang": "en" }
]
```

规则：
- `t_start` / `t_end` 来自 Remotion `<Sequence>` 时间轴，不硬编码
- 不是每个场景都需要配音——纯视觉场景留空，无静音填充
- 中英双版本各一份 json，lang 字段区分

### 5.2 Step 2：TTS 生成

**工具**：`edge-tts`（微软 Azure Neural TTS，免费，无需 API Key，需联网）

```bash
pip install edge-tts

# 英文
edge-tts --voice en-US-EricNeural \
         --text "Twenty-two hearts. One game. Twenty-twenty-six." \
         --write-media voice_outro.wav

# 中文（备选声线）
edge-tts --voice zh-CN-YunxiNeural \
         --text "二十二颗心，一场游戏，二零二六。" \
         --write-media voice_outro_cn.wav
```

生成速度：每句 < 1s（网络调用）。离线备选：CosyVoice（中文）/ OpenVoice。

### 5.3 Step 3：Timeline 拼合

把各句 wav 按时间轴拼成与视频等长的 `voice_full.wav`（空白处用静音填充）：

```bash
# 对每个有配音的段：在 t_start 位置插入对应 wav，其余填静音
ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t {video_duration} silence.wav

# 逐句叠加
ffmpeg -i silence.wav -i voice_S1.wav \
       -filter_complex "[0:a][1:a]amix=inputs=2:normalize=0:weights=1 1:duration=first[a]" \
       -map "[a]" voice_step1.wav
# 重复叠加直到所有句子合入

# 最终验证
ffprobe voice_full.wav  # 时长必须 == 视频时长 ±0.1s
```

### 5.4 Step 4：BGM Ducking + Mux

```bash
ffmpeg \
  -i video_bgm.mp4 \
  -i voice_full.wav \
  -filter_complex "
    [0:a]volume=1.0[bgm_base];
    [bgm_base]volume=enable='between(t,{T_VOICE_START},{T_VOICE_END})':volume=0.3[bgm_duck];
    [bgm_duck]afade=t=in:ss={T_FADE_IN}:d=0.5,afade=t=out:st={T_FADE_OUT}:d=0.5[bgm];
    [1:a]volume=0.7[voice];
    [bgm][voice]amix=inputs=2:normalize=0[aout]
  " \
  -map 0:v \
  -map "[aout]" \
  -c:v copy \
  -c:a aac -b:a 192k \
  final_vN.mp4
```

**参数说明**：
| 参数 | 值 | 原因 |
|---|---|---|
| voice volume | 0.7 | 人声不要顶满，留 headroom |
| BGM ducked volume | 0.3 | 约 −10dB，实测人声清晰可辨 |
| fade duration | 0.5s | 避免 duck 切换的咔哒声 |
| `normalize=0` | **必须加** | 默认 normalize=1 会把所有路音量 ÷ inputs 数，BGM 会莫名变软 |
| `-c:v copy` | **必须** | 视频流不重渲，只换音轨 |

多个配音段：`T_VOICE_START/END` 替换为每段的实际时间戳，多段 ducking 用 `+` 拼接 `between()` 条件：
```
enable='between(t,31.5,34.8)+between(t,5.0,9.0)'
```

**双语执行**：中文版和英文版各跑一次完整的 Step 1–4，共享同一条视频底层，只换音轨。

---

## 6 · 人工介入点汇总

| # | 时机 | User 做什么 | 未通过时 |
|---|---|---|---|
| **#1** | Agent 1 出 project_brief.md | 审阅，与 Agent 1 迭代到满意 | Agent 1 改写报告，再给 User |
| **#2** | OpenDesign HTML zip 上传 + Agent 2 录屏完成 | 确认 HTML 本地可加载；确认录屏只录项目窗口 | 不放行阶段三 |
| **#3** | 进入 Agent 3 / 每轮 vN 出来 | 先与 Agent 3 讨论剪辑思路；vN 出来后给 EditOp 或拍板 | Agent 3 收 EditOp 局部改 |
| **#4** | Agent 4 BGM 候选出来 | 听 BGM 给反馈（节奏/音色/深度） | Agent 4 调参重生 |
| **#5** | Agent 5 出 voiceover_script.json | 审稿改文案/时间戳 | 修脚本，重新 TTS |

**设计原则**：每一步都有 User 拍板，没有"全自动一路冲到底"——但 User 只**讨论和审阅**，不亲自写代码或剪片。

---

## 7 · 验收机制（Agent 不可信原则）

> Agent 说"完成了"不等于产物可用。每个阶段工作流必须主动验收。

### 7.1 视频验收（ffprobe）

```bash
ffprobe -v quiet -print_format json -show_streams -show_format output.mp4
```

检查项：
- `streams` 中有 `codec_type: "video"` 且 `codec_name: "h264"`
- `streams` 中有 `codec_type: "audio"` 且 `codec_name: "aac"`（BGM/配音阶段）
- `format.duration` 在预期时长 ±0.5s 内
- `format.size` > 5MB（太小说明渲染可能只出了几帧）

### 7.2 音频验收

```bash
ffprobe -v quiet -print_format json -show_streams voice_full.wav
# duration 必须 == 视频时长 ±0.1s
```

### 7.3 录屏验收

```bash
ffprobe -v quiet -show_format recording.webm
# duration ≥ User 指定时长
# 检查 width/height 符合预期分辨率
```

### 7.4 HTML 验收

用 Playwright / headless Chrome 加载 `index.html`，检查：
- 无控制台报错
- 动画可播放
- 字体/图片全部加载

---

## 8 · 类型定义（Python）

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from pathlib import Path

# ── 顶层输入 ─────────────────────────────────────────────
@dataclass
class ProjectInput:
    repo_url: str                              # GitHub URL
    branch: Optional[str] = None
    style_keywords: list[str] = field(default_factory=list)   # ["极简", "B&W", "金色"]
    style_reference: Optional[str] = None      # 参考视频/截图路径

# ── 产物 ─────────────────────────────────────────────────
@dataclass
class HtmlAsset:
    path: Path
    verified: bool

@dataclass
class RecordingAsset:
    path: Path
    resolution: str            # "1440p"
    duration: float            # seconds
    unusable_head: float = 5.0
    unusable_tail: float = 5.0

@dataclass
class VideoAsset:
    path: Path
    duration: float
    version: int               # vN
    has_bgm: bool
    has_voice: bool

@dataclass
class BgmAsset:
    path: Path
    bpm: int
    sourced_by: Literal['scaffold', 'musicgen']

@dataclass
class VoiceSegment:
    id: str
    t_start: float
    t_end: float
    text: str
    lang: Literal['en', 'zh']
    wav_path: Optional[Path] = None

@dataclass
class VoiceAsset:
    path: Path
    lang: Literal['en', 'zh']
    voice: str                 # "en-US-EricNeural"
    segments: list[VoiceSegment]

Asset = Union[HtmlAsset, RecordingAsset, VideoAsset, BgmAsset, VoiceAsset]

# ── 用户编辑指令（5 种局部编辑） ──────────────────────────
@dataclass
class ReplaceBg:
    op: Literal['replace_bg'] = 'replace_bg'
    range: tuple[float, float] = (0.0, 0.0)
    source_asset: str = ''
    source_range: tuple[float, float] = (0.0, 0.0)

@dataclass
class ExtendScene:
    op: Literal['extend_scene'] = 'extend_scene'
    scene_id: str = ''
    delta_sec: float = 0.0

@dataclass
class ChangeClip:
    op: Literal['change_clip'] = 'change_clip'
    scene_id: str = ''
    new_start_from: float = 0.0

@dataclass
class FixTransition:
    op: Literal['fix_transition'] = 'fix_transition'
    between_scenes: tuple[str, str] = ('', '')

@dataclass
class AdjustDarken:
    op: Literal['adjust_darken'] = 'adjust_darken'
    scene_id: str = ''
    value: float = 0.8        # 0–1

EditOp = Union[ReplaceBg, ExtendScene, ChangeClip, FixTransition, AdjustDarken]

# ── 流水线状态 ────────────────────────────────────────────
@dataclass
class PipelineState:
    phase: Literal[1, 2, 3, 4, 5]
    gate: Literal[
        'waiting_brief_approval',
        'waiting_html', 'waiting_recording',
        'waiting_video_approval',
        'waiting_bgm_approval',
        'waiting_script_approval',
        'running', 'done',
    ]
    assets: list[Asset]
    current_version: int
    manifest: dict[str, dict]   # name -> {path, verified, timestamp}
```

---

## 9 · 跨阶段约束与坑（10 条，全部实战踩过）

| # | 规则 | 出处 |
|---|---|---|
| **R1** | 工作目录必须 ASCII；Windows 中文路径用 `mklink /J` 做 junction | FFmpeg `0xC0000142` DLL init 崩在 `运营/` 目录 |
| **R2** | 录屏先录 1min 测试，确认只录项目窗口而非整桌面；正式录屏时长由 User 决定 | v0 录了 1 小时整桌面，全部作废 |
| **R3** | 录屏头尾各 5s 不可用，所有 startFrom ≥ 5s，结束 ≤ 总时长 − 5s | v3 GameplayClip1 一处漏改，被 User 发现 |
| **R4** | 大字标语 → 录屏背景；小字/多元素 → 干净背景 | v3 S5 的 22 个气泡压在录屏上，完全看不清 |
| **R5** | 相邻 Sequence 重叠 15 帧 crossfade；禁止每场景末尾单独 fade-out | v3 黑屏闪烁问题 |
| **R6** | Agent 说"完成"不算，工作流必须 ffprobe 验时长/大小/流数 | v2 子 Agent 死循环，报"完成"实则 encode 崩 |
| **R7** | 每轮产物独立命名 vN，不覆盖旧版本 | 便于回滚比对，出错时不丢数据 |
| **R8** | BGM 先做节拍脚手架对齐画面切点，再用 MusicGen 升级音色 | 直接用 MusicGen 节奏随机，无法精确对齐场景 |
| **R9** | ffmpeg amix 必须加 `normalize=0` | 默认 normalize=1 会把所有音轨音量 ÷ 路数，BGM 莫名变软 |
| **R10** | 全轨配音 = 所有有旁白场景都生成 wav；不能只配 Outro | v5 只配了结尾一句，其余场景无配音 |

---

## 10 · 工作流总状态机

```
[START: User 给 GitHub URL]
       │
       ▼
[Agent 1 ProjectAnalyzer]
       │
   project_brief.md
       │
   [User 审阅迭代] ← #1
       │ 通过
       │
       ├─────────────────────┬──────────────────────┐
       ▼                     ▼                      
[User × OpenDesign]   [Agent 2 SetupRunner]
       │                     │
   html_asset/✅         recording.webm✅
       │                     │
       └────── 阶段二完成 ────┘
                   │
                   ▼
          [Agent 3 RemotionComposer]
                   │
              [User 讨论思路] ← #3a
                   │
                vN.mp4
                   │
              [User 反馈/EditOp] ← #3b
                loop 直到通过
                   │
                   ▼
              [Agent 4 BGMComposer]
                   │
            bgm_candidate.wav
                   │
              [User 审听] ← #4
                loop 直到通过
                   │
                   ▼
              [Agent 5 VoiceOver]
                   ├── 与 User 讨论 → voiceover_script.json
                   ├── [User 审稿] ← #5
                   ├── edge-tts 逐句
                   ├── Timeline 拼合
                   └── BGM ducking + mux
                          │
                       ffprobe ✅
                          │
                     [END] final_vN.mp4
```

---

## 11 · 用户交互层（Web UI）

CLI 的 `input()` 模式不能用于反复迭代——每个 phase 都需要反馈/重跑/通过的流畅交互。**Web UI 是强制基础设施**，所有 phase 共用一套 review/iterate 界面。

### 11.1 技术栈

| 层 | 选择 | 原因 |
|---|---|---|
| 后端 | **FastAPI** | Python 原生，与 Pipeline 同进程 |
| 模板 | **Jinja2** | FastAPI 默认，轻量 |
| 交互 | **HTMX** | Partial update 替代 SPA 构建链，零 JS 写代码 |
| 样式 | **TailwindCSS（CDN）** | 不引入构建步骤 |
| 流式 | **Server-Sent Events** | Agent 日志单向流到 UI 即可，不用 WebSocket |

### 11.2 通用 UI 结构

每个 phase 复用同一布局：
- **Header**：project / run_id / 当前 phase / 当前 gate / Langfuse UI 链接
- **Sidebar**：events timeline 实时滚动（events.jsonl tail）
- **Main**：产物预览区（markdown → HTML / 视频 player / 等）
- **Iteration history**：折叠列表，显示历次产物
- **Live console**：Agent 跑时 SSE 推送日志
- **Action bar**：反馈 textarea + `[↻ 重新生成]` + `[✓ 通过 → Phase N+1]`

### 11.3 后端路由

```
GET  /                              run 列表
GET  /runs/{id}                     主视图（按 phase 渲染）
GET  /runs/{id}/events              SSE：events.jsonl + agent 日志 tail
POST /runs/{id}/iterate             提交反馈 → 异步触发 Agent 重跑
POST /runs/{id}/approve             Gate 通过 → 进下一 phase
GET  /runs/{id}/artifacts/{name}    下载产物
```

### 11.4 Agent 异步执行

每次 `iterate`/`approve` 触发的 Agent 跑在后台 task 中（FastAPI BackgroundTasks 或 asyncio.create_task）。UI 通过 SSE 收到进度日志，不阻塞用户。

### 11.5 强制约束

- 每新增一个 phase（M2/M3/M4/M5）必须实现对应的 `_phase_<n>.html` partial 模板
- Agent 入口必须支持 async 调用（用 `asyncio.to_thread` 包裹同步 Agent 即可）
- 所有 user-triggered 动作必须经 web UI；CLI 只用于 doctor / verify 等基础设施命令

---

## 12 · 可观测性架构（强制基础设施）

任何 Agent 不接观测三件套不允许进主线：**Langfuse OTLP tracing + loguru 结构化日志 + Pipeline 事件总线**。

### 11.1 Tracing · Langfuse (self-hosted)

依赖：`opentelemetry-{api,sdk,exporter-otlp-proto-http}` + `openinference-instrumentation-anthropic`
基础设施：`docker compose up -d` 起 langfuse-web :3000 + worker + postgres + clickhouse + redis + minio。

在 `pipeline.py` 入口启动一次（`src/observability/tracer.py::setup`）：
```python
exporter = OTLPSpanExporter(
    endpoint="http://localhost:3000/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {b64(pk:sk)}"},
)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
AnthropicInstrumentor().instrument()  # Anthropic SDK 全部自动追踪
```

效果：每次 `messages.create`、tool_use、tool_result、token 用量、耗时 + 每个 `traced_step(...)` 内部 step（如 `MusicGen.load_model`、`MusicGen.generate`、`http.GET /api/...`、`edge_tts.synth_<id>`）都嵌套挂在 Agent span 下，Langfuse UI 一棵树看完。Agent 业务代码**零侵入**（agent 入口加 `@traced_agent` 装饰器即可）。

### 11.2 Agent 装饰器 · `@traced_agent`

每个 Agent 入口函数必须挂装饰器：

```python
@traced_agent("Agent 1 ProjectAnalyzer", phase=1)
def run_project_analyzer(inp: ProjectInput) -> Path:
    ...
```

装饰器职责：
- 把整个 Agent 执行包成一个父 span
- 注入 `run_id` / `agent_name` / `phase` 标签
- 异常自动捕获 + 堆栈写到 span event
- 退出时把产物路径写到 span attribute

### 11.3 结构化日志 · loguru

```python
from loguru import logger

logger.add(
    "workspace/{project}/runs/{run_id}/logs/{agent}.jsonl",
    serialize=True, level="DEBUG",
)
```

**所有 shell 调用必须走 `tools.shell.run()`**，禁止裸 `subprocess.run`。封装自动记录：命令、exit_code、stdout 长度、stderr 尾部 500 字节、耗时。

### 11.4 Pipeline 事件总线

```python
@dataclass
class PipelineEvent:
    ts: str
    run_id: str
    event: Literal['agent_start', 'agent_done', 'gate_enter', 'gate_pass',
                   'asset_verified', 'asset_failed', 'user_input']
    agent: Optional[str]
    payload: dict
```

每次状态机转移 / 产物验收 / User 介入 → 写 `events.jsonl` + 同步推 Langfuse span event。事后能完整重放整条流水线。

### 11.5 故障排查动线

1. **Langfuse UI**（`localhost:3000`）：trace 树看哪一步出错，点开看 prompt / response / tool args / traced_step 内部子 step
2. **events.jsonl**：宏观看流水线卡在哪个 Gate
3. **agent_<n>.jsonl**：该 Agent 内部 shell 调用细节
4. **state.json**：当前 PipelineState 快照

### 11.6 目录约定

```
workspace/<project>/runs/<run_id>/
    ├── traces/              # 本地 events.jsonl 备份（Langfuse 主存在 docker volumes）
    ├── logs/
    │   ├── pipeline.jsonl
    │   └── agent_<n>.jsonl
    └── events.jsonl
```

`run_id` 是 UUID4，每次完整流水线一个独立目录。

### 11.7 强制约束

- 每个 Agent 入口必须 `@traced_agent`——CI lint 强制
- 任何 shell 调用必须走 `tools.shell.run()`
- 每次 Agent 启停 / 产物验收 / Gate 转移必须发 `PipelineEvent`
- run_id 贯穿整次流水线

---

## 13 · 待研究 / TBD

| 议题 | 现状 | 方向 |
|---|---|---|
| HTML → video 自动化 | 当前手工把 HTML DOM 元素移植成 Remotion 组件 | 探索 Playwright 录 DOM 动画 → 直接作为 Remotion `<Video>` 素材 |
| 录屏高光自动选段 | 人工从 5min 录屏挑 30s | 用视觉模型自动标注"动作密集/UI 变化大"片段 |
| 配音中文声线 | `zh-CN-YunxiNeural` 待实测 | 对比 CosyVoice 克隆声线效果 |
| BGM ducking 自动化 | 当前手写 `between(t,...)` 时间窗 | 能否从 voiceover_script.json 自动生成 ffmpeg filter 表达式 |
| 双语终片差异化 | 目前共享视频底层只换音轨 | 是否需要中文版场景文字也替换（字幕/标题） |

---

## 14 · 实战案例 · Football Match Simulator

| 阶段 | 产出 | 大小 | 备注 |
|---|---|---|---|
| Agent 1 ProjectAnalyzer | `football-cyber-football/` + `football-english-edition/` | — | 通过读 `llm_agent.py` prompt 区分中英版本 |
| OpenDesign | `football-video.zip` → `football-video-html/` | — | 极简 B&W + 金；`[agent_NN] decided.` 标记动效 |
| Agent 2 录屏 | `english_match_TEST5min_1440p_20260506_134203.webm` | — | 1440p / 5min / 头尾 5s 不可用 |
| Agent 3 v1 | `promo_concept_v1.mp4` | ~21 MB | 录屏占比 15% |
| Agent 3 v2 | `promo_concept_v2.mp4` | 25 MB | 录屏占比 46%；踩 FFmpeg 中文路径坑 → junction |
| Agent 3 v3 | `promo_concept_v3.mp4` | 31 MB | 大字/小字背景规则 + crossfade 修复；**User 拍板通过** |
| Agent 4 v4 | `promo_concept_v4.mp4` | 30.5 MB | 节拍脚手架 BGM（140 BPM hype trap） |
| Agent 5 v5 | `promo_concept_v5.mp4` | 29.1 MB | BGM + Outro 配音（`en-US-EricNeural`，31.5s 入） |
| Agent 5 v6 | `promo_concept_v6.mp4` | 66 MB | 当前最新版本（2026-05-06 18:47） |
| **待完成** | 全轨配音终片 | — | 中英双版本，所有场景补齐旁白 |

---

*Last updated 2026-05-07. 每个新项目完成后继续迭代此文档。*

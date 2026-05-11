"""Shared dataclasses. Mirrors WORKFLOW.md §8."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union


# ── Top-level input ─────────────────────────────────────────
@dataclass
class ProjectInput:
    repo_url: str
    branch: Optional[str] = None
    style_keywords: list[str] = field(default_factory=list)
    style_reference: Optional[str] = None


# ── Assets ─────────────────────────────────────────────────
@dataclass
class HtmlAsset:
    path: Path
    verified: bool = False


@dataclass
class RecordingAsset:
    path: Path
    resolution: str
    duration: float
    unusable_head: float = 5.0
    unusable_tail: float = 5.0


@dataclass
class VideoAsset:
    path: Path
    duration: float
    version: int
    has_bgm: bool = False
    has_voice: bool = False


@dataclass
class BgmAsset:
    path: Path
    bpm: int
    sourced_by: Literal['scaffold', 'musicgen'] = 'scaffold'


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
    voice: str
    segments: list[VoiceSegment] = field(default_factory=list)


Asset = Union[HtmlAsset, RecordingAsset, VideoAsset, BgmAsset, VoiceAsset]


# ── Edit operations (5 kinds) ─────────────────────────────
@dataclass
class ReplaceBg:
    range: tuple[float, float]
    source_asset: str
    source_range: tuple[float, float]
    op: Literal['replace_bg'] = 'replace_bg'


@dataclass
class ExtendScene:
    scene_id: str
    delta_sec: float
    op: Literal['extend_scene'] = 'extend_scene'


@dataclass
class ChangeClip:
    scene_id: str
    new_start_from: float
    op: Literal['change_clip'] = 'change_clip'


@dataclass
class FixTransition:
    between_scenes: tuple[str, str]
    op: Literal['fix_transition'] = 'fix_transition'


@dataclass
class AdjustDarken:
    scene_id: str
    value: float  # 0..1
    op: Literal['adjust_darken'] = 'adjust_darken'


EditOp = Union[ReplaceBg, ExtendScene, ChangeClip, FixTransition, AdjustDarken]


# ── Pipeline state ─────────────────────────────────────────
Phase = Literal[1, 2, 3, 4, 5]
Gate = Literal[
    'waiting_brief_approval',
    'waiting_html', 'waiting_recording',
    'waiting_video_approval',
    'waiting_bgm_approval',
    'waiting_script_approval',
    'running', 'done',
    'failed',
]


@dataclass
class PipelineState:
    run_id: str
    project: str
    phase: Phase = 1
    gate: Gate = 'running'
    current_version: int = 0
    manifest: dict = field(default_factory=dict)
    last_error: Optional[dict] = None
    repo_url: Optional[str] = None

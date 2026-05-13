"""Deterministic per-phase metrics — scraped from disk artifacts.

Complement to QualityJudge (which is subjective LLM-as-a-Judge): this
module produces objective measurements that need no LLM. Numbers you can
trust exactly:

  - wallclock_s        — phase duration from timer files
  - artifact_size      — bytes / words / lines / scene_count / etc.
  - media_meta         — fps / resolution / duration / has_audio via ffprobe
  - voice_timing_delta — TTS clip length vs cue.t_end - cue.t_start
  - retry_count        — failed LLM attempts from errors.jsonl
  - escalation_count   — number of times ErrorAgent was invoked

All values are appended to <run_dir>/metrics.jsonl alongside scores.jsonl.
Viewed on the /scores page. The metric record is also pushed as a
deterministic-source row into scores.jsonl so existing UI grouping shows
it inline next to LLM judge scores.
"""
from __future__ import annotations
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_WORD_RE = re.compile(r"\S+")


def _read_timer(run_dir: Path, phase: int) -> Optional[float]:
    """Wallclock seconds from phase{N}_started.txt → phase{N}_finished.txt
    (or now() if not finished). Returns None if no started.txt."""
    started = run_dir / f"phase{phase}_started.txt"
    finished = run_dir / f"phase{phase}_finished.txt"
    if not started.exists():
        return None
    try:
        t0 = float(started.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    if finished.exists():
        try:
            t1 = float(finished.read_text(encoding="utf-8").strip())
            return round(t1 - t0, 1)
        except Exception:
            pass
    return round(time.time() - t0, 1)


def _ffprobe_meta(mp4_path: Path) -> Optional[dict]:
    """Return {duration_s, width, height, fps, has_audio, size_mb} or None."""
    if not mp4_path.exists():
        return None
    try:
        from .ffbin import ffprobe
        from .shell import run as shell_run
        r = shell_run(
            [ffprobe(), "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(mp4_path)],
            check=False, timeout=15,
        )
        if r.exit_code != 0 or not r.stdout:
            return None
        data = json.loads(r.stdout)
    except Exception:
        return None

    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    if not v:
        return None
    try:
        num, den = (v.get("r_frame_rate") or "0/1").split("/")
        fps = round(int(num) / max(1, int(den)), 2)
    except Exception:
        fps = None
    return {
        "duration_s": round(float(fmt.get("duration", 0)), 2),
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "fps": fps,
        "has_audio": a is not None,
        "size_mb": round(int(fmt.get("size", 0)) / 1024 / 1024, 2),
    }


def _count_retries(run_dir: Path, phase: int) -> dict:
    """Read errors.jsonl, count attempts + escalations attributable to this phase."""
    p = run_dir / "errors.jsonl"
    out = {"retries": 0, "escalations": 0, "exhausted": 0}
    if not p.exists():
        return out
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            # Phase attribution: best-effort match via context_hint or step_label
            agent = rec.get("agent", "")
            step = rec.get("step_label", "")
            ctx = rec.get("context_hint") or {}
            rec_phase = ctx.get("phase") if isinstance(ctx, dict) else None
            if rec_phase is None:
                # Heuristic agent-name → phase
                if "ProjectAnalyzer" in agent or phase == 1 and "analyzer" in agent.lower():
                    rec_phase = 1
                elif "SetupRunner" in agent or "planner" in agent.lower():
                    rec_phase = 2
                elif "RemotionComposer" in agent or "phase3" in agent:
                    rec_phase = 3
                elif "BGM" in agent or "phase4" in agent:
                    rec_phase = 4
                elif "Voice" in agent or "phase5" in agent:
                    rec_phase = 5
            if rec_phase != phase:
                continue
            out["retries"] += 1
            if rec.get("escalated"):
                out["escalations"] += 1
            if rec.get("attempt") == rec.get("max_attempts"):
                out["exhausted"] += 1
    except Exception:
        pass
    return out


def _brief_stats(brief_md: Path) -> dict:
    """Word / line / section count for project_brief.md."""
    if not brief_md.exists():
        return {}
    try:
        text = brief_md.read_text(encoding="utf-8")
    except Exception:
        return {}
    return {
        "word_count": len(_WORD_RE.findall(text)),
        "line_count": text.count("\n") + 1,
        "section_count": sum(1 for ln in text.splitlines() if ln.startswith("## ")),
        "size_kb": round(len(text.encode("utf-8")) / 1024, 1),
    }


def _setup_plan_stats(plan_path: Path) -> dict:
    """Tool count / config_writes count / services count / user_secrets count."""
    if not plan_path.exists():
        return {}
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "manual_prereqs_count": len(plan.get("manual_prereqs") or []),
        "install_commands_count": len(plan.get("install_commands") or []),
        "config_writes_count": len(plan.get("config_writes") or []),
        "services_count": len(plan.get("services") or []),
        "user_secrets_needed_count": len(plan.get("user_secrets_needed") or []),
    }


def _setup_exec_stats(exec_path: Path) -> dict:
    """Healthy-services ratio + exec status."""
    if not exec_path.exists():
        return {}
    try:
        exc = json.loads(exec_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    services = exc.get("services") or []
    healthy = sum(1 for s in services if s.get("status") == "healthy")
    return {
        "status": exc.get("status"),
        "services_healthy": healthy,
        "services_total": len(services),
    }


def _cutting_plan_stats(plan_path: Path) -> dict:
    """Scene count + sum of scene durations + asset-mix breakdown."""
    if not plan_path.exists():
        return {}
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    scenes = plan.get("scenes") or []
    total = sum(float(s.get("duration_s") or 0) for s in scenes)
    asset_mix: dict[str, int] = {}
    for s in scenes:
        bg = s.get("background") or {}
        kind = bg.get("type") or bg.get("source_path", "").split("/")[0] or "unknown"
        asset_mix[kind] = asset_mix.get(kind, 0) + 1
    return {
        "scene_count": len(scenes),
        "total_duration_s": round(total, 1),
        "duration_target_30_45_s": 30 <= total <= 45,
        "asset_mix": asset_mix,
    }


def _voiceover_timing_delta(run_dir: Path) -> Optional[dict]:
    """For each (cue, generated TTS wav), compute (actual_wav_s - window_s).

    Positive = TTS overruns its window (will be ducked/cut off);
    negative = TTS underruns (gap). |delta| > 0.3s is usually noticeable.
    """
    voice_dir = run_dir / "voice"
    bilingual = voice_dir / "voiceover_script_bilingual.json"
    if not bilingual.exists():
        return None
    try:
        data = json.loads(bilingual.read_text(encoding="utf-8"))
    except Exception:
        return None

    out: dict = {"per_lang": {}}
    try:
        from .ffbin import ffprobe
        from .shell import run as shell_run
    except Exception:
        return None

    for lang in ("zh-CN", "en-US"):
        seg_dir = voice_dir / f"per_segment_{lang}"
        per_path = voice_dir / f"voiceover_script_{lang}.json"
        if not (seg_dir.exists() and per_path.exists()):
            continue
        try:
            cues = json.loads(per_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        deltas: list[float] = []
        for i, cue in enumerate(cues):
            t_start = float(cue.get("t_start") or 0)
            t_end = float(cue.get("t_end") or t_start)
            window_s = t_end - t_start
            wav = seg_dir / f"{i:03d}.mp3"  # tts_edge writes mp3 per segment
            if not wav.exists():
                wav = seg_dir / f"{i:03d}.wav"
            if not wav.exists():
                continue
            try:
                r = shell_run(
                    [ffprobe(), "-v", "quiet", "-print_format", "json",
                     "-show_format", str(wav)],
                    check=False, timeout=5,
                )
                if r.exit_code == 0 and r.stdout:
                    fmt = json.loads(r.stdout).get("format") or {}
                    actual_s = float(fmt.get("duration", 0))
                    deltas.append(round(actual_s - window_s, 2))
            except Exception:
                continue
        if deltas:
            abs_deltas = [abs(d) for d in deltas]
            out["per_lang"][lang] = {
                "cue_count": len(deltas),
                "max_overrun_s": round(max(deltas), 2),
                "max_underrun_s": round(min(deltas), 2),
                "avg_abs_delta_s": round(sum(abs_deltas) / len(abs_deltas), 2),
                "cues_off_by_300ms_plus": sum(1 for d in abs_deltas if d > 0.3),
            }
    return out if out["per_lang"] else None


# ────────────────────────────────────────────────────────────
# Public collectors — one per phase. Each returns dict (possibly empty).
# ────────────────────────────────────────────────────────────

def collect_phase1(run_dir: Path) -> dict:
    return {
        "phase": "phase1_metrics",
        "wallclock_s": _read_timer(run_dir, 1),
        "brief": _brief_stats(run_dir / "project_brief.md"),
        **_count_retries(run_dir, 1),
    }


def collect_phase2(run_dir: Path) -> dict:
    rec_test = run_dir / "recordings" / "test.mp4"
    return {
        "phase": "phase2_metrics",
        "wallclock_s": _read_timer(run_dir, 2),
        "setup_plan": _setup_plan_stats(run_dir / "setup_plan.json"),
        "setup_exec": _setup_exec_stats(run_dir / "setup_exec.json"),
        "test_recording": _ffprobe_meta(rec_test),
        **_count_retries(run_dir, 2),
    }


def collect_phase3(run_dir: Path) -> dict:
    return {
        "phase": "phase3_metrics",
        "wallclock_s": _read_timer(run_dir, 3),
        "cutting_plan": _cutting_plan_stats(run_dir / "cutting_plan.json"),
        "v1_video": _ffprobe_meta(run_dir / "outputs" / "v1.mp4"),
        **_count_retries(run_dir, 3),
    }


def collect_phase4(run_dir: Path) -> dict:
    return {
        "phase": "phase4_metrics",
        "wallclock_s": _read_timer(run_dir, 4),
        "v1_bgm_video": _ffprobe_meta(run_dir / "outputs" / "v1_bgm_final.mp4"),
        **_count_retries(run_dir, 4),
    }


def collect_phase5(run_dir: Path) -> dict:
    return {
        "phase": "phase5_metrics",
        "wallclock_s": _read_timer(run_dir, 5),
        "final_zh": _ffprobe_meta(run_dir / "outputs" / "final_zh-CN.mp4"),
        "final_en": _ffprobe_meta(run_dir / "outputs" / "final_en-US.mp4"),
        "voiceover_timing": _voiceover_timing_delta(run_dir),
        **_count_retries(run_dir, 5),
    }


_COLLECTORS = {
    1: collect_phase1, 2: collect_phase2, 3: collect_phase3,
    4: collect_phase4, 5: collect_phase5,
}


def collect_and_save(run_dir: Path, phase: int) -> Optional[dict]:
    """Run the phase's collector and append to metrics.jsonl + scores.jsonl.

    Returns the metric record, or None if phase out of range.
    """
    fn = _COLLECTORS.get(phase)
    if fn is None:
        return None
    rec = fn(run_dir)
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    rec["source"] = "deterministic_metric"

    # Write to dedicated metrics.jsonl
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ALSO append to scores.jsonl so /scores page can render it inline.
    scores_path = run_dir / "scores.jsonl"
    with scores_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return rec


def read_metrics(run_dir: Path) -> list[dict]:
    """Newest-first read of metrics.jsonl."""
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for ln in reversed(p.read_text(encoding="utf-8").splitlines()):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

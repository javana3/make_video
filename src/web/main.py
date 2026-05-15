"""Web UI for the pipeline.

FastAPI + Jinja2 + HTMX + SSE. One process serves multiple runs (read-only
overview), with one Agent execution at a time (iterate is queued/rejected if
busy). See WORKFLOW.md §11 for the full design.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt

from ..pipeline import Pipeline
from ..tools.dotenv import load_dotenv
from ..observability.audit import set_run_context

load_dotenv()

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))
md_renderer = MarkdownIt("commonmark", {"breaks": True, "linkify": True}).enable("table")

app = FastAPI(title="Promo Video Pipeline UI")


def _fire_quality_judge(pipe: Pipeline, phase: str) -> None:
    """Fire QualityJudge + deterministic metrics for a phase artifact.

    Two things happen async after each phase artifact lands:
      1. LLM-as-a-Judge (subjective) — score_phase(phase, run_dir)
      2. Deterministic scraper — collect_and_save(run_dir, phase_int)

    Both write into scores.jsonl + their own files (metrics.jsonl for the
    scraper). The scraper is cheap and synchronous; the LLM judge is the
    slow part. Phase 4 has no LLM judge artifact (BGM is deterministic
    generation), but still gets metrics collection.

    Failures swallowed — scoring/metrics are auxiliary.
    """
    _PHASE_NAME_TO_INT = {
        "brief": 1, "setup_plan": 2, "cutting_plan": 3, "voiceover_script": 5,
        # for cross_phase: trigger after the LAST phase artifact has been judged
        "cross_phase": 5,
    }
    phase_int = _PHASE_NAME_TO_INT.get(phase)

    async def _go():
        # 1) LLM judge (slow)
        try:
            from ..agents.quality_judge import score_phase
            await asyncio.to_thread(score_phase, phase, pipe.run_dir)
        except Exception as e:
            pipe.log.exception(f"quality_judge[{phase}] failed: {e}")
        # 2) Deterministic metrics (cheap)
        if phase_int is not None:
            try:
                from ..tools.run_metrics import collect_and_save
                await asyncio.to_thread(collect_and_save, pipe.run_dir, phase_int)
            except Exception as e:
                pipe.log.exception(f"run_metrics[{phase_int}] failed: {e}")

    try:
        asyncio.create_task(_go())
    except RuntimeError:
        pipe.log.warning(f"quality_judge[{phase}] could not schedule (no loop)")


def _fire_opendesign_bootstrap(pipe: Pipeline) -> None:
    """Auto-fire OpenDesign pipeline: bootstrap → first_turn → adopt → critic loop.

    Called when Phase 2 recording is accepted. The full chain happens in the
    background; UI polls opendesign/state.json (bootstrap done?) + opendesign
    session history (first_turn done?) + od_critic_progress.json (critic running?).

    Idempotent across each stage: if the session already exists, skip bootstrap.
    If the first turn already ran, skip it. If the artifact already exists, skip
    adopt. Critic always runs the full soft_max_iterations unless user takes over.
    """
    async def _go():
        try:
            from ..agents.opendesigner import bootstrap, load_session, iterate_stream, adopt
            from ..tools.opendesign_lifecycle import ensure_daemon
            from ..agents.od_critic import run_critic_loop

            brief_path = pipe.run_dir / "project_brief.md"
            if not brief_path.exists():
                pipe.log.warning("opendesign pipeline skipped: no project_brief.md")
                return

            # 1. bootstrap (create session if absent)
            existing = await asyncio.to_thread(load_session, pipe.run_dir)
            if existing is None:
                endpoint = await asyncio.to_thread(ensure_daemon)
                brief = brief_path.read_text(encoding="utf-8")
                await asyncio.to_thread(
                    bootstrap, pipe.run_dir, endpoint, brief,
                    f"{pipe.project}-promo",
                )
                pipe.log.info("opendesign: bootstrap done")
            else:
                pipe.log.info("opendesign: bootstrap session already exists, skipping")

            # 2. first turn (only if no history)
            sess = await asyncio.to_thread(load_session, pipe.run_dir)
            if sess is None:
                pipe.log.error("opendesign: session disappeared after bootstrap; abort")
                return
            if not sess.history:
                pipe.log.info("opendesign: firing first_turn (initial_prompt)")
                def _drain_first_turn():
                    n = 0
                    for _evt in iterate_stream(pipe.run_dir, timeout_s=1800.0):
                        n += 1
                    return n
                n_evt = await asyncio.to_thread(_drain_first_turn)
                pipe.log.info(f"opendesign: first_turn done, {n_evt} SSE events")
            else:
                pipe.log.info(f"opendesign: history already has {len(sess.history)} turn(s), skipping first_turn")

            # 3. adopt (best-effort; critic loop also re-adopts each iter)
            try:
                await asyncio.to_thread(adopt, pipe.run_dir, "hero")
                pipe.log.info("opendesign: initial adopt done")
            except Exception as e:
                pipe.log.warning(f"opendesign: initial adopt failed (critic loop will retry): {e}")

            # 4. critic loop — fire only if not already done / running.
            from ..agents.od_critic import read_progress
            prev = await asyncio.to_thread(read_progress, pipe.run_dir)
            if prev and prev.get("stage") == "done":
                pipe.log.info("opendesign: critic loop already completed for this run")
                return
            pipe.log.info("opendesign: firing critic loop (soft_max=5)")
            await asyncio.to_thread(run_critic_loop, pipe.run_dir, 5, None)
            pipe.log.info("opendesign: critic loop finished")

        except Exception as e:
            pipe.log.exception(f"opendesign pipeline failed: {e}")
            pipe.bus.emit("asset_failed", agent="agent6_opendesigner",
                           error=str(e), error_type=type(e).__name__)
    try:
        asyncio.create_task(_go())
    except RuntimeError:
        pipe.log.warning("opendesign pipeline could not schedule (no loop)")


def _fire_cross_phase_judge(pipe: Pipeline) -> None:
    """Fire the cross-phase consistency judge after all 4 source artifacts exist.

    Reads brief / setup_plan / cutting_plan / voiceover at once and rates
    info-flow drift across the pipeline. The high-value eval that per-phase
    rubrics by design can't see.
    """
    async def _go():
        try:
            from ..agents.quality_judge import score_cross_phase_consistency
            await asyncio.to_thread(score_cross_phase_consistency, pipe.run_dir)
        except Exception as e:
            pipe.log.exception(f"cross_phase judge failed: {e}")
    try:
        asyncio.create_task(_go())
    except RuntimeError:
        pass


def _fire_metrics_only(pipe: Pipeline, phase_int: int) -> None:
    """Fire deterministic metrics collection without a paired LLM judge.

    Used for phases where there's no LLM artifact to evaluate (Phase 4 BGM
    is deterministic) but the metrics scraper still produces useful numbers
    (ffprobe for v1_bgm_final.mp4, wallclock, retry count).
    """
    async def _go():
        try:
            from ..tools.run_metrics import collect_and_save
            await asyncio.to_thread(collect_and_save, pipe.run_dir, phase_int)
        except Exception as e:
            pipe.log.exception(f"run_metrics[{phase_int}] failed: {e}")
    try:
        asyncio.create_task(_go())
    except RuntimeError:
        pass

# Self-host Tailwind / htmx / htmx-sse from src/web/static/ — the project is
# meant to run offline on a local box, so we cannot rely on cdn.tailwindcss.com
# / unpkg.com. Files committed to repo to remove the network dependency.
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.middleware("http")
async def normalize_empty_run_ids(request: Request, call_next):
    """`/runs///observability` (empty project + run_id from manual URL or stale
    template) should land on the global Trace dashboard instead of bare 404.

    Catches any `/runs/.../observability|trace|logs|opendesign` where one or
    more path segments are empty (consecutive slashes) and redirects to the
    global aggregator (or home if no obvious aggregator).
    """
    from fastapi.responses import RedirectResponse
    raw = request.url.path
    if "//" in raw:
        # Collapse consecutive slashes to detect malformed IDs
        compact = "/" + "/".join(seg for seg in raw.split("/") if seg)
        # If after compacting we lose path segments, project/run_id were empty
        if compact != raw:
            tail = compact.rsplit("/", 1)[-1]
            if tail in ("observability", "trace", "traces", "logs"):
                return RedirectResponse(url="/observability", status_code=302)
            if tail in ("opendesign", "opendesign/preview", "opendesign/artifacts"):
                return RedirectResponse(url="/", status_code=302)
            # Generic: if path looks like /runs/.../<X> with empty IDs, go home
            if compact.startswith("/runs/"):
                return RedirectResponse(url="/", status_code=302)
    return await call_next(request)

WORKSPACE_ROOT = Path.cwd() / "workspace"


# ────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────

class _Registry:
    """Holds live Pipeline instances + agent execution state."""
    def __init__(self) -> None:
        self._pipelines: dict[str, Pipeline] = {}
        self._running_agents: dict[str, str] = {}  # run_id → agent_name
        self._lock = asyncio.Lock()

    def get_or_load(self, project: str, run_id: str) -> Pipeline:
        key = f"{project}:{run_id}"
        if key not in self._pipelines:
            self._pipelines[key] = Pipeline(
                project=project, run_id=run_id,
                workspace_root=WORKSPACE_ROOT,
                launch_observability_ui=False,
            )
        return self._pipelines[key]

    def is_running(self, run_id: str) -> bool:
        return run_id in self._running_agents

    async def mark_running(self, run_id: str, agent: str) -> None:
        async with self._lock:
            self._running_agents[run_id] = agent

    async def mark_done(self, run_id: str) -> None:
        async with self._lock:
            self._running_agents.pop(run_id, None)


REGISTRY = _Registry()


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _list_runs() -> list[dict]:
    out: list[dict] = []
    if not WORKSPACE_ROOT.exists():
        return out
    for project_dir in WORKSPACE_ROOT.iterdir():
        if not project_dir.is_dir():
            continue
        runs_dir = project_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in runs_dir.iterdir():
            sf = run_dir / "state.json"
            if not sf.exists():
                continue
            try:
                state = json.loads(sf.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "project": project_dir.name,
                "run_id": run_dir.name,
                "phase": state.get("phase"),
                "gate": state.get("gate"),
                "manifest_count": len(state.get("manifest", {})),
                "mtime": sf.stat().st_mtime,
            })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _read_events(run_dir: Path, since_line: int = 0) -> tuple[list[dict], int]:
    f = run_dir / "events.jsonl"
    if not f.exists():
        return [], since_line
    lines = f.read_text(encoding="utf-8").splitlines()
    new = []
    for ln in lines[since_line:]:
        if ln.strip():
            try:
                new.append(json.loads(ln))
            except Exception:
                pass
    return new, len(lines)


def _load_brief_versions(briefs_dir: Path) -> dict:
    """Return {standard: {md, html, meta}, deep: {...}} for whichever exists."""
    out = {}
    for mode in ("standard", "deep"):
        md_path = briefs_dir / f"{mode}.md"
        meta_path = briefs_dir / f"{mode}_meta.json"
        if not md_path.exists():
            continue
        md = md_path.read_text(encoding="utf-8")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        out[mode] = {
            "md": md,
            "html": md_renderer.render(md),
            "meta": meta,
        }
    return out


def _phase_label(phase: int) -> str:
    return {
        1: "Phase 1 · ProjectAnalyzer",
        2: "Phase 2 · SetupRunner + HTML",
        3: "Phase 3 · RemotionComposer",
        4: "Phase 4 · BGMComposer",
        5: "Phase 5 · VoiceOver",
    }.get(phase, f"Phase {phase}")


# ────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(request, "index.html", {
        "runs": _list_runs(),
    })


_PROJECT_NAME_RE = __import__("re").compile(r"[^A-Za-z0-9._-]+")


def _safe_project_name(raw: str) -> str:
    name = _PROJECT_NAME_RE.sub("-", raw.strip().strip("/"))
    name = name.strip("-")[:64]
    return name or "run"


@app.post("/runs/new", response_class=HTMLResponse)
async def runs_new(request: Request,
                    mode: str = Form(...),
                    repo_url: str = Form(""),
                    local_path: str = Form(""),
                    project_name: str = Form("")):
    """Create a new run from either GitHub URL or local path.

    mode = "url" → git clone --depth=1
    mode = "local" → copytree (skips .git, .venv, node_modules, workspace)
    Then auto-trigger Phase 1 (Agent 1 ProjectAnalyzer) in background.
    """
    import shutil
    from ..tools.shell import run as shell_run

    if mode == "url":
        if not repo_url.strip():
            raise HTTPException(400, "repo_url required for mode=url")
        url = repo_url.strip()
        derived = url.rstrip("/").split("/")[-1].removesuffix(".git")
    elif mode == "local":
        if not local_path.strip():
            raise HTTPException(400, "local_path required for mode=local")
        src_path = Path(local_path.strip()).expanduser().resolve()
        if not src_path.exists() or not src_path.is_dir():
            raise HTTPException(400, f"local_path does not exist or is not a directory: {src_path}")
        derived = src_path.name
    else:
        raise HTTPException(400, "mode must be 'url' or 'local'")

    project = _safe_project_name(project_name or derived)
    # Generate run_id ourselves THEN cache via REGISTRY so the singleton instance
    # is shared with later route handlers (otherwise each get_or_load creates a
    # separate Pipeline whose in-memory state can drift from disk).
    import uuid
    run_id = uuid.uuid4().hex[:8]
    pipe = REGISTRY.get_or_load(project, run_id)
    pipe.transition(phase=1, gate="running")
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)

    # Persist source-of-truth so retry can re-clone or re-copy without user re-input.
    if mode == "url":
        pipe.state.repo_url = url
        pipe.save()

    # Defer the clone/copy to a background task so the user lands on the run
    # page IMMEDIATELY and watches progress there (instead of staring at a
    # spinning form for minutes during a slow clone).
    if mode == "url":
        asyncio.create_task(_run_clone_and_analyze(pipe, url=url))
    else:
        asyncio.create_task(_run_localcopy_and_analyze(pipe, src_path=src_path))

    target = f"/runs/{project}/{pipe.run_id}"
    return HTMLResponse(
        f'<div class="text-emerald-400 text-sm p-4">'
        f'✅ Run created: <code>{project}/{pipe.run_id}</code> · '
        f'{"clone" if mode == "url" else "copy"} 启动中...<br>'
        f'<a class="underline" href="{target}">→ 跳转到 run 页面查看进度</a></div>',
        headers={"HX-Redirect": target, "Location": target},
    )


async def _run_clone_and_analyze(pipe: Pipeline, url: str) -> None:
    """Phase 1 entry for URL mode: stream-clone with progress, then analyze."""
    await REGISTRY.mark_running(pipe.run_id, "phase1-clone")
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    repo_dir = pipe.run_dir / "repo"
    progress_path = pipe.run_dir / "clone_progress.json"
    try:
        from ..tools.git_clone import clone_with_progress
        pipe.log.info(f"clone (streaming) {url} → {repo_dir}")
        exit_code, tail = await asyncio.to_thread(
            clone_with_progress, url, repo_dir, progress_path, 600,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"git clone exit {exit_code}. tail:\n{tail[-1500:]}"
            )
        pipe.record_asset("repo", repo_dir, verified=True, url=url, source="git_clone")
    except Exception as e:
        pipe.log.exception(f"clone failed: {e}")
        pipe.bus.emit("asset_failed", agent="pipeline",
                       name="repo", error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=1, agent="git-clone",
                           error_type=type(e).__name__, error_text=str(e))
        await REGISTRY.mark_done(pipe.run_id)
        return
    await REGISTRY.mark_done(pipe.run_id)
    # Hand off to analyzer (it has its own mark_running/mark_done).
    asyncio.create_task(_run_analyzer_async(pipe, repo_url_or_path=url))


async def _run_localcopy_and_analyze(pipe: Pipeline, src_path: Path) -> None:
    """Phase 1 entry for local mode: copytree, then analyze."""
    import shutil
    await REGISTRY.mark_running(pipe.run_id, "phase1-copy")
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    repo_dir = pipe.run_dir / "repo"
    progress_path = pipe.run_dir / "clone_progress.json"
    try:
        # Mark as starting so UI shows a copying indicator
        import json as _json, time as _time
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(_json.dumps({
            "phase": "copying", "source_path": str(src_path),
            "ts": _time.time(), "elapsed": 0,
        }), encoding="utf-8")
        pipe.log.info(f"copy {src_path} → {repo_dir}")
        ignore = shutil.ignore_patterns(
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".pytest_cache", "build", "dist", ".tox", ".mypy_cache",
            ".idea", ".vscode", "*.pyc", "*.pyo",
        )
        t0 = _time.time()
        await asyncio.to_thread(shutil.copytree, str(src_path), str(repo_dir),
                                  ignore=ignore, dirs_exist_ok=False)
        elapsed = round(_time.time() - t0, 1)
        progress_path.write_text(_json.dumps({
            "phase": "done", "source_path": str(src_path),
            "ts": _time.time(), "elapsed": elapsed,
        }), encoding="utf-8")
        pipe.record_asset("repo", repo_dir, verified=True,
                           source="local_copy", source_path=str(src_path))
    except Exception as e:
        pipe.log.exception(f"local copy failed: {e}")
        pipe.bus.emit("asset_failed", agent="pipeline",
                       name="repo", error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=1, agent="local-copy",
                           error_type=type(e).__name__, error_text=str(e))
        await REGISTRY.mark_done(pipe.run_id)
        return
    await REGISTRY.mark_done(pipe.run_id)
    asyncio.create_task(_run_analyzer_async(pipe, repo_url_or_path=str(src_path)))


async def _retry_clone_and_analyze(pipe: Pipeline) -> None:
    """Retry the Phase-1 clone (when initial clone exit ≠ 0), then run analyzer."""
    import shutil
    url = pipe.state.repo_url
    if not url:
        pipe.record_error(phase=1, agent="git-clone",
                           error_type="RuntimeError",
                           error_text="no repo_url in state — cannot retry clone (delete this run and create a new one)")
        return
    # Wipe any partial repo + stale progress so the new attempt starts clean.
    repo_dir = pipe.run_dir / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    progress_path = pipe.run_dir / "clone_progress.json"
    if progress_path.exists():
        progress_path.unlink()
    # Re-use the same streaming path used by the initial run.
    await _run_clone_and_analyze(pipe, url=url)


async def _run_analyzer_async(pipe: Pipeline, repo_url_or_path: str) -> None:
    """Phase 1: invoke Agent 1 ProjectAnalyzer."""
    await REGISTRY.mark_running(pipe.run_id, "phase1-analyze")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..agents.project_analyzer import run_project_analyzer
        repo_dir = pipe.run_dir / "repo"
        brief_path = pipe.run_dir / "project_brief.md"
        progress_path = pipe.run_dir / "progress.json"
        await asyncio.to_thread(
            run_project_analyzer,
            repo_dir=repo_dir,
            repo_url=repo_url_or_path,
            output_path=brief_path,
            mode="standard",
            progress_path=progress_path,
        )
        pipe.record_asset("project_brief", brief_path, verified=True)
        _fire_quality_judge(pipe, "brief")
        pipe.transition(phase=1, gate="waiting_brief_approval")
    except Exception as e:
        pipe.log.exception(f"analyzer failed: {e}")
        pipe.bus.emit("asset_failed", agent="Agent 1 ProjectAnalyzer",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=1, agent="Agent 1 ProjectAnalyzer",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


@app.get("/runs/{project}/{run_id}", response_class=HTMLResponse)
async def view_run(project: str, run_id: str, request: Request):
    pipe = REGISTRY.get_or_load(project, run_id)
    state = pipe.state

    phase_ctx: dict = {}
    if state.phase == 1:
        briefs_dir = pipe.run_dir / "briefs"
        phase_ctx["versions"] = _load_brief_versions(briefs_dir)
        # Distinguish clone phase from analyzer phase. Clone is "done" once
        # `repo/.git/` exists (git's atomic finalization marker).
        phase_ctx["clone_done"] = (pipe.run_dir / "repo" / ".git").exists()
        phase_ctx["clone_progress_exists"] = (pipe.run_dir / "clone_progress.json").exists()

    # Phase 3/4/5 status (file existence + sizes)
    out_dir = pipe.run_dir / "outputs"
    v1 = out_dir / "v1.mp4"
    v1_bgm = out_dir / "v1_bgm_final.mp4"
    final_zh = out_dir / "final_zh-CN.mp4"
    final_en = out_dir / "final_en-US.mp4"
    is_running = REGISTRY.is_running(run_id)
    phase3_state = {
        "v1_exists": v1.exists(),
        "v1_size_mb": round(v1.stat().st_size / 1024 / 1024, 2) if v1.exists() else 0,
        "is_running": is_running,
    }
    phase4_state = {
        "v1_bgm_exists": v1_bgm.exists(),
        "v1bgm_size_mb": round(v1_bgm.stat().st_size / 1024 / 1024, 2) if v1_bgm.exists() else 0,
        "is_running": is_running,
    }
    phase5_state = {
        "final_zh_exists": final_zh.exists(),
        "final_en_exists": final_en.exists(),
        "zh_size_mb": round(final_zh.stat().st_size / 1024 / 1024, 2) if final_zh.exists() else 0,
        "en_size_mb": round(final_en.stat().st_size / 1024 / 1024, 2) if final_en.exists() else 0,
        "is_running": is_running,
    }

    return TEMPLATES.TemplateResponse(request, "run.html", {
        "project": project,
        "run_id": run_id,
        "state": state,
        "phase_ctx": phase_ctx,
        "phase_label": _phase_label(state.phase),
        "phoenix_url": "http://localhost:6006",
        "agent_running": REGISTRY.is_running(run_id),
        "run_dir": str(pipe.run_dir),
        "phase3_state": phase3_state,
        "phase4_state": phase4_state,
        "phase5_state": phase5_state,
    })


@app.get("/runs/{project}/{run_id}/clone_panel", response_class=HTMLResponse)
async def clone_panel(project: str, run_id: str, request: Request):
    """Live clone-progress panel — polled by Phase 1 every 2s until clone done.

    Once clone is done (repo/.git materialized), we emit `HX-Refresh: true`
    so the parent page re-renders with `clone_done=True` and switches to
    the briefs view automatically — no manual "next step" button needed.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    run_dir = pipe.run_dir
    prog_path = run_dir / "clone_progress.json"
    progress = None
    if prog_path.exists():
        try:
            progress = json.loads(prog_path.read_text(encoding="utf-8"))
        except Exception:
            progress = {"phase": "unknown", "last_line": "(failed to read progress file)"}

    headers = {}
    # Trigger a full page refresh once the clone has materially completed.
    # We check both the progress phase AND the actual .git directory to
    # avoid a premature refresh if the JSON was updated but git is still
    # finalizing pack files.
    clone_done = (run_dir / "repo" / ".git").exists()
    if clone_done:
        headers["HX-Refresh"] = "true"

    return TEMPLATES.TemplateResponse(request, "_phase_1_cloning.html", {
        "project": project, "run_id": run_id,
        "progress": progress,
        "repo_url": pipe.state.repo_url,
    }, headers=headers)


@app.get("/runs/{project}/{run_id}/log_tail", response_class=HTMLResponse)
async def log_tail(project: str, run_id: str, request: Request,
                    n: int = 40, filter_agent: str = ""):
    """Render the last N lines of pipeline.jsonl as readable HTML.

    Used as an HTMX-polled component inside the cloning panel. Lines are
    parsed from loguru JSONL and rendered with level color + monospace.
    `filter_agent` (optional) keeps only entries matching `extra.agent`.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    log_path = pipe.run_dir / "logs" / "pipeline.jsonl"
    n = max(1, min(int(n or 40), 200))
    rows: list[dict] = []
    if log_path.exists():
        try:
            # Read tail efficiently: read whole file (it's bounded to one run).
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for ln in lines[-n * 3:]:  # over-fetch in case of filter
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                r = rec.get("record", {})
                agent = r.get("extra", {}).get("agent", "")
                if filter_agent and agent != filter_agent:
                    continue
                rows.append({
                    "ts": r.get("time", {}).get("repr", "")[11:19],
                    "level": r.get("level", {}).get("name", "INFO"),
                    "agent": agent,
                    "message": (r.get("message") or rec.get("text") or "").strip(),
                })
        except Exception:
            pass
    rows = rows[-n:]
    return TEMPLATES.TemplateResponse(request, "_log_tail.html", {
        "rows": rows, "n": n,
    })


@app.get("/runs/{project}/{run_id}/events")
async def stream_events(project: str, run_id: str):
    """SSE: stream new events.jsonl entries as they appear."""
    run_dir = WORKSPACE_ROOT / project / "runs" / run_id
    if not run_dir.exists():
        raise HTTPException(404)

    async def gen():
        line_pos = 0
        while True:
            events, line_pos = _read_events(run_dir, line_pos)
            for evt in events:
                ts = evt.get("ts", "")[:19]
                ev = evt.get("event", "")
                agent = evt.get("agent", "-")
                payload = evt.get("payload", {})
                payload_str = json.dumps(payload, ensure_ascii=False)
                # SSE-safe: escape newlines in data
                html = (
                    f'<div class="text-xs font-mono py-1 border-b border-slate-700 hover:bg-slate-800">'
                    f'<span class="text-slate-500">{ts}</span> '
                    f'<span class="text-emerald-400">{ev}</span> '
                    f'<span class="text-sky-400">{agent}</span> '
                    f'<span class="text-slate-400">{payload_str}</span>'
                    f'</div>'
                )
                # HTMX SSE expects multi-line data: prefix
                data_lines = "\n".join(f"data: {line}" for line in html.split("\n"))
                yield f"event: pipeline_event\n{data_lines}\n\n"
            # Heartbeat to keep connection alive
            yield ": ping\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control": "no-cache"})


@app.get("/runs/{project}/{run_id}/agent_log")
async def stream_agent_log(project: str, run_id: str):
    """SSE: tail the active agent log for live console."""
    run_dir = WORKSPACE_ROOT / project / "runs" / run_id
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        raise HTTPException(404)

    async def gen():
        # Track per-file read positions
        positions: dict[Path, int] = {}
        while True:
            for log_file in sorted(log_dir.glob("*.jsonl")):
                if "pipeline" in log_file.name:
                    continue  # skip pipeline.jsonl, we have events.jsonl for that
                pos = positions.get(log_file, log_file.stat().st_size)
                if not log_file.exists():
                    continue
                size = log_file.stat().st_size
                if size > pos:
                    with log_file.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        for ln in f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                rec = json.loads(ln)
                                ts = rec.get("record", {}).get("time", {}).get("repr", "")[11:19]
                                msg = rec.get("text", ln)[:600]
                                level = rec.get("record", {}).get("level", {}).get("name", "")
                            except Exception:
                                ts, level, msg = "", "", ln[:600]
                            color = {
                                "ERROR": "text-rose-400",
                                "WARNING": "text-amber-400",
                                "INFO": "text-slate-300",
                                "DEBUG": "text-slate-500",
                            }.get(level, "text-slate-300")
                            html = (
                                f'<div class="font-mono text-xs py-0.5 {color}">'
                                f'<span class="text-slate-600">{ts}</span> {msg}'
                                f'</div>'
                            )
                            data_lines = "\n".join(f"data: {l}" for l in html.split("\n"))
                            yield f"event: agent_log\n{data_lines}\n\n"
                    positions[log_file] = size
            yield ": ping\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no",
                                      "Cache-Control": "no-cache"})


@app.post("/runs/{project}/{run_id}/iterate", response_class=HTMLResponse)
async def iterate(project: str, run_id: str,
                  feedback: str = Form(""),
                  mode: str = Form("standard")):
    if REGISTRY.is_running(run_id):
        raise HTTPException(409, "agent already running")
    if mode not in ("standard", "deep"):
        mode = "standard"
    pipe = REGISTRY.get_or_load(project, run_id)
    asyncio.create_task(_run_agent_for_phase(pipe, feedback.strip(), mode=mode))
    label = "深度分析" if mode == "deep" else "标准重写"
    extra = ' 预计 60–120s（多次读源码）' if mode == "deep" else ''
    return HTMLResponse(
        f'<div class="px-4 py-3 bg-amber-500/20 border border-amber-500/40 text-amber-200 rounded">'
        f'⏳ Agent {label} 中…{extra} 看下方 Live Console。'
        f'</div>'
    )


@app.post("/runs/{project}/{run_id}/approve", response_class=HTMLResponse)
async def approve(project: str, run_id: str, version: str = Form("")):
    """Approve the Phase 1 brief and advance to Phase 2.

    PHASE 1 ONLY. Phase 2/3/4 have their own /accept_* routes with
    artifact-existence guards so we can't accidentally skip recording or
    Remotion render. Previously /approve blindly advanced any phase by 1,
    which let users (or a polling double-submit race) jump to Phase 3
    without test.mp4 — caught now with an explicit guard.
    """
    pipe = REGISTRY.get_or_load(project, run_id)

    if pipe.state.phase != 1:
        raise HTTPException(
            400,
            f"/approve is Phase-1 only. Current phase is {pipe.state.phase}; "
            f"use the phase-specific accept route on the run page."
        )

    if version not in ("standard", "deep"):
        raise HTTPException(400, "version must be 'standard' or 'deep'")
    briefs_dir = pipe.run_dir / "briefs"
    chosen_md = briefs_dir / f"{version}.md"
    chosen_meta = briefs_dir / f"{version}_meta.json"
    if not chosen_md.exists():
        raise HTTPException(400, f"briefs/{version}.md does not exist; generate it first")
    # Copy chosen → canonical
    canonical_md = pipe.run_dir / "project_brief.md"
    canonical_meta = pipe.run_dir / "brief_sources.json"
    canonical_md.write_text(chosen_md.read_text(encoding="utf-8"),
                            encoding="utf-8")
    if chosen_meta.exists():
        canonical_meta.write_text(chosen_meta.read_text(encoding="utf-8"),
                                  encoding="utf-8")
    pipe.record_asset("project_brief", canonical_md, verified=True,
                      version=version)

    pipe.gate_pass(pipe.state.gate, version=version or None)
    pipe.transition(phase=2, gate="running")
    target = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        f'<div class="px-4 py-3 bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 rounded">'
        f'✅ Gate passed → phase {pipe.state.phase}（{pipe.state.gate}），跳转中...'
        f'</div>',
        # HX-Redirect tells the htmx client to do a full-page navigation
        # right after the swap — no manual "Reload" click needed.
        headers={"HX-Redirect": target, "Location": target},
    )


def _phase2_state_key(run_dir: Path, run_id: str) -> str:
    """Compute a fingerprint of phase 2 disk state. Changes when something happened."""
    parts: list[str] = []
    for rel in ("setup_plan.json", "setup_exec.json", "progress.json",
                "accepted_window.json", "recordings/test_state.json",
                "recordings/test.mp4"):
        f = run_dir / rel
        if f.exists():
            try:
                parts.append(f"{rel}:{int(f.stat().st_mtime * 1000)}:{f.stat().st_size}")
            except Exception:
                pass
    parts.append(f"running:{REGISTRY.is_running(run_id)}")
    return "|".join(parts)


@app.get("/runs/{project}/{run_id}/phase2_state_stream")
async def phase2_state_stream(project: str, run_id: str):
    """SSE: emit `phase2-changed` whenever the phase-2 disk state fingerprint
    changes. UI listens and refetches the panel only when needed — so user
    interactions (open dropdown, type in input) are NEVER interrupted."""
    run_dir = WORKSPACE_ROOT / project / "runs" / run_id

    async def gen():
        last_key: Optional[str] = None
        while True:
            key = _phase2_state_key(run_dir, run_id)
            if key != last_key:
                yield "event: phase2-changed\ndata: state-changed\n\n"
                last_key = key
            yield ": ping\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/runs/{project}/{run_id}/phase2_panel", response_class=HTMLResponse)
async def phase2_panel(project: str, run_id: str, request: Request):
    """Render the right Phase 2 partial based on file state on disk."""
    run_dir = WORKSPACE_ROOT / project / "runs" / run_id
    plan_path = run_dir / "setup_plan.json"
    exec_path = run_dir / "setup_exec.json"
    progress_path = run_dir / "progress.json"
    rec_dir = run_dir / "recordings"
    test_recording = rec_dir / "test.mp4"
    test_state_path = rec_dir / "test_state.json"
    accepted_path = run_dir / "accepted_window.json"

    progress = None
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = None

    is_running = REGISTRY.is_running(run_id)
    progress_phase = (progress or {}).get("phase", "")
    progress_status = (progress or {}).get("status", "")
    pipe_state = REGISTRY.get_or_load(project, run_id).state

    # 0. OpenDesigner waiting — recording is in hand, HTML hero generation
    # is the next gate before Phase 3 (Remotion compose). Default ON; user
    # must explicitly click "取消 OpenDesign" to skip and jump straight to
    # Phase 3 with only the recording.
    if pipe_state.gate == "waiting_html":
        from ..agents.opendesigner import load_session, list_artifacts
        sess = load_session(run_dir)
        artifacts = {}
        try:
            if sess is not None:
                artifacts = list_artifacts(run_dir)
        except Exception:
            artifacts = {}
        html_asset_dir = run_dir / "html_asset"
        adopted_files = sorted(html_asset_dir.glob("*")) if html_asset_dir.exists() else []
        return TEMPLATES.TemplateResponse(request, "_phase_2_opendesign_pending.html", {
            "project": project, "run_id": run_id,
            "session": sess, "artifacts": artifacts,
            "adopted_count": len(adopted_files),
            "adopted_files": [f.name for f in adopted_files][:10],
        })

    # 1. Agent 2 planner running
    if is_running and progress_phase == "2a-plan" and progress_status == "running":
        return TEMPLATES.TemplateResponse(request, "_phase_2_drafting.html", {
            "project": project, "run_id": run_id, "progress": progress,
        })

    # 2. Plan executor running / done with services up
    exec_state = None
    if exec_path.exists():
        try:
            exec_state = json.loads(exec_path.read_text(encoding="utf-8"))
        except Exception:
            exec_state = {"status": "failed", "error": "setup_exec.json malformed"}

    # 3. Recording in progress (M2b)
    test_state = None
    if test_state_path.exists():
        try:
            test_state = json.loads(test_state_path.read_text(encoding="utf-8"))
        except Exception:
            test_state = None
    if test_state and test_state.get("status") == "recording":
        return TEMPLATES.TemplateResponse(request, "_phase_2b_recording_active.html", {
            "project": project, "run_id": run_id,
            "test_state": test_state,
            "exec_state": exec_state,
        })

    # 4. Test recording exists, awaiting user approval (M2b done, accept to enter M2c)
    if test_recording.exists() and not accepted_path.exists():
        return TEMPLATES.TemplateResponse(request, "_phase_2b_test_done.html", {
            "project": project, "run_id": run_id,
            "test_state": test_state,
            "exec_state": exec_state,
        })

    # 5. Window accepted — ready for formal recording (M2c, placeholder for now)
    if accepted_path.exists():
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        return TEMPLATES.TemplateResponse(request, "_phase_2c_placeholder.html", {
            "project": project, "run_id": run_id,
            "accepted": accepted,
            "exec_state": exec_state,
        })

    # 6. exec done OK — but check if services are actually still alive
    # (e.g. after a reboot the PIDs in services.json are stale).
    if exec_state and exec_state.get("status") == "ok":
        from ..tools.service_manager import ServiceManager
        mgr = ServiceManager(run_dir)
        mgr.refresh_status()
        services = mgr.list()
        dead = [r for r in services if r.status not in ("healthy", "starting")]
        if dead:
            return TEMPLATES.TemplateResponse(request, "_phase_2_services_dead.html", {
                "project": project, "run_id": run_id,
                "services": [
                    {"name": r.name, "status": r.status, "pid": r.pid,
                     "port": r.port, "last_error": r.last_error}
                    for r in services
                ],
            })

        from ..tools.window_enum import list_windows_ranked
        # Build hints from the setup_plan so we can score windows.
        hints: dict = {"project_name": project, "service_urls": [], "run_id": run_id}
        plan_path = run_dir / "setup_plan.json"
        if plan_path.exists():
            try:
                plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                hints["service_urls"] = [
                    s.get("health_url") for s in (plan_data.get("services") or [])
                    if s.get("health_url")
                ]
            except Exception:
                pass
        ranked = list_windows_ranked(hints)
        windows = [
            {"title": w.title, "pid": w.pid,
             "score": w.score, "score_reasons": w.score_reasons}
            for w in ranked
        ]
        # Smart demo state (LLM-planned playwright recording with synced captions)
        demo_script_path = rec_dir / "demo_script.json"
        demo_recording_path = rec_dir / "demo_recording.mp4"
        demo_timings_path = rec_dir / "demo_timings.json"
        demo_script_obj = None
        demo_timings_obj = None
        if demo_script_path.exists():
            try:
                demo_script_obj = json.loads(demo_script_path.read_text(encoding="utf-8"))
            except Exception:
                demo_script_obj = {"steps": [], "_load_error": True}
        if demo_timings_path.exists():
            try:
                demo_timings_obj = json.loads(demo_timings_path.read_text(encoding="utf-8"))
            except Exception:
                demo_timings_obj = None

        demo_state = {
            "run_dir": str(run_dir),
            "demo_script_exists": demo_script_path.exists(),
            "demo_recording_exists": demo_recording_path.exists(),
            "demo_script": demo_script_obj,
            "demo_timings": demo_timings_obj,
        }

        return TEMPLATES.TemplateResponse(request, "_phase_2b_ready.html", {
            "project": project, "run_id": run_id,
            "windows": windows,
            "exec_state": exec_state,
            "state": demo_state,
            **_driver_context(run_dir),
        })

    # 7. exec running / failed — show executing panel + CLI recorder fallback
    if exec_state:
        cli_state = _cli_state(run_dir)
        return TEMPLATES.TemplateResponse(request, "_phase_2_executing.html", {
            "project": project, "run_id": run_id, "exec_state": exec_state,
            "cli_state": cli_state,
            **_driver_context(run_dir),
        })

    # 8. Plan exists, awaiting approval
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            plan = {}
        return TEMPLATES.TemplateResponse(request, "_phase_2_plan_review.html", {
            "project": project, "run_id": run_id, "plan": plan,
        })

    # 9. Initial: no plan
    return TEMPLATES.TemplateResponse(request, "_phase_2_no_plan.html", {
        "project": project, "run_id": run_id,
    })


@app.get("/runs/{project}/{run_id}/windows_json")
async def windows_json(project: str, run_id: str):
    """Refresh window list — returns JSON for HTMX out-of-band swap or AJAX."""
    from ..tools.window_enum import list_windows
    return {"windows": [{"title": w.title, "pid": w.pid} for w in list_windows()]}


@app.post("/runs/{project}/{run_id}/record_test", response_class=HTMLResponse)
async def record_test(project: str, run_id: str,
                      window_title: str = Form(...),
                      duration_s: float = Form(30.0)):
    if REGISTRY.is_running(run_id):
        raise HTTPException(409, "another action already running")
    if not window_title.strip():
        raise HTTPException(400, "window_title required")
    if duration_s < 5 or duration_s > 600:
        raise HTTPException(400, "duration_s must be 5..600")

    pipe = REGISTRY.get_or_load(project, run_id)
    asyncio.create_task(_record_test_async(pipe, window_title.strip(), duration_s))
    return HTMLResponse(
        '<div class="px-4 py-3 bg-amber-500/20 border border-amber-500/40 text-amber-200 rounded">'
        f'⏳ 测试录屏中... 抓窗口 <code class="text-emerald-300">{window_title[:60]}</code>'
        f' 时长 {duration_s:.0f}s'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


@app.post("/runs/{project}/{run_id}/accept_test", response_class=HTMLResponse)
async def accept_test(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 2:
        raise HTTPException(400, f"accept_test requires phase=2, current={pipe.state.phase}")
    rec_dir = pipe.run_dir / "recordings"
    state_path = rec_dir / "test_state.json"
    if not state_path.exists():
        raise HTTPException(400, "no test recording to accept")
    if not (rec_dir / "test.mp4").exists():
        raise HTTPException(400, "recordings/test.mp4 missing — record first")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    accepted = {
        "window_title": state.get("window_title"),
        "test_recording": state.get("output_path"),
        "ffprobe": state.get("ffprobe"),
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }
    accepted_path = pipe.run_dir / "accepted_window.json"
    accepted_path.write_text(
        json.dumps(accepted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pipe.bus.emit("user_input", agent="pipeline", action="accept_test_recording",
                  window_title=accepted["window_title"])
    pipe.record_asset("recording_test", rec_dir / "test.mp4", verified=True, source="window_record")
    # Route through OpenDesigner before Phase 3 — see accept_demo for rationale.
    pipe.transition(phase=2, gate="waiting_html")
    _fire_opendesign_bootstrap(pipe)
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        '<div class="px-4 py-3 bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 rounded">'
        '✅ 录屏已接受 · 启动 OpenDesigner...'
        '</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


@app.post("/runs/{project}/{run_id}/reject_test", response_class=HTMLResponse)
async def reject_test(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    rec_dir = pipe.run_dir / "recordings"
    for f in (rec_dir / "test.mp4", rec_dir / "test_state.json"):
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass
    pipe.bus.emit("user_input", agent="pipeline", action="reject_test_recording")
    return HTMLResponse(
        '<div class="px-4 py-3 bg-slate-500/20 border border-slate-500/40 text-slate-200 rounded">'
        '🗑 测试录屏已删除，可以重新选窗口录'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


_PHASE2_REFRESH = {"HX-Trigger": "phase2-refresh"}


@app.post("/runs/{project}/{run_id}/draft_plan", response_class=HTMLResponse)
async def draft_plan(project: str, run_id: str, feedback: str = Form("")):
    if REGISTRY.is_running(run_id):
        raise HTTPException(409, "agent already running")
    pipe = REGISTRY.get_or_load(project, run_id)
    asyncio.create_task(_run_planner_async(pipe, feedback.strip()))
    return HTMLResponse(
        '<div class="px-4 py-3 bg-amber-500/20 border border-amber-500/40 text-amber-200 rounded">'
        '⏳ Agent 2 草拟启动计划中…'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


@app.post("/runs/{project}/{run_id}/execute_plan", response_class=HTMLResponse)
async def execute_plan_handler(project: str, run_id: str):
    if REGISTRY.is_running(run_id):
        raise HTTPException(409, "another action already running")
    pipe = REGISTRY.get_or_load(project, run_id)
    plan_path = pipe.run_dir / "setup_plan.json"
    if not plan_path.exists():
        raise HTTPException(400, "no setup_plan.json — generate one first")
    asyncio.create_task(_execute_plan_async(pipe))
    return HTMLResponse(
        '<div class="px-4 py-3 bg-amber-500/20 border border-amber-500/40 text-amber-200 rounded">'
        '⏳ 执行中：装依赖 → seed → 起服务 → health check'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


@app.get("/runs/{project}/{run_id}/errors", response_class=HTMLResponse)
async def errors_panel(project: str, run_id: str, request: Request):
    """Show all error escalations + ErrorAgent suggestions pending review.

    Each suggestion comes from <run_dir>/error_suggestions.jsonl (written by
    ErrorAgent when an agent's LLM call retries are exhausted). User can
    Apply / Reject / Mark as informational.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.error_agent import read_pending_suggestions
    from ..observability.error_log import read_errors
    suggestions = read_pending_suggestions(pipe.run_dir)
    recent_errors = read_errors(pipe.run_dir, limit=50)
    return TEMPLATES.TemplateResponse(request, "errors.html", {
        "project": project, "run_id": run_id,
        "suggestions": suggestions,
        "recent_errors": recent_errors,
    })


@app.post("/runs/{project}/{run_id}/errors/mark", response_class=HTMLResponse)
async def errors_mark(project: str, run_id: str,
                         ts: str = Form(...), status: str = Form(...)):
    """Update a suggestion's status (applied / rejected / informational)."""
    pipe = REGISTRY.get_or_load(project, run_id)
    path = pipe.run_dir / "error_suggestions.jsonl"
    if not path.exists():
        return HTMLResponse('<div class="text-rose-300 text-xs">no suggestions file</div>', 404)
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = 0
    for i, ln in enumerate(lines):
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        if rec.get("ts") == ts:
            rec["status"] = status
            rec["status_updated_at"] = datetime.now(timezone.utc).isoformat()
            lines[i] = json.dumps(rec, ensure_ascii=False)
            updated += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pipe.bus.emit("user_input", agent="error_review",
                   action=status, suggestion_ts=ts)
    return HTMLResponse(
        f'<div class="text-emerald-300 text-xs">✓ marked {status} ({updated} record(s))</div>',
        headers={"HX-Trigger": "errors-refresh"},
    )


@app.post("/runs/{project}/{run_id}/retry", response_class=HTMLResponse)
async def retry_failed_phase(project: str, run_id: str):
    """Re-fire the agent that recorded last_error.

    The retry mapping is keyed by `last_error.agent` — no auto-fallback chains,
    no thresholds, just user-clicked re-run of the same code with the same
    persisted inputs. If the agent name isn't mapped (e.g. Phase 2b recording
    needs window_title that wasn't persisted), we clear the error + redirect
    so the user re-enters params manually.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.last_error is None:
        return HTMLResponse(
            '<div class="text-rose-300 text-xs">没有可重试的错误</div>', 400,
        )
    err = pipe.state.last_error
    agent = err.get("agent", "")
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)

    # Auto-retry mapping for agents whose inputs are persisted on disk.
    # Phase 2b recording / demo_planner / demo_driver / etc. need params that
    # aren't recoverable, so they fall through to "manual" branch below.
    auto: Optional[str] = None
    if agent in ("git-clone", "local-copy"):
        auto = "clone+analyze"
    elif agent == "Agent 1 ProjectAnalyzer":
        auto = "analyze"
    elif agent == "Agent 2 SetupRunner":
        auto = "planner"
    elif agent == "Plan Executor":
        auto = "execute_plan"
    elif agent == "Agent 3 RemotionComposer":
        auto = "phase3"
    elif agent == "Agent 4 BGMComposer":
        auto = "phase4"
    elif agent == "Agent 5 VoiceOver":
        auto = "phase5"

    if auto is None:
        pipe.clear_error()
        pipe.transition(gate="running")
        return HTMLResponse(
            '<div class="text-amber-300 text-sm">'
            'ℹ 这个 agent（' + agent + '）需要你手动重启动 — '
            '错误已清除，请在下方面板里点回相应的按钮。</div>',
            headers={"HX-Trigger": "retry-cleared"},
        )

    pipe.clear_error()
    pipe.transition(gate="running")
    pipe.log.info(f"user retry: {agent} → {auto}")
    target_url = f"/runs/{project}/{run_id}"

    if auto == "clone+analyze":
        asyncio.create_task(_retry_clone_and_analyze(pipe))
    elif auto == "analyze":
        asyncio.create_task(_run_analyzer_async(pipe, pipe.state.repo_url or ""))
    elif auto == "planner":
        asyncio.create_task(_run_planner_async(pipe, feedback=""))
    elif auto == "execute_plan":
        asyncio.create_task(_execute_plan_async(pipe))
    elif auto == "phase3":
        asyncio.create_task(_run_phase3_async(pipe))
    elif auto == "phase4":
        asyncio.create_task(_run_phase4_async(pipe))
    elif auto == "phase5":
        asyncio.create_task(_run_phase5_async(pipe, lang="zh-CN"))

    return HTMLResponse(
        f'<div class="text-emerald-300 text-sm">⏳ 重试 <code>{agent}</code> 启动中 · 跳转中...</div>',
        # HX-Redirect = full page reload so user sees fresh state (banner gone,
        # agent re-running, etc) — without this the retry happens but the page
        # stays on the stale view and feels like "nothing happened".
        headers={"HX-Redirect": target_url, "Location": target_url, "HX-Trigger": "retry-started"},
    )


@app.get("/runs/{project}/{run_id}/scores", response_class=HTMLResponse)
async def scores_panel(project: str, run_id: str, request: Request):
    """Show LLM-as-a-Judge scores per phase + final video user rating slot.

    Auto scores live at <run_dir>/scores.jsonl (one record per judge call).
    The final video rating is also written there with source='user'.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..tools.score_log import read_local_scores
    from ..tools.run_metrics import read_metrics

    all_scores = read_local_scores(pipe.run_dir)

    # Most-recent auto_judge per phase (deduped by phase). Includes the
    # 4 core rubrics plus the new ones: demo_captions / voiceover_zh /
    # voiceover_en / cross_phase_consistency.
    phases_seen: set[str] = set()
    auto_by_phase: list[dict] = []
    cross_phase: Optional[dict] = None
    for rec in all_scores:
        if rec.get("source") != "auto_judge":
            continue
        ph = rec.get("phase")
        if ph == "cross_phase_consistency":
            if cross_phase is None:  # newest only
                cross_phase = rec
            continue
        if ph in phases_seen:
            continue
        phases_seen.add(ph)
        auto_by_phase.append(rec)

    # Deterministic metrics — newest per phase
    metric_phases_seen: set[str] = set()
    metrics_by_phase: list[dict] = []
    for rec in read_metrics(pipe.run_dir):
        ph = rec.get("phase")
        if ph in metric_phases_seen:
            continue
        metric_phases_seen.add(ph)
        metrics_by_phase.append(rec)
    # Order phase1..phase5
    _phase_order = {f"phase{i}_metrics": i for i in range(1, 6)}
    metrics_by_phase.sort(key=lambda r: _phase_order.get(r.get("phase", ""), 99))

    # Most-recent user rating (final video)
    user_rating = next((r for r in all_scores if r.get("source") == "user"), None)

    # Find final video for the rating widget
    final_zh = pipe.run_dir / "outputs" / "final_zh-CN.mp4"
    final_en = pipe.run_dir / "outputs" / "final_en-US.mp4"
    final_path = None
    for cand in (final_zh, final_en):
        if cand.exists():
            final_path = cand.name
            break

    return TEMPLATES.TemplateResponse(request, "scores.html", {
        "project": project, "run_id": run_id,
        "auto_scores": auto_by_phase,
        "cross_phase": cross_phase,
        "metrics_by_phase": metrics_by_phase,
        "user_rating": user_rating,
        "final_video": final_path,
        "raw_scores": all_scores[:20],
    })


@app.post("/runs/{project}/{run_id}/score_final", response_class=HTMLResponse)
async def submit_final_rating(project: str, run_id: str,
                                rating: float = Form(...),
                                comment: str = Form("")):
    """User submits a 1-5 star rating for the final video.

    Distinct from the auto-judge — user rates the END artifact only.
    Stored via record_user_video_rating() in scores.jsonl; view on /scores.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.quality_judge import record_user_video_rating
    try:
        rec = record_user_video_rating(pipe.run_dir, rating, comment)
    except ValueError as e:
        return HTMLResponse(
            f'<div class="text-rose-300 text-xs">{e}</div>', 400,
        )
    pipe.bus.emit("user_input", agent="user_rating",
                   rating=rating, comment=comment[:200])
    return HTMLResponse(
        f'<div class="text-emerald-300 text-sm">'
        f'✓ 已提交：{rating}/5 stars'
        + (f'<br><span class="text-xs text-slate-400">{comment[:200]}</span>' if comment else '')
        + '</div>',
        headers={"HX-Trigger": "scores-refresh"},
    )


@app.post("/runs/{project}/{run_id}/score_rerun", response_class=HTMLResponse)
async def rerun_judge(project: str, run_id: str, phase: str = Form(...)):
    """Re-run auto-judge for one phase (useful after editing prompts)."""
    pipe = REGISTRY.get_or_load(project, run_id)
    if phase not in ("brief", "setup_plan", "cutting_plan", "voiceover_script"):
        return HTMLResponse(
            f'<div class="text-rose-300 text-xs">invalid phase: {phase}</div>', 400,
        )
    _fire_quality_judge(pipe, phase)
    return HTMLResponse(
        f'<div class="text-emerald-300 text-xs">⏳ re-judging {phase}… 刷新看结果</div>',
    )


@app.get("/runs/{project}/{run_id}/prompts", response_class=HTMLResponse)
async def prompts_panel(project: str, run_id: str, request: Request):
    """Show all agent SYSTEM_PROMPTs + per-run override editor.

    Default prompts come from the agent modules. Overrides live at
    `<run_dir>/prompts/<agent_key>.txt` — saved by the editor below.
    Empty save = remove override.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents._prompt_override import list_overrides
    from ..agents import project_analyzer, setup_runner, demo_driver, remotion_composer, voice_over

    overrides = list_overrides(pipe.run_dir)

    def _read_override(key: str) -> str:
        p = pipe.run_dir / "prompts" / f"{key}.txt"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    agents_info = [
        {"key": "project_analyzer", "name": "Agent 1 · ProjectAnalyzer",
         "phase": "Phase 1", "model": "LLM_REASONING (default: glm-5.1)",
         "default_prompt": project_analyzer.SYSTEM_PROMPT_BASE + project_analyzer.DEEP_MODE_ADDENDUM,
         "default_note": "Base + Deep-mode addendum (deep mode adds extra exploration instructions)",
         "override": _read_override("project_analyzer"),
         "has_override": overrides.get("project_analyzer", False)},
        {"key": "setup_runner", "name": "Agent 2 · SetupRunner",
         "phase": "Phase 2a",
         "model": "LLM_REASONING",
         "default_prompt": setup_runner.SYSTEM_PROMPT,
         "default_note": "Drives check_tool / config_writes / install_commands / services.",
         "override": _read_override("setup_runner"),
         "has_override": overrides.get("setup_runner", False)},
        {"key": "demo_driver", "name": "Agent · Demo Driver",
         "phase": "Phase 2c",
         "model": "LLM_VISION (default: kimi-k2.6) for web mode",
         "default_prompt": demo_driver.SYSTEM_PROMPT,
         "default_note": "Operates the running project (browser/CLI) + emits captions.",
         "override": _read_override("demo_driver"),
         "has_override": overrides.get("demo_driver", False)},
        {"key": "remotion_composer", "name": "Agent 3 · RemotionComposer",
         "phase": "Phase 3",
         "model": "LLM_REASONING",
         "default_prompt": remotion_composer.SYSTEM_PROMPT,
         "default_note": "Drafts the cutting plan (5-8 scenes, 30-45s).",
         "override": _read_override("remotion_composer"),
         "has_override": overrides.get("remotion_composer", False)},
        {"key": "voice_over", "name": "Agent 5 · VoiceOver",
         "phase": "Phase 5",
         "model": "LLM_REASONING",
         "default_prompt": voice_over.SYSTEM_PROMPT,
         "default_note": "Drafts bilingual voiceover script.",
         "override": _read_override("voice_over"),
         "has_override": overrides.get("voice_over", False)},
    ]
    return TEMPLATES.TemplateResponse(request, "prompts.html", {
        "project": project, "run_id": run_id, "agents": agents_info,
    })


@app.post("/runs/{project}/{run_id}/prompts/{agent_key}", response_class=HTMLResponse)
async def save_prompt_override(project: str, run_id: str, agent_key: str,
                                  text: str = Form("")):
    """Save (or clear) a per-run override for one agent's SYSTEM_PROMPT."""
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents._prompt_override import save_override
    target = save_override(agent_key, text, pipe.run_dir)
    if not text.strip():
        return HTMLResponse(
            f'<div class="text-xs text-slate-400 py-1">已删除 override，{agent_key} 下次跑用默认 prompt</div>')
    pipe.log.info(f"prompt override saved: {agent_key} ({len(text)} chars)")
    return HTMLResponse(
        f'<div class="text-xs text-emerald-400 py-1">✓ 已保存 ({len(text)} chars)。{agent_key} 下次启动会用这个</div>')


@app.post("/runs/{project}/{run_id}/provide_secrets", response_class=HTMLResponse)
async def provide_secrets_handler(project: str, run_id: str, request: Request):
    """User fills the user_secrets_needed form → write user_secrets.json, then
    re-trigger execute_plan so it can pass the gate.

    Form body: each declared `var_name` is a field. We accept whatever the
    user typed (may be empty if they want to skip that secret; if a config_writes
    template references it, executor will halt with a missing-var error which
    surfaces back to them).
    """
    if REGISTRY.is_running(run_id):
        raise HTTPException(409, "another action already running")
    pipe = REGISTRY.get_or_load(project, run_id)
    plan_path = pipe.run_dir / "setup_plan.json"
    if not plan_path.exists():
        raise HTTPException(400, "no setup_plan.json")

    form = await request.form()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    declared = {s["var_name"] for s in (plan.get("user_secrets_needed") or [])
                if isinstance(s, dict) and s.get("var_name")}
    secrets = {k: (v or "").strip() for k, v in form.items() if k in declared}

    # Merge with existing (in case user is updating after a previous fill)
    existing_path = pipe.run_dir / "user_secrets.json"
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            secrets = {**existing, **secrets}
        except Exception:
            pass
    existing_path.write_text(json.dumps(secrets, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    pipe.bus.emit("user_input", agent="executor",
                   action="provide_secrets", vars=list(secrets.keys()))
    pipe.log.info(f"user provided {len(secrets)} secret(s): {list(secrets.keys())}")

    # Re-trigger execution now that gate should pass.
    asyncio.create_task(_execute_plan_async(pipe))
    return HTMLResponse(
        '<div class="px-4 py-3 bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 rounded">'
        f'✓ 已保存 {len(secrets)} 个密钥，继续执行 install / seed / 起服务…'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


@app.post("/runs/{project}/{run_id}/restart_services", response_class=HTMLResponse)
async def restart_services(project: str, run_id: str):
    """Re-run install/seed/start of the existing plan (e.g. after reboot)."""
    if REGISTRY.is_running(run_id):
        raise HTTPException(409, "another action already running")
    pipe = REGISTRY.get_or_load(project, run_id)
    plan_path = pipe.run_dir / "setup_plan.json"
    if not plan_path.exists():
        raise HTTPException(400, "no setup_plan.json")
    asyncio.create_task(_execute_plan_async(pipe))
    return HTMLResponse(
        '<div class="px-4 py-3 bg-amber-500/20 border border-amber-500/40 text-amber-200 rounded">'
        '⏳ 重新执行启动计划：装依赖（缓存命中很快） → seed → 起服务 → health check'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


@app.post("/runs/{project}/{run_id}/open_project_url", response_class=HTMLResponse)
async def open_project_url(project: str, run_id: str):
    """Open the frontend service URL in the user's default browser."""
    pipe = REGISTRY.get_or_load(project, run_id)
    plan_path = pipe.run_dir / "setup_plan.json"
    if not plan_path.exists():
        raise HTTPException(400, "no setup_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    services = plan.get("services") or []
    front = next((s for s in services if any(
        h in (s.get("name", "") + " " + s.get("purpose", "")).lower()
        for h in ("frontend", "ui", "web", "static"))), None) or (services[0] if services else None)
    if not front:
        raise HTTPException(400, "no service to open")
    url = front.get("health_url") or ""
    if url.endswith("/health"):
        url = url[:-len("/health")] + "/"
    import webbrowser
    webbrowser.open(url)
    return HTMLResponse(
        f'<div class="px-4 py-3 bg-emerald-500/20 border border-emerald-500/40 text-emerald-200 rounded">'
        f'🌐 已尝试打开 <code class="text-emerald-100">{url}</code>。'
        f' 几秒后点 ↻ 刷新窗口列表，新窗口会被自动检测并推荐。'
        f'</div>',
        headers=_PHASE2_REFRESH,
    )


@app.post("/runs/{project}/{run_id}/stop_services", response_class=HTMLResponse)
async def stop_services(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..tools.service_manager import ServiceManager
    mgr = ServiceManager(pipe.run_dir)
    mgr.stop_all()
    pipe.bus.emit("user_input", agent="pipeline", action="stop_services")
    return HTMLResponse(
        '<div class="px-4 py-3 bg-slate-500/20 border border-slate-500/40 text-slate-200 rounded">'
        '⏹ 服务已停止'
        '</div>',
        headers=_PHASE2_REFRESH,
    )


@app.get("/runs/{project}/{run_id}/briefs_panel", response_class=HTMLResponse)
async def briefs_panel(project: str, run_id: str, request: Request):
    """Render the two-version brief cards (standard + deep). Polled every 3s."""
    briefs_dir = WORKSPACE_ROOT / project / "runs" / run_id / "briefs"
    versions = _load_brief_versions(briefs_dir)
    return TEMPLATES.TemplateResponse(request, "_briefs_cards.html", {
        "project": project, "run_id": run_id, "versions": versions,
    })


@app.get("/runs/{project}/{run_id}/iterate_panel", response_class=HTMLResponse)
async def iterate_panel(project: str, run_id: str, request: Request):
    """Polling target: returns progress card if agent running, else iterate form."""
    run_dir = WORKSPACE_ROOT / project / "runs" / run_id
    is_running = REGISTRY.is_running(run_id)

    progress = None
    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            progress = None

    if is_running:
        return TEMPLATES.TemplateResponse(request, "_iterate_running.html", {
            "project": project, "run_id": run_id,
            "progress": progress,
        })
    versions = _load_brief_versions(run_dir / "briefs")
    return TEMPLATES.TemplateResponse(request, "_iterate_form.html", {
        "project": project, "run_id": run_id,
        "versions": versions,
    })


@app.get("/runs/{project}/{run_id}/demo_preview")
async def demo_preview(project: str, run_id: str):
    """Latest Demo Driver page screenshot — refreshed after every browser-* tool
    call. Used by the Phase 2c UI to show a ~2s-delay live preview without
    pausing the recording. Returns 404 until the first screenshot lands."""
    run_dir = WORKSPACE_ROOT / project / "runs" / run_id
    p = run_dir / "demo_preview.png"
    if not p.exists():
        raise HTTPException(404, "no preview yet — Demo Driver may not have taken its first browser action")
    return FileResponse(
        p, media_type="image/png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/runs/{project}/{run_id}/artifacts/{rel_path:path}")
async def get_artifact(project: str, run_id: str, rel_path: str):
    """Serve a file from the run dir. Supports nested paths e.g. recordings/test.mp4."""
    base = (WORKSPACE_ROOT / project / "runs" / run_id).resolve()
    p = (base / rel_path).resolve()
    # Path-traversal guard
    if base != p and base not in p.parents:
        raise HTTPException(403, "path escapes run directory")
    if not p.exists() or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/runs/{project}/{run_id}/status_chip", response_class=HTMLResponse)
async def status_chip(project: str, run_id: str):
    """HTMX poll target — current agent running state."""
    if REGISTRY.is_running(run_id):
        return HTMLResponse(
            '<span class="inline-flex items-center gap-2 px-2 py-1 bg-amber-500/20 '
            'border border-amber-500/40 text-amber-300 text-xs rounded">'
            '<span class="w-2 h-2 bg-amber-400 rounded-full animate-pulse"></span>'
            'Agent running'
            '</span>'
        )
    return HTMLResponse(
        '<span class="inline-flex items-center gap-2 px-2 py-1 bg-slate-700/50 '
        'border border-slate-600 text-slate-400 text-xs rounded">'
        'Idle'
        '</span>'
    )


# ────────────────────────────────────────────────────────────
# Observability viewer · /runs/{project}/{run_id}/observability
# ────────────────────────────────────────────────────────────

def _read_events_recent(run_dir: Path, limit: int = 500) -> list[dict]:
    """Read events.jsonl, return last `limit` parsed entries."""
    p = run_dir / "events.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except Exception:
        return []
    return out[-limit:]


def _summarize_logs(run_dir: Path) -> list[dict]:
    """Inventory of logs/*.jsonl: name, lines, size_bytes, last_line."""
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        return []
    items: list[dict] = []
    for f in sorted(log_dir.glob("*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            lines = [l for l in text.splitlines() if l.strip()]
            n = len(lines)
            last = lines[-1] if lines else ""
            # Try to extract human-readable message from last loguru-serialize line
            preview = ""
            try:
                rec = json.loads(last)
                preview = (rec.get("record", {}).get("message") or rec.get("text") or last)[:200]
            except Exception:
                preview = last[:200]
            items.append({
                "name": f.name,
                "lines": n,
                "size_bytes": f.stat().st_size,
                "preview": preview,
            })
        except Exception:
            pass
    return items


def _events_summary(events: list[dict]) -> dict:
    """Count by event type + collect unique agent names."""
    by_type: dict = {}
    by_agent: dict = {}
    asset_count = 0
    error_count = 0
    for e in events:
        etype = e.get("event", "?")
        by_type[etype] = by_type.get(etype, 0) + 1
        agent = e.get("agent") or "(none)"
        by_agent[agent] = by_agent.get(agent, 0) + 1
        if etype == "asset_verified":
            asset_count += 1
        if etype == "asset_failed":
            error_count += 1
    return {
        "total": len(events),
        "by_type": by_type,
        "by_agent": by_agent,
        "asset_verified": asset_count,
        "asset_failed": error_count,
    }


@app.get("/runs/{project}/{run_id}/trace")
@app.get("/runs/{project}/{run_id}/logs")
@app.get("/runs/{project}/{run_id}/traces")
async def observability_alias(project: str, run_id: str):
    """Convenience aliases — redirect to /observability."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/runs/{project}/{run_id}/observability", status_code=302)


@app.get("/observability", response_class=HTMLResponse)
@app.get("/trace", response_class=HTMLResponse)
@app.get("/traces", response_class=HTMLResponse)
@app.get("/logs", response_class=HTMLResponse)
async def observability_global(request: Request):
    """Global Trace + Logs dashboard — lists all runs with their event counts."""
    runs_data = []
    for r in _list_runs():
        run_dir = WORKSPACE_ROOT / r["project"] / "runs" / r["run_id"]
        events = _read_events_recent(run_dir, limit=10000)
        summary = _events_summary(events)
        log_files = _summarize_logs(run_dir)
        runs_data.append({
            "project": r["project"],
            "run_id": r["run_id"],
            "phase": r["phase"],
            "gate": r["gate"],
            "events_total": summary["total"],
            "asset_verified": summary["asset_verified"],
            "asset_failed": summary["asset_failed"],
            "n_logs": len(log_files),
            "last_event": events[-1] if events else None,
        })
    return TEMPLATES.TemplateResponse(request, "observability_global.html", {
        "runs": runs_data,
        "phoenix_url": "http://localhost:6006/",
    })


@app.get("/runs/{project}/{run_id}/observability", response_class=HTMLResponse)
async def observability_page(project: str, run_id: str, request: Request):
    pipe = REGISTRY.get_or_load(project, run_id)
    events = _read_events_recent(pipe.run_dir)
    summary = _events_summary(events)
    log_files = _summarize_logs(pipe.run_dir)
    return TEMPLATES.TemplateResponse(request, "observability.html", {
        "project": project, "run_id": run_id,
        "events": events,
        "summary": summary,
        "log_files": log_files,
        "phoenix_url": "http://localhost:6006/",
    })


@app.get("/runs/{project}/{run_id}/observability/log/{name}")
async def observability_log_view(project: str, run_id: str, name: str,
                                   tail: int = 200):
    """Stream last `tail` lines of a specific JSONL log (raw text)."""
    pipe = REGISTRY.get_or_load(project, run_id)
    p = (pipe.run_dir / "logs" / name).resolve()
    base = (pipe.run_dir / "logs").resolve()
    if base != p.parent:
        raise HTTPException(403, "path traversal")
    if not p.exists() or not p.is_file():
        raise HTTPException(404)
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        raise HTTPException(500, str(e))
    body_lines: list[str] = []
    for raw in lines[-tail:]:
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
            r = rec.get("record", {})
            ts = r.get("time", {}).get("repr", "") or rec.get("time", "")
            level = r.get("level", {}).get("name", "?") or rec.get("level", "")
            msg = r.get("message", "") or rec.get("message", "") or rec.get("text", raw)
            extra = r.get("extra", {})
            agent = extra.get("agent", "") or rec.get("agent", "")
            body_lines.append(f"{ts}  {level:<7}  {agent:<25}  {msg}")
        except Exception:
            body_lines.append(raw)
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(body_lines))


@app.get("/runs/{project}/{run_id}/observability/events_stream")
async def observability_events_stream(project: str, run_id: str):
    """SSE stream — pushes 'observability_changed' event when events.jsonl mtime changes."""
    pipe = REGISTRY.get_or_load(project, run_id)
    events_path = pipe.run_dir / "events.jsonl"

    async def gen():
        last_mtime: float = -1.0
        last_size: int = -1
        while True:
            try:
                if events_path.exists():
                    st = events_path.stat()
                    if st.st_mtime != last_mtime or st.st_size != last_size:
                        last_mtime = st.st_mtime
                        last_size = st.st_size
                        yield f"event: observability_changed\ndata: {{\"size\": {st.st_size}}}\n\n"
            except Exception:
                pass
            await asyncio.sleep(2.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ────────────────────────────────────────────────────────────
# Phase 2B · Smart Demo (LLM plan → playwright record → captions)
# ────────────────────────────────────────────────────────────

def _service_url(pipe: Pipeline) -> Optional[str]:
    """Read services.json and return the first service's health_url."""
    sf = pipe.run_dir / "services.json"
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text(encoding="utf-8"))
        services = data.get("services", data) if isinstance(data, dict) else data
        if isinstance(services, list) and services:
            s0 = services[0]
            return s0.get("health_url") or s0.get("url")
    except Exception:
        pass
    return None


@app.post("/runs/{project}/{run_id}/plan_demo", response_class=HTMLResponse)
async def plan_demo_route(project: str, run_id: str):
    """Kick off LLM demo planner in background."""
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    url = _service_url(pipe)
    if not url:
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">没找到服务 URL — 先确保 Phase 2A services 起来了</div>',
            status_code=400,
        )
    brief_path = pipe.run_dir / "project_brief.md"
    if not brief_path.exists():
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">缺 project_brief.md（先完成 Phase 1）</div>',
            status_code=400,
        )

    asyncio.create_task(_plan_demo_async(pipe, url, brief_path))
    return HTMLResponse(
        '<div class="text-amber-300 text-sm py-2">⏳ LLM 规划 demo 中（snapshot DOM + 选 5-8 步）...</div>',
        headers={"HX-Trigger": "phase2-refresh"},
    )


async def _plan_demo_async(pipe: Pipeline, service_url: str, brief_path: Path) -> None:
    await REGISTRY.mark_running(pipe.run_id, "phase2b-demo-plan")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..agents.demo_planner import plan_demo
        out = pipe.run_dir / "recordings" / "demo_script.json"
        brief = brief_path.read_text(encoding="utf-8")
        await asyncio.to_thread(plan_demo, pipe.run_dir, service_url, brief, out, 25.0)
    except Exception as e:
        pipe.log.exception(f"plan_demo failed: {e}")
        pipe.bus.emit("asset_failed", agent="demo_planner",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="Demo Planner",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


@app.post("/runs/{project}/{run_id}/record_demo", response_class=HTMLResponse)
async def record_demo_route(project: str, run_id: str):
    """Execute the existing demo_script + record + write captions."""
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    script = pipe.run_dir / "recordings" / "demo_script.json"
    if not script.exists():
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">先点 "📐 LLM 规划 demo" 生成 demo_script.json</div>',
            status_code=400,
        )
    asyncio.create_task(_record_demo_async(pipe, script))
    return HTMLResponse(
        '<div class="text-amber-300 text-sm py-2">⏳ 执行 demo + 录屏 + 同步字幕中（25-45s）...</div>',
        headers={"HX-Trigger": "phase2-refresh"},
    )


async def _record_demo_async(pipe: Pipeline, script_path: Path) -> None:
    await REGISTRY.mark_running(pipe.run_id, "phase2b-demo-record")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..tools.demo_executor import execute_demo
        from ..tools.captions import write_caption_tracks
        rec_dir = pipe.run_dir / "recordings"
        out_video = rec_dir / "demo_recording.mp4"
        timings_path = rec_dir / "demo_timings.json"
        await asyncio.to_thread(
            execute_demo, script_path, out_video, timings_path,
            1920, 1080, True,
        )
        # Write captions immediately so UI can show count + preview
        await asyncio.to_thread(write_caption_tracks, timings_path, rec_dir)
    except Exception as e:
        pipe.log.exception(f"record_demo failed: {e}")
        pipe.bus.emit("asset_failed", agent="demo_executor",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="Demo Executor",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


# NOTE: legacy /accept_demo for demo_executor (with_captions burn-in) was here
# before. Removed because it shadowed the new Demo-Driver /accept_demo route
# below (FastAPI uses first-registered match). The new route at line ~1899
# handles demo.mp4 → test.mp4 + advances to Phase 3.


# ────────────────────────────────────────────────────────────
# Phase 2B · CLI/TUI Terminal Recorder (for non-web projects)
# ────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────
# Phase 3 · Remotion (cutting_plan + codegen + render)
# ────────────────────────────────────────────────────────────

@app.post("/runs/{project}/{run_id}/run_phase3", response_class=HTMLResponse)
async def run_phase3_route(project: str, run_id: str, feedback: str = Form("")):
    pipe = REGISTRY.get_or_load(project, run_id)
    # Guard against concurrent fire: if a Phase 3 (or any agent) is already
    # running for this run, refuse a second trigger. Otherwise rapid clicks
    # spawn N concurrent renders / N concurrent BGM calls.
    if REGISTRY.is_running(run_id):
        return HTMLResponse(
            '<div class="text-amber-300 text-sm py-2 px-3 bg-amber-500/10 border border-amber-500/30 rounded">'
            '⏳ 已经在跑 — 等当前 phase 完成再触发（面板会自动刷新进度）'
            '</div>', status_code=409,
        )
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    # Record start time so the panel can show "elapsed Xs" without parsing logs.
    import time as _time
    (pipe.run_dir / "phase3_started.txt").write_text(str(_time.time()), encoding="utf-8")
    asyncio.create_task(_run_phase3_async(pipe, feedback=feedback.strip() or None))
    return HTMLResponse(
        '<div class="text-amber-300 text-sm py-2 px-3 bg-amber-500/10 border border-amber-500/30 rounded">'
        '⏳ Phase 3 启动中 · 面板会自动刷新显示进度（~5-10min）'
        '</div>',
        headers={"HX-Trigger": "phase3-refresh"},
    )


@app.post("/runs/{project}/{run_id}/rewind_to_phase/{n}", response_class=HTMLResponse)
async def rewind_to_phase(project: str, run_id: str, n: int):
    """User-driven rewind: pull state.phase back to `n` (must be < current).

    Use case: state advanced prematurely (e.g. before bug fix /approve blindly
    bumped any phase, or a polling double-submit raced). User clicks rewind
    on a phase-N panel that's missing prereqs → state.phase resets so they
    can complete the missing step. NOT auto-rewind — user click only.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    if not (1 <= n < pipe.state.phase):
        raise HTTPException(
            400,
            f"rewind target must be < current phase. Got n={n}, current={pipe.state.phase}",
        )
    prev = pipe.state.phase
    pipe.state.last_error = None
    pipe.transition(phase=n, gate="running")
    pipe.log.info(f"user rewind: phase {prev} → {n}")
    target = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        f'<div class="text-amber-300 text-sm">↩ 已回退到 Phase {n} · 跳转中...</div>',
        headers={"HX-Redirect": target, "Location": target},
    )


@app.get("/runs/{project}/{run_id}/view_phase/{n}", response_class=HTMLResponse)
async def view_phase_readonly(project: str, run_id: str, n: int, request: Request):
    """Read-only view of a past phase's artifacts.

    Click a green P1..P{state.phase-1} dot in the header → land here. Pure
    artifact display: brief / setup_plan / cutting_plan / v1.mp4 / final.mp4.
    No buttons that mutate state. "← Back to live" returns to the live run.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    state = pipe.state
    if not (1 <= n <= 5):
        raise HTTPException(400, "phase must be 1-5")
    if n > state.phase:
        raise HTTPException(400, f"phase {n} not reached yet (current={state.phase})")

    run_dir = pipe.run_dir
    artifacts: dict = {}

    if n == 1:
        b = run_dir / "project_brief.md"
        artifacts["brief_md"] = b.read_text(encoding="utf-8") if b.exists() else None
        if artifacts["brief_md"]:
            artifacts["brief_html"] = md_renderer.render(artifacts["brief_md"])
        briefs_dir = run_dir / "briefs"
        artifacts["versions"] = _load_brief_versions(briefs_dir) if briefs_dir.exists() else {}

    elif n == 2:
        sp = run_dir / "setup_plan.json"
        if sp.exists():
            try:
                artifacts["setup_plan"] = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                artifacts["setup_plan"] = None
        se = run_dir / "setup_exec.json"
        if se.exists():
            try:
                artifacts["setup_exec"] = json.loads(se.read_text(encoding="utf-8"))
            except Exception:
                artifacts["setup_exec"] = None
        test_rec = run_dir / "recordings" / "test.mp4"
        if test_rec.exists():
            artifacts["test_recording_size_mb"] = round(test_rec.stat().st_size / 1024 / 1024, 2)
        demo_rec = run_dir / "recordings" / "demo_recording.mp4"
        if demo_rec.exists():
            artifacts["demo_recording_size_mb"] = round(demo_rec.stat().st_size / 1024 / 1024, 2)
        captions = run_dir / "demo_captions.jsonl"
        if captions.exists():
            artifacts["caption_count"] = len(
                [l for l in captions.read_text(encoding="utf-8").splitlines() if l.strip()]
            )

    elif n == 3:
        cp = run_dir / "cutting_plan.json"
        if cp.exists():
            try:
                artifacts["cutting_plan"] = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                artifacts["cutting_plan"] = None
        v1 = run_dir / "outputs" / "v1.mp4"
        if v1.exists():
            artifacts["v1_size_mb"] = round(v1.stat().st_size / 1024 / 1024, 2)

    elif n == 4:
        v1bgm = run_dir / "outputs" / "v1_bgm_final.mp4"
        if v1bgm.exists():
            artifacts["v1_bgm_size_mb"] = round(v1bgm.stat().st_size / 1024 / 1024, 2)
        bgm_dir = run_dir / "bgm"
        if bgm_dir.exists():
            artifacts["bgm_files"] = sorted(
                f.name for f in bgm_dir.glob("*.wav")
            )

    elif n == 5:
        final_zh = run_dir / "outputs" / "final_zh-CN.mp4"
        final_en = run_dir / "outputs" / "final_en-US.mp4"
        if final_zh.exists():
            artifacts["final_zh_size_mb"] = round(final_zh.stat().st_size / 1024 / 1024, 2)
        if final_en.exists():
            artifacts["final_en_size_mb"] = round(final_en.stat().st_size / 1024 / 1024, 2)
        vs = run_dir / "voice" / "voiceover_script_bilingual.json"
        if vs.exists():
            try:
                artifacts["voiceover_script"] = json.loads(vs.read_text(encoding="utf-8"))
            except Exception:
                artifacts["voiceover_script"] = None

    return TEMPLATES.TemplateResponse(request, "phase_readonly.html", {
        "project": project, "run_id": run_id,
        "phase_num": n, "state": state,
        "artifacts": artifacts,
        "phase_label": _phase_label(n),
    })


@app.get("/runs/{project}/{run_id}/phase3_panel", response_class=HTMLResponse)
async def phase3_panel(project: str, run_id: str, request: Request):
    """Live Phase-3 panel — polled every 3s by _phase_3.html.

    Surfaces: is_running flag, progress.json (latest step), elapsed time
    since phase3_started.txt, cutting_plan.json size, v1.mp4 existence/size.
    Removes the user's "is it running or dead?" guesswork during the
    multi-minute Remotion render.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    run_dir = pipe.run_dir
    is_running = REGISTRY.is_running(run_id)
    progress = None
    p_path = run_dir / "progress.json"
    if p_path.exists():
        try:
            progress = json.loads(p_path.read_text(encoding="utf-8"))
        except Exception:
            progress = None

    started_at = None
    started_path = run_dir / "phase3_started.txt"
    if started_path.exists():
        try:
            started_at = float(started_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    import time as _time
    elapsed_s = int(_time.time() - started_at) if started_at else None

    cutting_plan = run_dir / "cutting_plan.json"
    v1 = run_dir / "outputs" / "v1.mp4"
    # Prereq from Phase 2 — Remotion render needs test.mp4 as the recording asset.
    test_mp4 = run_dir / "recordings" / "test.mp4"
    project_brief = run_dir / "project_brief.md"

    return TEMPLATES.TemplateResponse(request, "_phase_3_panel.html", {
        "project": project, "run_id": run_id,
        "is_running": is_running,
        "progress": progress,
        "elapsed_s": elapsed_s,
        "cutting_plan_exists": cutting_plan.exists(),
        "cutting_plan_size_kb": (cutting_plan.stat().st_size // 1024) if cutting_plan.exists() else 0,
        "v1_exists": v1.exists(),
        "v1_size_mb": round(v1.stat().st_size / 1024 / 1024, 2) if v1.exists() else 0,
        "last_error": pipe.state.last_error,
        "state_gate": pipe.state.gate,
        "test_mp4_exists": test_mp4.exists(),
        "project_brief_exists": project_brief.exists(),
    })


async def _run_phase3_async(pipe: Pipeline, feedback: Optional[str] = None) -> None:
    await REGISTRY.mark_running(pipe.run_id, "phase3")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..agents.remotion_composer import run_cutting_planner
        from ..tools.remotion_codegen import generate_project
        from ..tools.remotion_render import npm_install, render
        from ..tools.shell import run as shell_run
        from ..tools.ffbin import ffprobe

        run_dir = pipe.run_dir
        recording = run_dir / "recordings" / "test.mp4"
        if not recording.exists():
            raise RuntimeError("recordings/test.mp4 missing — Phase 2 not done")
        brief = (run_dir / "project_brief.md").read_text(encoding="utf-8")

        # Probe the demo recording
        probe = await asyncio.to_thread(shell_run, [
            ffprobe(), "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", str(recording),
        ], check=True)
        data = json.loads(probe.stdout)
        v = next(s for s in data["streams"] if s["codec_type"] == "video")
        recording_meta = {
            "source_path": "recordings/test.mp4",
            "duration_s": float(data["format"]["duration"]),
            "width": int(v["width"]), "height": int(v["height"]),
            "codec": v["codec_name"], "fps": v["r_frame_rate"],
        }

        # Scan OpenDesigner outputs into available_assets. adopt(as_role='hero')
        # for motion_film-mode sessions lands the mp4 at hero/intro.mp4 — copy
        # it into hyperframes/ so the codegen's existing asset-staging path
        # (which only looks at hyperframes/) picks it up uniformly.
        hyperframes_dir = run_dir / "hyperframes"
        hero_intro = run_dir / "hero" / "intro.mp4"
        if hero_intro.exists():
            hyperframes_dir.mkdir(parents=True, exist_ok=True)
            staged = hyperframes_dir / "intro.mp4"
            if not staged.exists() or staged.stat().st_size != hero_intro.stat().st_size:
                import shutil as _shutil
                _shutil.copy2(hero_intro, staged)
        hyperframes = []
        mp4_sources: list[Path] = []
        if hyperframes_dir.exists():
            mp4_sources.extend(sorted(hyperframes_dir.glob("*.mp4")))
        for mp4 in mp4_sources:
            try:
                hp = await asyncio.to_thread(shell_run, [
                    ffprobe(), "-v", "quiet", "-print_format", "json",
                    "-show_streams", "-show_format", str(mp4),
                ], check=True)
                hd = json.loads(hp.stdout)
                hv = next((s for s in hd["streams"] if s["codec_type"] == "video"), {})
                rel = str(mp4.relative_to(run_dir)).replace("\\", "/")
                hyperframes.append({
                    "source_path": rel,
                    "duration_s": float(hd["format"].get("duration", 0)),
                    "width": int(hv.get("width", 0)),
                    "height": int(hv.get("height", 0)),
                })
            except Exception:
                continue

        html_pages = []
        html_dir = run_dir / "html_asset"
        if html_dir.exists():
            for html in sorted(html_dir.glob("*.html")):
                rel = str(html.relative_to(run_dir)).replace("\\", "/")
                # crude title extraction
                title = ""
                try:
                    text = html.read_text(encoding="utf-8", errors="replace")
                    import re as _re
                    m = _re.search(r"<title[^>]*>([^<]+)</title>", text, _re.I)
                    if m:
                        title = m.group(1).strip()
                except Exception:
                    pass
                html_pages.append({"source_path": rel, "title": title})

        captions_rel = None
        cap_path = run_dir / "demo_captions.jsonl"
        if cap_path.exists() and cap_path.stat().st_size > 0:
            captions_rel = "demo_captions.jsonl"

        available_assets = {
            "recording": recording_meta,
            "hyperframes": hyperframes,
            "html_pages": html_pages,
            "captions_path": captions_rel,
        }

        # M3a · cutting plan
        plan_path = run_dir / "cutting_plan.json"
        await asyncio.to_thread(
            run_cutting_planner,
            run_dir=run_dir, project_brief=brief,
            available_assets=available_assets, output_path=plan_path,
            progress_path=run_dir / "progress.json",
            feedback=feedback,
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        _fire_quality_judge(pipe, "cutting_plan")

        # M3b · codegen
        num, den = recording_meta["fps"].split("/")
        src_fps = max(1, round(int(num) / int(den)))
        remotion_dir = run_dir / "remotion"
        await asyncio.to_thread(generate_project, plan, remotion_dir, recording,
                                 src_fps=src_fps,
                                 hyperframes_dir=(hyperframes_dir if hyperframes else None),
                                 html_dir=(html_dir if html_pages else None))

        # M3b · npm install (skipped if package-lock.json + node_modules exist)
        if not (remotion_dir / "node_modules").exists():
            await asyncio.to_thread(npm_install, remotion_dir, timeout=600)

        # M3b · render
        out = run_dir / "outputs" / "v1.mp4"
        await asyncio.to_thread(render, remotion_dir, output_path=out, timeout=900)

        pipe.record_asset("v1_video", out, verified=True)
        pipe.transition(phase=3, gate="waiting_v1_review")
    except Exception as e:
        pipe.log.exception(f"phase3 failed: {e}")
        pipe.bus.emit("asset_failed", agent="phase3", error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=3, agent="Agent 3 RemotionComposer",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


@app.post("/runs/{project}/{run_id}/skip_opendesign", response_class=HTMLResponse)
async def skip_opendesign(project: str, run_id: str):
    """User explicit opt-out — advance to Phase 3 without HTML hero.

    Default behavior is opposite: after recording, OpenDesigner runs to
    produce HTML for the Remotion compose step. If the user doesn't want
    that (faster, recording-only video), they click "取消 OpenDesign" and
    we transition directly to phase=3.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 2 or pipe.state.gate != "waiting_html":
        raise HTTPException(
            400,
            f"skip_opendesign requires phase=2/waiting_html, current={pipe.state.phase}/{pipe.state.gate}",
        )
    pipe.bus.emit("user_input", agent="pipeline",
                   action="skip_opendesign", phase=2)
    pipe.transition(phase=3, gate="running")
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        '<div class="text-amber-300 text-sm py-2">⏭ 跳过 OpenDesign · 直接进 Phase 3（只用录屏）</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


@app.post("/runs/{project}/{run_id}/accept_opendesign", response_class=HTMLResponse)
async def accept_opendesign(project: str, run_id: str):
    """User done iterating with OpenDesigner — advance to Phase 3.

    Requires html_asset/*.html to exist (i.e., user already clicked
    "采纳为 Hero" in the /opendesign tab, which copies the HTML into
    html_asset/). If not adopted yet, returns 400 with a hint.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 2 or pipe.state.gate != "waiting_html":
        raise HTTPException(
            400,
            f"accept_opendesign requires phase=2/waiting_html, current={pipe.state.phase}/{pipe.state.gate}",
        )
    # OpenDesign can produce EITHER static_hero (html_asset/*.html) OR
    # motion_film (hero/intro.mp4 or hyperframes/*.mp4). Accept any of those —
    # the agent chose the mode that fit the brief.
    html_dir = pipe.run_dir / "html_asset"
    html_files = list(html_dir.glob("*.html")) if html_dir.exists() else []
    hero_mp4 = pipe.run_dir / "hero" / "intro.mp4"
    hyper_dir = pipe.run_dir / "hyperframes"
    hyper_mp4s = list(hyper_dir.glob("*.mp4")) if hyper_dir.exists() else []
    n_assets = len(html_files) + (1 if hero_mp4.exists() else 0) + len(hyper_mp4s)
    if n_assets == 0:
        raise HTTPException(
            400,
            "no OpenDesign asset adopted yet — 到 /opendesign tab 点 '采纳为 Hero' "
            "（mp4 落在 hero/intro.mp4，html 落在 html_asset/）再回来",
        )
    summary = (
        f"{len(html_files)} HTML"
        + (f", hero mp4" if hero_mp4.exists() else "")
        + (f", {len(hyper_mp4s)} hyperframe mp4" if hyper_mp4s else "")
    )
    pipe.bus.emit("user_input", agent="pipeline",
                   action="accept_opendesign",
                   html_count=len(html_files),
                   hero_mp4=hero_mp4.exists(),
                   hyperframes_count=len(hyper_mp4s))
    pipe.transition(phase=3, gate="running")
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        f'<div class="text-emerald-400 text-sm py-2">✅ 已采纳 OpenDesign ({summary}) → Phase 3</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


# ────────────────────────────────────────────────────────────
# OpenDesign Critic — Agent 6b
# ────────────────────────────────────────────────────────────

@app.post("/runs/{project}/{run_id}/opendesign/critic_loop",
           response_class=HTMLResponse)
async def opendesign_critic_loop_route(project: str, run_id: str,
                                        user_hint: str = Form(""),
                                        soft_max: int = Form(5)):
    """Fire a fresh critic loop in the background.

    Use cases (all routed here):
      - 🔄 让 critic 再改一轮  → no user_hint
      - 📝 我说一句              → with user_hint set
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    if pipe.state.phase != 2 or pipe.state.gate != "waiting_html":
        raise HTTPException(
            400,
            f"opendesign critic_loop requires phase=2/waiting_html, "
            f"current={pipe.state.phase}/{pipe.state.gate}",
        )

    async def _go():
        try:
            from ..agents.od_critic import run_critic_loop
            await asyncio.to_thread(
                run_critic_loop, pipe.run_dir,
                int(soft_max), (user_hint.strip() or None),
            )
        except Exception as e:
            pipe.log.exception(f"critic loop failed: {e}")
    asyncio.create_task(_go())

    msg = "🎯 critic loop 再启 (no hint)" if not user_hint.strip() else f"📝 critic loop 启 with hint: {user_hint[:80]}"
    return HTMLResponse(
        f'<div class="text-amber-300 text-sm py-2">{msg} · 看下方进度</div>',
        headers={"HX-Trigger": "opendesign-critic-refresh"},
    )


@app.post("/runs/{project}/{run_id}/opendesign/satisfied",
           response_class=HTMLResponse)
async def opendesign_satisfied_route(project: str, run_id: str):
    """User accepts the current artifact (critic loop result OR raw)
    and advances to Phase 3. Delegates to accept_opendesign behavior."""
    return await accept_opendesign(project, run_id)


@app.get("/runs/{project}/{run_id}/opendesign/critic_state")
async def opendesign_critic_state_route(project: str, run_id: str):
    """Return the current critic loop progress (raw JSON, for polling)."""
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.od_critic import read_progress
    data = read_progress(pipe.run_dir)
    return data or {"stage": "not_started"}


@app.get("/runs/{project}/{run_id}/opendesign/critic_panel",
          response_class=HTMLResponse)
async def opendesign_critic_panel_route(project: str, run_id: str,
                                         request: Request):
    """Inner panel for the OpenDesign pending screen — shows critic state
    + history + 3 user buttons. Polled by HTMX every ~3s."""
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.od_critic import read_progress
    from ..agents.opendesigner import load_session
    crit = read_progress(pipe.run_dir) or {"stage": "not_started"}
    sess = load_session(pipe.run_dir)
    artifact_kind = crit.get("final_artifact_kind") or ""
    artifact_path = crit.get("final_artifact") or ""
    # if loop still running, peek at most recent iteration's artifact_path
    if not artifact_path and crit.get("iterations"):
        last = crit["iterations"][-1]
        artifact_kind = last.get("artifact_kind", "")
        artifact_path = last.get("artifact_path", "")
    return TEMPLATES.TemplateResponse(request, "_phase_2_od_critic.html", {
        "project": project, "run_id": run_id,
        "session": sess.__dict__ if sess else None,
        "crit": crit,
        "artifact_kind": artifact_kind,
        "artifact_path": artifact_path,
    })


@app.post("/runs/{project}/{run_id}/accept_phase3", response_class=HTMLResponse)
async def accept_phase3(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 3:
        raise HTTPException(400, f"accept_phase3 requires phase=3, current={pipe.state.phase}")
    v1 = pipe.run_dir / "outputs" / "v1.mp4"
    if not v1.exists():
        raise HTTPException(400, "outputs/v1.mp4 not produced yet — run Phase 3 first")
    pipe.transition(phase=4, gate="running")
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        '<div class="text-emerald-400 text-sm">✅ Phase 3 通过 → Phase 4 · 跳转中...</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


# ────────────────────────────────────────────────────────────
# Phase 4 · BGM (scaffold + musicgen + mux)
# ────────────────────────────────────────────────────────────

@app.post("/runs/{project}/{run_id}/run_phase4", response_class=HTMLResponse)
async def run_phase4_route(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    if REGISTRY.is_running(run_id):
        return HTMLResponse(
            '<div class="text-amber-300 text-sm py-2 px-3 bg-amber-500/10 border border-amber-500/30 rounded">'
            '⏳ 已经在跑 — 等当前 phase 完成再触发（MiniMax 云端有时要 2-3 分钟）'
            '</div>', status_code=409,
        )
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    import time as _time
    (pipe.run_dir / "phase4_started.txt").write_text(str(_time.time()), encoding="utf-8")
    asyncio.create_task(_run_phase4_async(pipe))
    return HTMLResponse(
        '<div class="text-amber-300 text-sm py-2 px-3 bg-amber-500/10 border border-amber-500/30 rounded">'
        '⏳ Phase 4 启动中 · 面板会自动刷新显示进度（~1-3 min CUDA）'
        '</div>',
        headers={"HX-Trigger": "phase4-refresh"},
    )


@app.get("/runs/{project}/{run_id}/phase4_panel", response_class=HTMLResponse)
async def phase4_panel(project: str, run_id: str, request: Request):
    """Live Phase-4 panel — polled every 3s by _phase_4.html.

    Surfaces: is_running flag, elapsed time, BGM scaffold + MusicGen status,
    v1_bgm_final.mp4 existence. Same UX pattern as phase3_panel.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    run_dir = pipe.run_dir
    is_running = REGISTRY.is_running(run_id)

    started_at = None
    started_path = run_dir / "phase4_started.txt"
    if started_path.exists():
        try:
            started_at = float(started_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    import time as _time
    elapsed_s = int(_time.time() - started_at) if started_at else None

    bgm_musicgen = run_dir / "bgm" / "musicgen.wav"  # MusicGen output
    bgm_minimax = run_dir / "bgm" / "minimax.wav"  # MiniMax output
    bgm_final = run_dir / "bgm" / "bgm_final.wav"  # whichever path produced it
    v1_bgm = run_dir / "outputs" / "v1_bgm_final.mp4"
    v1 = run_dir / "outputs" / "v1.mp4"

    return TEMPLATES.TemplateResponse(request, "_phase_4_panel.html", {
        "project": project, "run_id": run_id,
        "is_running": is_running,
        "elapsed_s": elapsed_s,
        "musicgen_exists": bgm_musicgen.exists() or bgm_final.exists(),
        "minimax_exists": bgm_minimax.exists() or bgm_final.exists(),
        "v1_bgm_exists": v1_bgm.exists(),
        "v1_bgm_size_mb": round(v1_bgm.stat().st_size / 1024 / 1024, 2) if v1_bgm.exists() else 0,
        "v1_exists": v1.exists(),
        "last_error": pipe.state.last_error,
        "state_gate": pipe.state.gate,
    })


async def _run_phase4_async(pipe: Pipeline) -> None:
    await REGISTRY.mark_running(pipe.run_id, "phase4")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        # Phase 4 = call BGM generator directly, no python-side scaffolding.
        # The old M4a step synthesized a numpy kick/snare/hihat track to feed
        # into musicgen-melody as a melody guide; that approach is gone.
        # MiniMax music-2.6 (the default path) ignored the scaffold anyway,
        # and forcing the result toward a 4-on-the-floor pattern made every
        # video sound like a stock demo. Now: prompt → BGM, that's it.
        from ..tools.bgm_musicgen import generate_bgm, build_prompt_from_brief
        from ..tools.bgm_minimax import generate_bgm_minimax, has_minimax_key
        from ..tools.bgm_mux import mux_bgm
        import tempfile

        run_dir = pipe.run_dir
        plan = json.loads((run_dir / "cutting_plan.json").read_text(encoding="utf-8"))
        brief = (run_dir / "project_brief.md").read_text(encoding="utf-8")

        # Compute target BGM duration from cutting_plan (sum of scene durations
        # minus the 15-frame crossfade overlap between consecutive scenes).
        fps = plan.get("fps", 30)
        scenes = plan.get("scenes", [])
        total_frames = 0
        for i, sc in enumerate(scenes):
            total_frames += int(sc["duration_s"] * fps)
            if i < len(scenes) - 1:
                total_frames -= 15
        total_s = total_frames / fps if total_frames > 0 else 30.0

        bgm_dir = run_dir / "bgm"
        bgm_dir.mkdir(parents=True, exist_ok=True)
        bgm_final = bgm_dir / "bgm_final.wav"
        prompt = build_prompt_from_brief(brief, bpm=120)
        used_minimax = False
        if has_minimax_key():
            try:
                await asyncio.to_thread(
                    generate_bgm_minimax,
                    None, bgm_final, prompt, total_s,
                )
                used_minimax = True
            except Exception as mm_err:
                pipe.log.warning(f"MiniMax BGM failed, falling back to MusicGen: {mm_err}")

        if not used_minimax:
            local_model = str(Path(tempfile.gettempdir()) / "musicgen-small-local")
            if Path(local_model).exists():
                model_arg = local_model
            else:
                model_arg = "facebook/musicgen-small"
            # use_melody=False — prompt-only generation (no scaffold input).
            await asyncio.to_thread(
                generate_bgm,
                None, bgm_final, prompt,
                total_s, model_arg, None, False,
            )

        # M4c · mux
        v1 = run_dir / "outputs" / "v1.mp4"
        v1_bgm = run_dir / "outputs" / "v1_bgm_final.mp4"
        await asyncio.to_thread(mux_bgm, v1, bgm_final, v1_bgm, 0.7)

        pipe.record_asset("v1_bgm_final", v1_bgm, verified=True)
        _fire_metrics_only(pipe, 4)
        pipe.transition(phase=4, gate="waiting_bgm_review")
    except Exception as e:
        pipe.log.exception(f"phase4 failed: {e}")
        pipe.bus.emit("asset_failed", agent="phase4", error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=4, agent="Agent 4 BGMComposer",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


@app.post("/runs/{project}/{run_id}/accept_phase4", response_class=HTMLResponse)
async def accept_phase4(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 4:
        raise HTTPException(400, f"accept_phase4 requires phase=4, current={pipe.state.phase}")
    v1bgm = pipe.run_dir / "outputs" / "v1_bgm_final.mp4"
    if not v1bgm.exists():
        raise HTTPException(400, "outputs/v1_bgm_final.mp4 not produced yet — run Phase 4 first")
    pipe.transition(phase=5, gate="running")
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        '<div class="text-emerald-400 text-sm">✅ Phase 4 通过 → Phase 5 · 跳转中...</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


# ────────────────────────────────────────────────────────────
# Phase 5 · VoiceOver (script + tts + timeline + ducking)
# ────────────────────────────────────────────────────────────

@app.post("/runs/{project}/{run_id}/run_phase5", response_class=HTMLResponse)
async def run_phase5_route(project: str, run_id: str, lang: str = Form("zh-CN")):
    pipe = REGISTRY.get_or_load(project, run_id)
    if REGISTRY.is_running(run_id):
        return HTMLResponse(
            '<div class="text-amber-300 text-sm py-2 px-3 bg-amber-500/10 border border-amber-500/30 rounded">'
            '⏳ 已经在跑 — 等当前 phase 完成（edge-tts 大约 30-60s）'
            '</div>', status_code=409,
        )
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    import time as _time
    (pipe.run_dir / "phase5_started.txt").write_text(str(_time.time()), encoding="utf-8")
    (pipe.run_dir / "phase5_running_lang.txt").write_text(lang, encoding="utf-8")
    asyncio.create_task(_run_phase5_async(pipe, lang))
    return HTMLResponse(
        f'<div class="text-amber-300 text-sm py-2 px-3 bg-amber-500/10 border border-amber-500/30 rounded">'
        f'⏳ Phase 5 启动中 · {lang} · 面板会自动刷新显示进度（~30-60s）'
        f'</div>',
        headers={"HX-Trigger": "phase5-refresh"},
    )


@app.get("/runs/{project}/{run_id}/phase5_panel", response_class=HTMLResponse)
async def phase5_panel(project: str, run_id: str, request: Request):
    """Live Phase-5 panel — polled every 3s by _phase_5.html."""
    pipe = REGISTRY.get_or_load(project, run_id)
    run_dir = pipe.run_dir
    is_running = REGISTRY.is_running(run_id)

    started_at = None
    started_path = run_dir / "phase5_started.txt"
    if started_path.exists():
        try:
            started_at = float(started_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    import time as _time
    elapsed_s = int(_time.time() - started_at) if started_at else None

    running_lang = None
    lang_path = run_dir / "phase5_running_lang.txt"
    if lang_path.exists():
        running_lang = lang_path.read_text(encoding="utf-8").strip()

    voice_dir = run_dir / "voice"
    bilingual = voice_dir / "voiceover_script_bilingual.json"
    final_zh = run_dir / "outputs" / "final_zh-CN.mp4"
    final_en = run_dir / "outputs" / "final_en-US.mp4"

    # Per-language step states — count rendered segment files for "tts" step
    zh_seg = voice_dir / "per_segment_zh-CN"
    en_seg = voice_dir / "per_segment_en-US"
    zh_full = voice_dir / "voice_full_zh-CN.wav"
    en_full = voice_dir / "voice_full_en-US.wav"

    def _segs(d): return len(list(d.glob("*.mp3")) + list(d.glob("*.wav"))) if d.exists() else 0

    v1_bgm = run_dir / "outputs" / "v1_bgm_final.mp4"
    v1 = run_dir / "outputs" / "v1.mp4"

    return TEMPLATES.TemplateResponse(request, "_phase_5_panel.html", {
        "project": project, "run_id": run_id,
        "is_running": is_running,
        "elapsed_s": elapsed_s,
        "running_lang": running_lang,
        "bilingual_exists": bilingual.exists(),
        "zh_segments": _segs(zh_seg),
        "en_segments": _segs(en_seg),
        "zh_full_exists": zh_full.exists(),
        "en_full_exists": en_full.exists(),
        "final_zh_exists": final_zh.exists(),
        "final_en_exists": final_en.exists(),
        "zh_size_mb": round(final_zh.stat().st_size / 1024 / 1024, 2) if final_zh.exists() else 0,
        "en_size_mb": round(final_en.stat().st_size / 1024 / 1024, 2) if final_en.exists() else 0,
        "v1_bgm_exists": v1_bgm.exists(),
        "last_error": pipe.state.last_error,
        "state_gate": pipe.state.gate,
    })


async def _run_phase5_async(pipe: Pipeline, lang: str = "zh-CN") -> None:
    await REGISTRY.mark_running(pipe.run_id, "phase5")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        # Strip system proxies so edge-tts wss isn't intercepted by VPN MITM
        import os
        for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","all_proxy"):
            os.environ.pop(k, None)
        os.environ["NO_PROXY"] = "*"

        from ..agents.voice_over import propose_script
        from ..tools.tts_edge import synth_script
        from ..tools.voice_timeline import assemble_timeline
        from ..tools.bgm_duck_mux import duck_and_mux

        run_dir = pipe.run_dir
        plan = json.loads((run_dir / "cutting_plan.json").read_text(encoding="utf-8"))
        brief = (run_dir / "project_brief.md").read_text(encoding="utf-8")
        bgm_video = run_dir / "outputs" / "v1_bgm_final.mp4"
        if not bgm_video.exists():
            bgm_video = run_dir / "outputs" / "v1.mp4"

        voice_dir = run_dir / "voice"
        bilingual = voice_dir / "voiceover_script_bilingual.json"
        if not bilingual.exists():
            await asyncio.to_thread(propose_script, run_dir, brief, plan, bilingual)
            # Fire all 4 voiceover-related judges + cross-phase consistency.
            # propose_script writes bilingual.json AND per-language splits,
            # so all 3 voice artifacts exist at this point.
            _fire_quality_judge(pipe, "voiceover_script")
            _fire_quality_judge(pipe, "voiceover_zh")
            _fire_quality_judge(pipe, "voiceover_en")
            _fire_cross_phase_judge(pipe)

        per = voice_dir / f"voiceover_script_{lang}.json"
        seg_dir = voice_dir / f"per_segment_{lang}"
        voice_full = voice_dir / f"voice_full_{lang}.wav"
        final = run_dir / "outputs" / f"final_{lang}.mp4"

        # Step 2 · TTS with retry (bing.com flaky on this network)
        import time as _t, shutil as _sh
        last_sr = None
        last_err = None
        for attempt in range(5):
            if seg_dir.exists():
                _sh.rmtree(seg_dir)
            try:
                last_sr = await asyncio.to_thread(synth_script, per, seg_dir)
                break
            except Exception as e:
                last_err = e
                pipe.log.warning(f"tts attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:120]}")
                _t.sleep(15)
        if last_sr is None:
            raise RuntimeError(f"TTS still failing after 5 retries: {last_err}")

        # Step 3 · timeline
        await asyncio.to_thread(assemble_timeline, last_sr, bgm_video, voice_full)

        # Step 4 · ducking + mux
        await asyncio.to_thread(duck_and_mux, bgm_video, voice_full, per, final, 0.7, 0.3)

        pipe.record_asset(f"final_{lang}", final, verified=True)
        pipe.transition(phase=5, gate="done")
    except Exception as e:
        pipe.log.exception(f"phase5 failed: {e}")
        pipe.bus.emit("asset_failed", agent="phase5", error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=5, agent="Agent 5 VoiceOver",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


def _cli_state(run_dir: Path) -> dict:
    """Aggregate CLI-recorder state for templates: planned_command + recording presence."""
    plan_path = run_dir / "setup_plan.json"
    cli_rec = run_dir / "recordings" / "cli_recording.mp4"
    planned_cmd = ""
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            services = plan.get("services") or []
            if services:
                planned_cmd = services[0].get("command", "")
        except Exception:
            pass
    cli_meta = None
    if cli_rec.exists():
        # Lightweight: just read fs stats; the actual ffprobe was done at record-time
        cli_meta = {
            "video_size_bytes": cli_rec.stat().st_size,
            "video_duration_s": 0,  # filled by ffprobe at render time; UI shows N/A if 0
        }
    return {
        "planned_command": planned_cmd,
        "cli_recording_exists": cli_rec.exists(),
        "cli_meta": cli_meta,
    }


@app.post("/runs/{project}/{run_id}/record_cli", response_class=HTMLResponse)
async def record_cli_route(project: str, run_id: str,
                            duration_s: float = Form(30.0)):
    """Run the planned service command + capture stdout → terminal-style mp4."""
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    plan_path = pipe.run_dir / "setup_plan.json"
    if not plan_path.exists():
        return HTMLResponse('<div class="text-rose-400 text-sm">缺 setup_plan.json</div>', 500)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    services = plan.get("services") or []
    if not services:
        return HTMLResponse('<div class="text-rose-400 text-sm">setup_plan.json 没有 service 命令可跑</div>', 400)
    s = services[0]
    cmd_str = s.get("command", "")
    cwd_rel = s.get("cwd", "./")
    if not cmd_str:
        return HTMLResponse('<div class="text-rose-400 text-sm">service.command 空</div>', 400)
    import shlex
    cmd_list = shlex.split(cmd_str)
    # Resolve cwd relative to repo dir
    repo_dir = pipe.run_dir / "repo"
    cwd_abs = (repo_dir / cwd_rel).resolve() if cwd_rel != "./" else repo_dir

    asyncio.create_task(_record_cli_async(pipe, cmd_list, cwd_abs, duration_s))
    return HTMLResponse(
        f'<div class="text-amber-300 text-sm py-2">⏳ 跑 <code>{cmd_str}</code> 在 <code>{cwd_abs.name}</code> 录屏 {duration_s}s（pyte 模拟终端 + PIL 渲染）...</div>',
        headers={"HX-Trigger": "phase2-refresh"},
    )


async def _record_cli_async(pipe: Pipeline, cmd: list, cwd: Path, duration_s: float) -> None:
    await REGISTRY.mark_running(pipe.run_id, "phase2b-cli-record")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..tools.cli_recorder import record_cli
        out = pipe.run_dir / "recordings" / "cli_recording.mp4"
        await asyncio.to_thread(
            record_cli, cmd, cwd, out, duration_s, 15, 1920, 1080,
        )
    except Exception as e:
        pipe.log.exception(f"record_cli failed: {e}")
        pipe.bus.emit("asset_failed", agent="cli_recorder",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="CLI Recorder",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


@app.post("/runs/{project}/{run_id}/accept_cli", response_class=HTMLResponse)
async def accept_cli_route(project: str, run_id: str):
    """Accept cli_recording.mp4 → recordings/test.mp4."""
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 2:
        raise HTTPException(400, f"accept_cli requires phase=2, current={pipe.state.phase}")
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    rec_dir = pipe.run_dir / "recordings"
    src = rec_dir / "cli_recording.mp4"
    if not src.exists():
        return HTMLResponse('<div class="text-rose-400 text-sm">cli_recording.mp4 不存在</div>', 400)
    target = rec_dir / "test.mp4"
    import shutil as _sh
    await asyncio.to_thread(_sh.copy2, src, target)
    pipe.record_asset("recording_test", target, verified=True, source="cli_recorder")
    pipe.bus.emit("asset_verified", agent="pipeline", name="recording_test", path=str(target))
    # Write accepted_window.json so phase2_panel doesn't keep serving the
    # stale "窗口对，进 M2c" button.
    (pipe.run_dir / "accepted_window.json").write_text(json.dumps({
        "source": "cli_recorder",
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "test_recording": str(target),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # Route through OpenDesigner before Phase 3 — see accept_demo for rationale.
    pipe.transition(phase=2, gate="waiting_html")
    _fire_opendesign_bootstrap(pipe)
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        '<div class="text-emerald-400 text-sm py-2">✅ 已采纳 · 启动 OpenDesigner...</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


# ────────────────────────────────────────────────────────────
# Demo Driver — autonomous-agent project demonstrator (Phase 2c)
# Replaces the old fixed-duration cli/web recorder.
# ────────────────────────────────────────────────────────────

def _driver_context(run_dir: Path) -> dict:
    """Read demo_driver_progress.json + check demo.mp4 for template context."""
    p = run_dir / "demo_driver_progress.json"
    progress: dict = {"status": "not_started"}
    if p.exists():
        try:
            progress = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            progress = {"status": "error", "error": "progress.json unparseable"}
    return {
        "driver_progress": progress,
        "demo_recording_exists": (run_dir / "recordings" / "demo.mp4").exists(),
    }


def _read_demo_targets(plan: dict) -> tuple[str, Optional[str], Optional[list], Optional[str]]:
    """Read demo mode + targets from the plan the Planner agent already decided.

    IRON LAW: NO state machine here. The Planner explicitly sets
    `demo_mode` ('web'|'cli') in setup_plan.json — we just respect it.
    Returns (mode, web_url, cli_command_argv, cli_cwd).
    """
    import shlex
    from urllib.parse import urlparse

    mode = plan.get("demo_mode") or ""
    if mode == "web":
        # Take the first service's health_url as the web entry point.
        services = plan.get("services") or []
        if services:
            url = services[0].get("health_url") or ""
            if url.startswith("http://") or url.startswith("https://"):
                p = urlparse(url)
                return "web", f"{p.scheme}://{p.netloc}/", None, None
        return "web", None, None, None
    if mode == "cli":
        cmd_str = plan.get("demo_command") or ""
        cwd = plan.get("demo_cwd") or "./"
        argv = shlex.split(cmd_str) if cmd_str.strip() else None
        return "cli", None, argv, cwd
    # Plan is missing demo_mode (older plan written before the field existed).
    # Don't guess — return empty mode; the route will tell the user to either
    # re-plan or pick a mode manually.
    return "", None, None, None


@app.post("/runs/{project}/{run_id}/run_demo_driver", response_class=HTMLResponse)
async def run_demo_driver_route(project: str, run_id: str,
                                  mode: Optional[str] = Form(None),
                                  web_url: Optional[str] = Form(None),
                                  cli_command: Optional[str] = Form(None),
                                  cli_cwd: Optional[str] = Form(None)):
    """Launch the Demo Driver agent as a background task.

    Form fields are optional; missing fields are derived from setup_plan.json.
    The agent operates the project autonomously (read source, decide what to
    demo, drive the running app via browser/pty, emit captions, listen to
    user feedback) until it calls finish_demo. NO duration cap.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    if REGISTRY.is_running(run_id):
        return HTMLResponse(
            '<div class="text-amber-400 text-sm">已有 Agent 在跑，等它结束</div>', 409)

    plan_path = pipe.run_dir / "setup_plan.json"
    plan = {}
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            plan = {}

    # Read mode + targets from the plan the Planner agent already picked.
    # The form fields (mode / web_url / cli_command / cli_cwd) are still honored
    # if the user wants to override, but we never AUTO-INFER mode anymore.
    plan_mode, plan_url, plan_cmd, plan_cwd = _read_demo_targets(plan)
    chosen_mode = mode or plan_mode
    chosen_url = web_url or plan_url
    if cli_command:
        import shlex
        chosen_cmd = shlex.split(cli_command)
    else:
        chosen_cmd = plan_cmd
    chosen_cwd = cli_cwd or plan_cwd or "./"

    if not chosen_mode:
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">setup_plan.json 没有 <code>demo_mode</code> 字段——这是用 旧版 Planner 跑的 plan。重新跑 Phase 2 plan 就好。</div>', 400)
    if chosen_mode == "web" and not chosen_url:
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">web 模式缺 URL（setup_plan.services 为空且未在表单填写）</div>', 400)
    if chosen_mode == "cli" and not chosen_cmd:
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">cli 模式缺 command（setup_plan.demo_command 为空且未在表单填写）</div>', 400)

    # Read project_brief for tone context (NOT a checklist for demo content)
    brief_path = pipe.run_dir / "project_brief.md"
    brief = brief_path.read_text(encoding="utf-8") if brief_path.exists() else ""

    repo_dir = pipe.run_dir / "repo"

    # Reset live_feedback so the new run starts with a clean channel
    fb_path = pipe.run_dir / "live_feedback.jsonl"
    fb_path.write_text("", encoding="utf-8")

    asyncio.create_task(_run_demo_driver_async(
        pipe=pipe, repo_dir=repo_dir, project_brief=brief,
        mode=chosen_mode, web_url=chosen_url,
        cli_command=chosen_cmd,
        cli_cwd=(repo_dir / chosen_cwd).resolve() if chosen_mode == "cli" else None,
    ))

    label = (f"web → {chosen_url}" if chosen_mode == "web"
             else f"cli → {' '.join(chosen_cmd or [])}")
    return HTMLResponse(
        f'<div class="text-amber-300 text-sm py-2">'
        f'⏳ Demo Driver 启动中 · {label} · 演完才停（无时长 cap）</div>',
        headers={"HX-Trigger": "phase2-refresh"},
    )


async def _run_demo_driver_async(*, pipe: Pipeline, repo_dir: Path,
                                  project_brief: str,
                                  mode: str, web_url: Optional[str],
                                  cli_command: Optional[list],
                                  cli_cwd: Optional[Path]) -> None:
    await REGISTRY.mark_running(pipe.run_id, "demo-driver")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..agents.demo_driver import run_demo_driver
        await asyncio.to_thread(
            run_demo_driver,
            run_dir=pipe.run_dir,
            repo_dir=repo_dir,
            project_brief=project_brief,
            mode=mode,
            web_url=web_url,
            cli_command=cli_command,
            cli_cwd=cli_cwd,
        )
    except Exception as e:
        pipe.log.exception(f"demo_driver failed: {e}")
        pipe.bus.emit("asset_failed", agent="demo_driver",
                       error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="Demo Driver",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


@app.post("/runs/{project}/{run_id}/demo_driver/feedback", response_class=HTMLResponse)
async def demo_driver_feedback_route(project: str, run_id: str,
                                       text: str = Form(...)):
    """Append a live-feedback entry the running driver agent will see on its next turn."""
    pipe = REGISTRY.get_or_load(project, run_id)
    if not text.strip():
        return HTMLResponse('<div class="text-slate-500 text-xs">空消息忽略</div>', 200)
    fb_path = pipe.run_dir / "live_feedback.jsonl"
    fb_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"text": text.strip(),
             "ts": datetime.now(timezone.utc).isoformat()}
    with fb_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    pipe.bus.emit("user_input", agent="demo_driver", text=text.strip()[:500])
    return HTMLResponse(
        '<div class="text-emerald-400 text-xs py-1">✓ 已发送给 Driver（下一轮可见）</div>')


@app.get("/runs/{project}/{run_id}/demo_driver/progress.json")
async def demo_driver_progress(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    p = pipe.run_dir / "demo_driver_progress.json"
    if not p.exists():
        return {"phase": "2c-demo-driver", "status": "not_started"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"phase": "2c-demo-driver", "status": "error", "error": str(e)}


@app.get("/runs/{project}/{run_id}/demo_driver/captions")
async def demo_driver_captions(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    p = pipe.run_dir / "demo_captions.jsonl"
    if not p.exists():
        return {"captions": []}
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return {"captions": out}


@app.get("/runs/{project}/{run_id}/demo_driver/feedback_log")
async def demo_driver_feedback_log(project: str, run_id: str):
    pipe = REGISTRY.get_or_load(project, run_id)
    p = pipe.run_dir / "live_feedback.jsonl"
    if not p.exists():
        return {"feedback": []}
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return {"feedback": out}


@app.post("/runs/{project}/{run_id}/accept_demo", response_class=HTMLResponse)
async def accept_demo_route(project: str, run_id: str):
    """Accept Demo Driver's recordings/demo.mp4 → recordings/test.mp4 and advance to Phase 3."""
    pipe = REGISTRY.get_or_load(project, run_id)
    if pipe.state.phase != 2:
        raise HTTPException(400, f"accept_demo requires phase=2, current={pipe.state.phase}")
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    rec_dir = pipe.run_dir / "recordings"
    src = rec_dir / "demo.mp4"
    if not src.exists():
        return HTMLResponse('<div class="text-rose-400 text-sm">demo.mp4 不存在（Driver 还没跑完）</div>', 400)
    target = rec_dir / "test.mp4"
    import shutil as _sh
    await asyncio.to_thread(_sh.copy2, src, target)
    pipe.record_asset("recording_test", target, verified=True, source="demo_driver")
    # Captions go forward as a separate asset for Phase 3 to consume
    cap_src = pipe.run_dir / "demo_captions.jsonl"
    if cap_src.exists():
        pipe.record_asset("demo_captions", cap_src, verified=True, source="demo_driver")
        # Score the demo captions before they get folded into Phase 3 — gives
        # the user a quality signal on the demo BEFORE committing to Remotion.
        _fire_quality_judge(pipe, "demo_captions")
    # Write accepted_window.json so phase2_panel's stale-state heuristic
    # (test.mp4 exists && accepted_window.json missing → render M2b
    # test_done partial) doesn't keep serving the old "窗口对，进 M2c"
    # button after we've already moved on.
    accepted_path = pipe.run_dir / "accepted_window.json"
    accepted_path.write_text(json.dumps({
        "source": "demo_driver",
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "test_recording": str(target),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    pipe.bus.emit("asset_verified", agent="pipeline",
                   name="recording_test", path=str(target))
    # Phase 2 → Phase 3 transition now goes through OpenDesigner. Recording
    # is in hand; next step is HTML hero generation (default ON; user must
    # explicitly click "跳过 OpenDesign" to bypass). After user adopts the
    # HTML (or skips), the page advances to phase=3.
    pipe.transition(phase=2, gate="waiting_html")
    _fire_opendesign_bootstrap(pipe)
    target_url = f"/runs/{project}/{run_id}"
    return HTMLResponse(
        '<div class="text-emerald-400 text-sm py-2">✅ 已采纳 Demo · 启动 OpenDesigner...</div>',
        headers={"HX-Redirect": target_url, "Location": target_url},
    )


# ────────────────────────────────────────────────────────────
# OpenDesign (M2a · Agent 6) routes
# ────────────────────────────────────────────────────────────

@app.get("/runs/{project}/{run_id}/opendesign", response_class=HTMLResponse)
async def opendesign_page(project: str, run_id: str, request: Request):
    """Full-page OpenDesign tab: chat + iframe preview + adopt."""
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.opendesigner import load_session
    sess = load_session(pipe.run_dir)
    return TEMPLATES.TemplateResponse(
        request, "opendesign.html",
        {
            "project": project, "run_id": run_id,
            "session": sess.__dict__ if sess else None,
        },
    )


@app.post("/runs/{project}/{run_id}/opendesign/init", response_class=HTMLResponse)
async def opendesign_init(project: str, run_id: str):
    """One-time bootstrap: ensure daemon, LLM picks setup, create project."""
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    brief_path = pipe.run_dir / "project_brief.md"
    if not brief_path.exists():
        return HTMLResponse(
            '<div class="text-rose-400 text-sm">project_brief.md 不存在 — 先完成 Phase 1。</div>',
            status_code=400,
        )
    from ..agents.opendesigner import bootstrap
    from ..tools.opendesign_lifecycle import ensure_daemon

    try:
        endpoint = await asyncio.to_thread(ensure_daemon)
        brief = brief_path.read_text(encoding="utf-8")
        sess = await asyncio.to_thread(
            bootstrap, pipe.run_dir, endpoint, brief, f"{project}-promo",
        )
    except Exception as e:
        return HTMLResponse(
            f'<div class="text-rose-400 text-sm">init 失败: {type(e).__name__}: {e}</div>',
            status_code=500,
        )
    return HTMLResponse(
        '<div hx-get="/runs/{}/{}/opendesign" hx-trigger="load" hx-target="body" hx-swap="outerHTML"></div>'.format(
            project, run_id,
        ),
    )


@app.post("/runs/{project}/{run_id}/opendesign/iterate")
async def opendesign_iterate(project: str, run_id: str,
                              feedback: Optional[str] = Form(None),
                              first_turn: Optional[str] = Form(None)):
    """Stream SSE events from OpenDesign for one chat turn.

    Modes:
      - feedback="..." : user natural-language → Agent translates → OpenDesign
      - first_turn=true: Agent uses its stored initial_prompt (no user input)
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    from ..agents.opendesigner import iterate_stream

    # Capture run_context bits for thread (ContextVar doesn't inherit across threading.Thread)
    _run_id = pipe.run_id
    _bus = pipe.bus
    _run_dir = pipe.run_dir

    async def gen():
        loop = asyncio.get_running_loop()

        def producer(q: asyncio.Queue):
            # threading.Thread doesn't inherit ContextVars from the parent
            # asyncio context. Re-set run_context inside the thread so
            # iterate_stream can find bus + emit events.
            set_run_context(_run_id, _bus, _run_dir)
            try:
                if first_turn:
                    iterator = iterate_stream(pipe.run_dir)  # uses initial_prompt
                else:
                    iterator = iterate_stream(pipe.run_dir, raw_user_feedback=feedback or "")
                for evt in iterator:
                    asyncio.run_coroutine_threadsafe(q.put(evt), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    q.put({"event": "error", "data": {"message": f"{type(e).__name__}: {e}"}}), loop,
                )
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        q: asyncio.Queue = asyncio.Queue()
        import threading
        threading.Thread(target=producer, args=(q,), daemon=True).start()
        while True:
            evt = await q.get()
            if evt is None:
                break
            etype = evt.get("event", "message")
            payload = json.dumps(evt.get("data", {}), ensure_ascii=False)
            yield f"event: {etype}\ndata: {payload}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/runs/{project}/{run_id}/opendesign/artifacts")
async def opendesign_artifacts(project: str, run_id: str):
    """Return JSON: {primary_kind, primary_name, files: [...]}.

    Frontend uses this to decide whether to render an <iframe> (HTML preview)
    or a <video> tag (HyperFrames .mp4 preview).
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.opendesigner import list_artifacts
    data = await asyncio.to_thread(list_artifacts, pipe.run_dir)
    return data


@app.get("/runs/{project}/{run_id}/opendesign/preview")
async def opendesign_preview(project: str, run_id: str, file: Optional[str] = None):
    """Stream a project artifact from the OpenDesign daemon.

    Without ?file= : returns the primary artifact (smart-pick HTML or MP4).
    With ?file=name.mp4 : streams that specific file with proper Content-Type.
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    from ..agents.opendesigner import list_artifacts, load_session
    from ..tools.opendesign_client import read_artifact_bytes

    sess = load_session(pipe.run_dir)
    if sess is None:
        return HTMLResponse(
            '<!doctype html><html><body style="font-family:monospace;padding:2em;color:#888;background:#111;">'
            '<p>no OpenDesign session yet</p></body></html>',
            status_code=200,
        )

    if file is None:
        info = await asyncio.to_thread(list_artifacts, pipe.run_dir)
        file = info.get("primary_name")
        if file is None:
            return HTMLResponse(
                '<!doctype html><html><body style="font-family:monospace;padding:2em;color:#888;background:#111;">'
                '<p>no artifact yet — waiting for OpenCode to produce output...</p></body></html>',
                status_code=200,
            )

    try:
        raw = await asyncio.to_thread(
            read_artifact_bytes, sess.daemon_url, sess.project_id, file,
        )
    except Exception as e:
        return HTMLResponse(
            f'<!doctype html><html><body style="font-family:monospace;padding:2em;color:#888;background:#111;">'
            f'<p>preview unavailable: {type(e).__name__}: {e}</p></body></html>',
            status_code=200,
        )

    name_lower = file.lower()
    from fastapi.responses import Response
    if name_lower.endswith(".mp4"):
        return Response(content=raw, media_type="video/mp4")
    if name_lower.endswith(".webm"):
        return Response(content=raw, media_type="video/webm")
    if name_lower.endswith(".png"):
        return Response(content=raw, media_type="image/png")
    if name_lower.endswith((".jpg", ".jpeg")):
        return Response(content=raw, media_type="image/jpeg")
    # Default: HTML
    return HTMLResponse(content=raw)


@app.post("/runs/{project}/{run_id}/opendesign/adopt", response_class=HTMLResponse)
async def opendesign_adopt(project: str, run_id: str, as_role: str = Form("auto")):
    """Adopt OpenDesign artifact with routing.

    as_role:
      - "auto"  : session.mode decides (motion_film→final, static_hero→hero)
      - "hero"  : route to run_dir/hero/intro.mp4 (or html_asset/) for Phase 3
      - "final" : route to run_dir/outputs/final.mp4 (skip Phase 3-5, advance to done)
    """
    pipe = REGISTRY.get_or_load(project, run_id)
    set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
    from ..agents.opendesigner import adopt

    try:
        result = await asyncio.to_thread(adopt, pipe.run_dir, as_role)
    except Exception as e:
        return HTMLResponse(
            f'<div class="text-rose-400 text-sm">adopt 失败: {type(e).__name__}: {e}</div>',
            status_code=500,
        )

    actual_role = result.get("as_role", as_role)
    primary = result.get("primary_target", "")
    primary_kind = result.get("primary_kind", "?")
    sz = result.get("primary_bytes") or 0
    sz_str = f"{sz/1024/1024:.2f} MB" if sz else f"{result.get('primary_files', '?')} files"

    # If user picked "final" + got a video → skip Phase 3-5, advance state to done
    advance_msg = ""
    if actual_role == "final" and primary_kind == "video":
        try:
            pipe.transition(phase=5, gate="done")
            pipe.bus.emit("gate_pass", agent="pipeline", gate="adopt_final_skip_phase_3_5")
            advance_msg = " · 已直跳 phase=5 gate=done"
        except Exception as e:
            advance_msg = f" · ⚠ phase advance 失败: {e}"

    return HTMLResponse(
        f'<div class="text-emerald-400 text-sm">'
        f'✅ 已采纳为 <b>{actual_role}</b> ({primary_kind}) → <code>{primary}</code> · {sz_str}'
        f'{advance_msg}'
        f'</div>',
    )


# ────────────────────────────────────────────────────────────
# Background agent execution
# ────────────────────────────────────────────────────────────

async def _run_planner_async(pipe: Pipeline, feedback: str) -> None:
    """Phase 2a: invoke Agent 2 planner."""
    await REGISTRY.mark_running(pipe.run_id, "phase2-plan")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..agents.setup_runner import run_planner
        # delete any stale exec state since we're (re)drafting
        exec_path = pipe.run_dir / "setup_exec.json"
        if exec_path.exists():
            exec_path.unlink()
        plan_path = pipe.run_dir / "setup_plan.json"
        progress_path = pipe.run_dir / "progress.json"
        # load brief for context
        canonical_brief = pipe.run_dir / "project_brief.md"
        brief_text = canonical_brief.read_text(encoding="utf-8") if canonical_brief.exists() else None
        repo_dir = pipe.run_dir / "repo"
        await asyncio.to_thread(
            run_planner,
            repo_dir=repo_dir,
            output_path=plan_path,
            project_brief=brief_text,
            feedback=feedback or None,
            progress_path=progress_path,
        )
        pipe.record_asset("setup_plan", plan_path, verified=True)
        _fire_quality_judge(pipe, "setup_plan")
        pipe.transition(phase=2, gate="waiting_plan_approval")
    except Exception as e:
        pipe.log.exception(f"planner failed: {e}")
        pipe.bus.emit("asset_failed", agent="Agent 2 SetupRunner",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="Agent 2 SetupRunner",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


async def _record_test_async(pipe: Pipeline, window_title: str, duration_s: float) -> None:
    """Phase 2b: spawn ffmpeg gdigrab to capture the chosen window for ~30s."""
    await REGISTRY.mark_running(pipe.run_id, "phase2b-record-test")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..tools.recorder import record_window
        from ..tools.window_enum import bring_to_foreground

        # Force target window visible before ffmpeg starts capturing.
        # gdigrab captures real screen pixels — a hidden window means black frames.
        brought = await asyncio.to_thread(bring_to_foreground, window_title)
        pipe.log.info(f"bring_to_foreground({window_title!r}) → {brought}")
        await asyncio.sleep(0.7)  # let the window paint after foregrounding

        rec_dir = pipe.run_dir / "recordings"
        rec_dir.mkdir(parents=True, exist_ok=True)
        out_path = rec_dir / "test.mp4"
        state_path = rec_dir / "test_state.json"
        result = await asyncio.to_thread(
            record_window,
            window_title=window_title,
            duration_s=duration_s,
            output_path=out_path,
            state_path=state_path,
        )
        if result.status == "done":
            pipe.record_asset("recording_test", out_path, verified=True,
                              window_title=window_title,
                              duration_s=duration_s)
        else:
            pipe.bus.emit("asset_failed", agent="pipeline",
                          name="recording_test", error=result.error)
    except Exception as e:
        pipe.log.exception(f"record_test failed: {e}")
        pipe.bus.emit("asset_failed", agent="pipeline",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="Record Test",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


async def _execute_plan_async(pipe: Pipeline) -> None:
    """Phase 2a: host-side plan executor (no LLM)."""
    await REGISTRY.mark_running(pipe.run_id, "phase2-execute")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)
        from ..tools.plan_executor import execute_plan
        plan = json.loads((pipe.run_dir / "setup_plan.json").read_text(encoding="utf-8"))
        repo_dir = pipe.run_dir / "repo"
        state_path = pipe.run_dir / "setup_exec.json"
        result = await asyncio.to_thread(
            execute_plan,
            plan=plan, repo_dir=repo_dir,
            state_path=state_path,
            services_dir=pipe.run_dir,
        )
        if result.status == "ok":
            pipe.bus.emit("asset_verified", agent="pipeline",
                          name="services_healthy",
                          path=str(pipe.run_dir / "services.json"))
        else:
            pipe.bus.emit("asset_failed", agent="pipeline",
                          name="setup_exec", error=result.error)
    except Exception as e:
        pipe.log.exception(f"plan execution failed: {e}")
        pipe.bus.emit("asset_failed", agent="pipeline",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=2, agent="Plan Executor",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


async def _run_agent_for_phase(pipe: Pipeline, feedback: str,
                               mode: str = "standard") -> None:
    """Dispatch by current phase; each phase has its own Agent."""
    await REGISTRY.mark_running(pipe.run_id, f"phase{pipe.state.phase}")
    try:
        set_run_context(pipe.run_id, pipe.bus, pipe.run_dir)

        if pipe.state.phase == 1:
            from ..agents.project_analyzer import run_project_analyzer
            repo_dir = pipe.run_dir / "repo"
            briefs_dir = pipe.run_dir / "briefs"
            briefs_dir.mkdir(exist_ok=True)
            brief_path = briefs_dir / f"{mode}.md"
            progress_path = pipe.run_dir / "progress.json"
            previous = brief_path.read_text(encoding="utf-8") if brief_path.exists() else None
            repo_url = pipe.state.manifest.get("repo", {}).get("url", "")
            await asyncio.to_thread(
                run_project_analyzer,
                repo_dir=repo_dir, repo_url=repo_url,
                output_path=brief_path,
                feedback=feedback or None,
                previous_brief=previous,
                mode=mode,  # type: ignore[arg-type]
                progress_path=progress_path,
            )
            pipe.record_asset(f"brief_{mode}", brief_path, verified=True, mode=mode)
        else:
            pipe.log.warning(f"phase {pipe.state.phase} has no Agent yet; iterate is a no-op")
    except Exception as e:
        pipe.log.exception(f"agent failed: {e}")
        pipe.bus.emit("asset_failed", agent=f"phase{pipe.state.phase}",
                      error=str(e), error_type=type(e).__name__)
        pipe.record_error(phase=pipe.state.phase, agent=f"phase{pipe.state.phase}-iterate",
                           error_type=type(e).__name__, error_text=str(e))
    finally:
        await REGISTRY.mark_done(pipe.run_id)


# ────────────────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────────────────

def run_server(host: str = "127.0.0.1", port: int = 7860, reload: bool = False) -> None:
    # Configure OTLP → Phoenix once for the server lifetime so all web-triggered
    # Agent runs export spans. Pipeline instances created via Registry use
    # launch_ui=False, which is now a no-op (already initialized) but the
    # Anthropic SDK auto-instrument was activated here, so their LLM calls
    # still emit spans into Phoenix.
    if not reload:
        from ..observability.tracer import phoenix_url
        from ..observability.tracer import setup as setup_tracing
        setup_tracing(project_name="video-workflow-web", launch_ui=True)
        print(f"Phoenix UI: {phoenix_url()}")
    else:
        print("⚠ --reload mode skips OTLP setup (worker isolation). "
              "Restart without --reload to enable tracing.")

    import uvicorn
    uvicorn.run("src.web.main:app" if reload else app,
                host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run_server()

"""CLI entry. Subcommands:

    doctor                        Probe local toolchain
    verify <path>                 Verify a video / audio / html asset
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Windows console UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def cmd_verify(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        return 1

    if path.is_dir():
        from .verify.html import verify_html
        r = verify_html(path)
        print(f"HTML asset: {path}")
        print(f"  ok = {r.ok}")
        for issue in r.issues:
            print(f"  [FAIL] {issue}")
        for warn in r.warnings:
            print(f"  [WARN] {warn}")
        return 0 if r.ok else 2

    suffix = path.suffix.lower()
    if suffix in {".webm", ".mp4", ".mkv", ".mov"}:
        from .verify.recording import verify_recording
        r = verify_recording(path, min_duration=None)
        p = r.probe
        print(f"Video file: {path}")
        print(f"  duration:   {p.duration:.2f}s")
        print(f"  resolution: {p.width}x{p.height}")
        print(f"  video:      {p.video_codec}")
        print(f"  audio:      {p.audio_codec or '(none)'}")
        print(f"  size:       {p.size_bytes / 1024 / 1024:.2f} MB")
        print(f"  ok = {r.ok}")
        for issue in r.issues:
            print(f"  [FAIL] {issue}")
        return 0 if r.ok else 2

    if suffix in {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}:
        from .verify.audio import verify_audio
        r = verify_audio(path)
        p = r.probe
        print(f"Audio file: {path}")
        print(f"  duration:   {p.duration:.3f}s")
        print(f"  codec:      {p.codec}")
        print(f"  sr/ch:      {p.sample_rate}Hz / {p.channels}ch")
        print(f"  ok = {r.ok}")
        for issue in r.issues:
            print(f"  [FAIL] {issue}")
        return 0 if r.ok else 2

    print(f"ERROR: unsupported file extension {suffix}", file=sys.stderr)
    return 1


def cmd_doctor(args) -> int:
    from .tools.ffbin import find as find_bin
    from .tools.shell import run as shell_run
    from .tools.dotenv import load_dotenv

    load_dotenv()
    print("== doctor ==")

    checks = [
        ("python", ["python", "--version"]),
        ("git",    ["git", "--version"]),
        ("node",   ["node", "--version"]),
        ("npm",    ["npm", "--version"]),
    ]
    ok = True
    for name, cmd in checks:
        try:
            r = shell_run(cmd, timeout=10)
            v = (r.stdout or r.stderr).strip().splitlines()[0]
            print(f"  [OK]   {name:18} {v}")
        except Exception as e:
            print(f"  [FAIL] {name:18} {e}")
            ok = False

    for bin_name in ["ffmpeg", "ffprobe"]:
        try:
            p = find_bin(bin_name)
            print(f"  [OK]   {bin_name:18} {p}")
        except FileNotFoundError as e:
            print(f"  [FAIL] {bin_name:18} {e}")
            ok = False

    if os.environ.get("ARK_KEY_1") or os.environ.get("ANTHROPIC_API_KEY"):
        print(f"  [OK]   {'ARK / API key':18} (set)")
    else:
        print(f"  [FAIL] {'ARK / API key':18} not set — copy .env.example to .env")
        ok = False

    # Live API connectivity probe
    if os.environ.get("ARK_KEY_1") or os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from .tools.llm import anthropic_client, model_for
            client = anthropic_client()
            resp = client.messages.create(
                model=model_for("reasoning"),
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            usage = resp.usage
            print(f"  [OK]   {'ark endpoint':18} {model_for('reasoning')} "
                  f"(in={usage.input_tokens} out={usage.output_tokens})")
        except Exception as e:
            print(f"  [FAIL] {'ark endpoint':18} {type(e).__name__}: {e}")
            ok = False

    return 0 if ok else 2


def cmd_analyze(args) -> int:
    """Run Agent 1 ProjectAnalyzer with iterative User approval loop (Gate #1)."""
    from .pipeline import Pipeline
    from .agents.project_analyzer import run_project_analyzer
    from .tools.dotenv import load_dotenv
    from .tools.shell import run as shell_run

    load_dotenv()

    repo_url = args.repo_url
    project = args.project or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")

    pipe = Pipeline(project=project, run_id=args.run_id,
                    launch_observability_ui=not args.no_observability)
    pipe.transition(phase=1, gate="running")

    repo_dir = pipe.run_dir / "repo"
    if repo_dir.exists():
        pipe.log.info(f"reusing existing clone at {repo_dir}")
    else:
        pipe.log.info(f"cloning {repo_url} → {repo_dir}")
        shell_run(["git", "clone", "--depth=1", repo_url, str(repo_dir)],
                  timeout=300, check=True)
    pipe.record_asset("repo", repo_dir, verified=True, url=repo_url)

    brief_path = pipe.run_dir / "project_brief.md"
    pipe.transition(gate="waiting_brief_approval")

    feedback: Optional[str] = None
    previous_brief: Optional[str] = None
    iteration = 0
    while True:
        iteration += 1
        run_project_analyzer(
            repo_dir=repo_dir, repo_url=repo_url,
            output_path=brief_path,
            feedback=feedback, previous_brief=previous_brief,
            mode=getattr(args, "mode", "standard"),
        )
        previous_brief = brief_path.read_text(encoding="utf-8")

        print("\n" + "=" * 70)
        print(f"project_brief.md  (iteration {iteration})")
        print("=" * 70 + "\n")
        print(previous_brief)
        print("\n" + "=" * 70)
        print(f"Saved to: {brief_path}")
        print("=" * 70)

        print("\n[a]pprove  /  [r]evise  /  [q]uit")
        choice = input("> ").strip().lower()
        if choice in {"a", "approve"}:
            pipe.gate_pass("waiting_brief_approval", iterations=iteration)
            pipe.record_asset("project_brief", brief_path, verified=True,
                              iterations=iteration)
            print(f"\n[OK] Gate #1 passed after {iteration} iteration(s).")
            return 0
        if choice in {"q", "quit"}:
            print("aborted by user")
            return 130
        # default: revise
        print("\nEnter feedback (multi-line OK; finish with empty line):")
        lines: list[str] = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if not ln:
                break
            lines.append(ln)
        feedback = "\n".join(lines).strip()
        if not feedback:
            print("(empty feedback — re-running with no guidance)")
            feedback = None


def cmd_serve(args) -> int:
    from .web.main import run_server
    print(f"Web UI: http://{args.host}:{args.port}")
    run_server(host=args.host, port=args.port, reload=args.reload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="viedo", description="Promo video pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="Verify a video/audio/html artifact")
    pv.add_argument("path")
    pv.set_defaults(func=cmd_verify)

    pd = sub.add_parser("doctor", help="Probe local toolchain")
    pd.set_defaults(func=cmd_doctor)

    pa = sub.add_parser("analyze", help="Run Agent 1 ProjectAnalyzer on a GitHub repo")
    pa.add_argument("repo_url", help="GitHub URL")
    pa.add_argument("--project", help="Project name (default: derived from URL)")
    pa.add_argument("--run-id", help="Existing run_id to resume")
    pa.add_argument("--mode", choices=["standard", "deep"], default="standard",
                    help="standard = README + key docs; deep = read source files too")
    pa.add_argument("--no-observability", action="store_true",
                    help="Skip OTLP tracing setup (Phoenix export)")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("serve", help="Launch the Web UI (FastAPI + HTMX)")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=7860)
    ps.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    ps.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

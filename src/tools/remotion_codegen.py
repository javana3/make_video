"""Deterministic codegen: cutting_plan.json → Remotion TSX project.

Produces under <remotion_dir>:
  package.json       deps: remotion 4, @remotion/transitions, react 18
  tsconfig.json      minimal TS config
  src/index.ts       registerRoot
  src/Root.tsx       <Composition> definition
  src/MyVideo.tsx    actual scenes + transitions
  public/recording.mp4   (caller copies)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..observability.audit import get_run_context, traced_agent
from ..observability.logger import agent_logger


def _pos_to_css(val: Any, axis: str) -> str:
    """Translate cutting_plan x/y values to CSS top/left + transform."""
    s = str(val).strip().lower() if val is not None else "center"
    if s == "center":
        prop = "top" if axis == "y" else "left"
        return f"{prop}:'50%'"
    if s.endswith("%") or s.endswith("px"):
        prop = "top" if axis == "y" else "left"
        return f"{prop}:'{s}'"
    if s.isdigit():
        prop = "top" if axis == "y" else "left"
        return f"{prop}:{int(s)}"
    prop = "top" if axis == "y" else "left"
    return f"{prop}:'50%'"


def _is_centered(v: Any) -> bool:
    return str(v).strip().lower() == "center"


def _gen_package_json() -> str:
    return json.dumps({
        "name": "promo-video",
        "version": "1.0.0",
        "private": True,
        "dependencies": {
            "@remotion/cli": "4.0.401",
            "@remotion/transitions": "4.0.401",
            "remotion": "4.0.401",
            "react": "^19.2.0",
            "react-dom": "^19.2.0",
        },
        "devDependencies": {
            "@types/react": "^19.0.0",
            "typescript": "^5.0.0",
        },
        "scripts": {
            "studio": "remotion studio",
            "render": "remotion render src/index.ts MyVideo out.mp4",
        },
    }, indent=2)


def _gen_tsconfig() -> str:
    return json.dumps({
        "compilerOptions": {
            "target": "ES2018",
            "module": "ESNext",
            "moduleResolution": "node",
            "jsx": "react-jsx",
            "strict": True,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "resolveJsonModule": True,
            "allowSyntheticDefaultImports": True,
        },
    }, indent=2)


def _gen_index_ts() -> str:
    return (
        'import { registerRoot } from "remotion";\n'
        'import { RemotionRoot } from "./Root";\n'
        'registerRoot(RemotionRoot);\n'
    )


def _gen_root_tsx(plan: dict, total_frames: int) -> str:
    return (
        'import { Composition } from "remotion";\n'
        'import { MyVideo } from "./MyVideo";\n'
        '\n'
        'export const RemotionRoot: React.FC = () => (\n'
        '  <Composition\n'
        '    id="MyVideo"\n'
        '    component={MyVideo}\n'
        f'    durationInFrames={{{total_frames}}}\n'
        f'    fps={{{plan["fps"]}}}\n'
        f'    width={{{plan["resolution_w"]}}}\n'
        f'    height={{{plan["resolution_h"]}}}\n'
        '  />\n'
        ');\n'
    )


def _bg_jsx(bg: dict, scene_duration_s: float, src_fps: int) -> str:
    """Render the background layer for one scene."""
    t = bg.get("type")
    if t == "color":
        c = bg.get("color", "#000000")
        return f'<AbsoluteFill style={{{{backgroundColor: "{c}"}}}} />'
    if t == "gradient":
        a = bg.get("color", "#000000")
        b = bg.get("gradient_to", a)
        return (f'<AbsoluteFill style={{{{background: "linear-gradient(135deg, {a} 0%, {b} 100%)"}}}} />')
    if t == "recording":
        src = bg.get("source_path", "recording.mp4")
        # source_path can be 'recordings/test.mp4' — copy to public/recording.mp4
        # so we always reference 'recording.mp4'
        start_s = float(bg.get("start_in_source_s", 5.0))
        start_frames = int(start_s * src_fps)
        return (f'<Video src={{staticFile("recording.mp4")}} startFrom={{{start_frames}}} '
                f'style={{{{width: "100%", height: "100%", objectFit: "cover"}}}} />')
    return '<AbsoluteFill style={{backgroundColor: "#000"}} />'


def _text_el_jsx(el: dict, scene_frames: int, fps: int) -> str:
    """Render one text element as an absolutely-positioned div with explicit
    centering via translate. Avoids flex layout quirks of AbsoluteFill."""
    content = el.get("content", "").replace('"', '\\"').replace("\n", "\\n")
    fs = int(el.get("font_size_px", 48))
    color = el.get("color", "#FFFFFF")
    x = el.get("x", "center")
    y = el.get("y", "center")
    in_frames = max(3, int(0.4 * fps))   # ~12 frames
    out_frames = in_frames

    # Translate x/y to CSS absolute coords. For "center" use 50% + translate.
    if _is_centered(x):
        left_css = '"50%"'
        translate_x = "-50%"
    elif str(x).endswith("%") or str(x).endswith("px"):
        left_css = f'"{x}"'
        translate_x = "0"
    else:
        left_css = '"50%"'; translate_x = "-50%"

    if _is_centered(y):
        top_css = '"50%"'
        translate_y = "-50%"
    elif str(y).endswith("%") or str(y).endswith("px"):
        top_css = f'"{y}"'
        translate_y = "-50%"  # vertically center on the y point
    else:
        top_css = '"50%"'; translate_y = "-50%"

    # Use a TextEl component (defined in MyVideo.tsx) — hooks at component top
    # level, plays nicely with TransitionSeries. Fade-in/out via interpolate.
    return (
        '<TextEl '
        f'content="{content}" '
        f'fontSize={{{fs}}} '
        f'color="{color}" '
        f'top={top_css} '
        f'left={left_css} '
        f'translate="translate({translate_x}, {translate_y})" '
        f'sceneFrames={{{scene_frames}}} '
        f'fadeFrames={{{in_frames}}} '
        '/>'
    )


def _scene_jsx(scene: dict, fps: int, src_fps: int) -> str:
    duration_frames = int(scene["duration_s"] * fps)
    bg = scene.get("background") or {}
    bg_jsx = _bg_jsx(bg, scene["duration_s"], src_fps)

    # darken can be at scene level OR background level (Agent freedom)
    darken = scene.get("darken")
    if darken is None:
        darken = bg.get("darken") or 0
    darken_jsx = ""
    if darken and float(darken) > 0:
        darken_jsx = (f'<AbsoluteFill style={{{{backgroundColor: "rgba(0,0,0,{float(darken)})"}}}} />')

    text_els = [el for el in (scene.get("elements") or []) if el.get("type") == "text"]
    elements_jsx = "\n        ".join(_text_el_jsx(el, duration_frames, fps) for el in text_els)
    if elements_jsx:
        elements_jsx = "\n        " + elements_jsx

    inner = f"""<AbsoluteFill>
        {bg_jsx}
        {darken_jsx}{elements_jsx}
      </AbsoluteFill>"""

    return (
        f'      <TransitionSeries.Sequence durationInFrames={{{duration_frames}}}>\n'
        f'        {inner}\n'
        '      </TransitionSeries.Sequence>'
    )


def _gen_myvideo_tsx(plan: dict, src_fps: int) -> str:
    fps = int(plan["fps"])
    scenes = plan["scenes"]

    parts = []
    for i, scene in enumerate(scenes):
        parts.append(_scene_jsx(scene, fps, src_fps))
        if i < len(scenes) - 1:
            parts.append(
                '      <TransitionSeries.Transition presentation={fade()} '
                'timing={linearTiming({durationInFrames: 15})} />'
            )

    body = "\n".join(parts)
    return (
        'import { AbsoluteFill, Video, useCurrentFrame, interpolate, staticFile } from "remotion";\n'
        'import { TransitionSeries, linearTiming } from "@remotion/transitions";\n'
        'import { fade } from "@remotion/transitions/fade";\n'
        '\n'
        '// Helper: animated text element. Hooks at component top level (per\n'
        '// React rules of hooks). Used by every scene needing fade-in/out text.\n'
        'type TextElProps = {\n'
        '  content: string;\n'
        '  fontSize: number;\n'
        '  color: string;\n'
        '  top: string;\n'
        '  left: string;\n'
        '  translate: string;\n'
        '  sceneFrames: number;\n'
        '  fadeFrames: number;\n'
        '};\n'
        'const TextEl: React.FC<TextElProps> = (p) => {\n'
        '  const frame = useCurrentFrame();\n'
        '  const opacity = interpolate(\n'
        '    frame,\n'
        '    [0, p.fadeFrames, p.sceneFrames - p.fadeFrames, p.sceneFrames],\n'
        '    [0, 1, 1, 0],\n'
        '    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },\n'
        '  );\n'
        '  return (\n'
        '    <div style={{\n'
        '      position: "absolute",\n'
        '      top: p.top,\n'
        '      left: p.left,\n'
        '      transform: p.translate,\n'
        '      color: p.color,\n'
        '      fontSize: p.fontSize,\n'
        '      fontWeight: 700,\n'
        '      fontFamily: \'"Microsoft YaHei", "PingFang SC", "Noto Sans SC", system-ui, sans-serif\',\n'
        '      textShadow: "0 2px 12px rgba(0,0,0,0.7)",\n'
        '      whiteSpace: "nowrap",\n'
        '      textAlign: "center",\n'
        '      opacity,\n'
        '    }}>{p.content}</div>\n'
        '  );\n'
        '};\n'
        '\n'
        'export const MyVideo: React.FC = () => {\n'
        '  return (\n'
        '    <TransitionSeries>\n'
        f'{body}\n'
        '    </TransitionSeries>\n'
        '  );\n'
        '};\n'
    )


def total_frames(plan: dict) -> int:
    """TransitionSeries totals: sum scene durations - (n-1) * crossfade_frames."""
    fps = int(plan["fps"])
    scenes = plan["scenes"]
    scene_total = sum(int(s["duration_s"] * fps) for s in scenes)
    overlap = (len(scenes) - 1) * 15  # crossfade overlap
    return max(scene_total - overlap, fps * 1)


@traced_agent("Agent 3 RemotionComposer · codegen", phase=3)
def generate_project(plan: dict,
                     remotion_dir: Path,
                     recording_path: Path,
                     src_fps: int = 30) -> dict:
    """Write the Remotion project to remotion_dir. Returns summary dict."""
    log = agent_logger("agent3_remotion")
    log.info(f"codegen → {remotion_dir}  scenes={len(plan.get('scenes', []))}  src_fps={src_fps}")
    remotion_dir.mkdir(parents=True, exist_ok=True)
    src_dir = remotion_dir / "src"
    public_dir = remotion_dir / "public"
    src_dir.mkdir(exist_ok=True)
    public_dir.mkdir(exist_ok=True)

    (remotion_dir / "package.json").write_text(_gen_package_json(), encoding="utf-8")
    (remotion_dir / "tsconfig.json").write_text(_gen_tsconfig(), encoding="utf-8")
    (src_dir / "index.ts").write_text(_gen_index_ts(), encoding="utf-8")

    tf = total_frames(plan)
    (src_dir / "Root.tsx").write_text(_gen_root_tsx(plan, tf), encoding="utf-8")
    (src_dir / "MyVideo.tsx").write_text(_gen_myvideo_tsx(plan, src_fps), encoding="utf-8")

    # Copy recording into public/ (Remotion's staticFile() resolves there)
    target_recording = public_dir / "recording.mp4"
    if recording_path.exists():
        shutil.copy2(recording_path, target_recording)

    summary = {
        "remotion_dir": str(remotion_dir),
        "total_frames": tf,
        "fps": plan["fps"],
        "duration_s": tf / plan["fps"],
        "scenes": len(plan["scenes"]),
    }
    log.info(f"codegen done: {summary['scenes']} scenes, {summary['total_frames']} frames "
             f"@ {summary['fps']}fps = {summary['duration_s']:.1f}s")
    bus = get_run_context().get("event_bus")
    if bus is not None:
        bus.emit("asset_verified", agent="agent3_remotion",
                 name="remotion_project", path=str(remotion_dir),
                 total_frames=tf, fps=plan["fps"],
                 duration_s=summary["duration_s"], scenes=summary["scenes"])
    return summary

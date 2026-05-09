"""M2a · Open Design daemon HTTP client.

Talks to the Open Design dev server at http://127.0.0.1:<DAEMON_PORT>/. The
daemon spawns a coding-agent CLI (we use OpenCode + 火山方舟 from
~/.config/opencode/opencode.json) to actually generate the HTML, returns SSE
stream events, and writes artifacts to .od/projects/<id>/.

Routes used:
  GET  /api/health
  GET  /api/agents              available coding-agent CLIs on PATH
  GET  /api/skills              31 skills + many imported templates
  GET  /api/design-systems      138 design systems
  POST /api/projects            create project (we generate id + default conv)
  GET  /api/projects/:id        read project metadata (incl. default conversation)
  POST /api/chat                start a chat run (SSE response)
  GET  /api/projects/:id/files/:name   read artifact (e.g. index.html)
  GET  /api/projects/:id/archive       download project as zip
"""
from __future__ import annotations

import io
import json
import re
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import httpx

# httpx by default reads HTTP(S)_PROXY from env AND the Windows system proxy
# settings (registry). When the user has a VPN proxy configured (we observed
# 502 Bad Gateway via WinINET on a 127.0.0.1 daemon), routing 127.0.0.1
# through that proxy fails. trust_env=False sidesteps both env + registry.
def _get(url: str, **kwargs):
    with httpx.Client(trust_env=False) as c:
        return c.get(url, **kwargs)


def _post(url: str, **kwargs):
    with httpx.Client(trust_env=False) as c:
        return c.post(url, **kwargs)
from loguru import logger


_PORT_RE_WEB = re.compile(r"Web:\s*http://127\.0\.0\.1:(\d+)")
_PORT_RE_DAEMON = re.compile(r"Daemon:\s*http://127\.0\.0\.1:(\d+)")


@dataclass
class OpenDesignEndpoint:
    web_url: str       # e.g. http://127.0.0.1:61792/
    daemon_url: str    # e.g. http://127.0.0.1:54230/

    @classmethod
    def from_dev_log(cls, log_text: str) -> "OpenDesignEndpoint":
        wm = _PORT_RE_WEB.search(log_text)
        dm = _PORT_RE_DAEMON.search(log_text)
        if not wm or not dm:
            raise RuntimeError(f"could not find Web/Daemon URL in dev server log:\n{log_text[-400:]}")
        return cls(
            web_url=f"http://127.0.0.1:{wm.group(1)}/",
            daemon_url=f"http://127.0.0.1:{dm.group(1)}/",
        )


# ---------------------------------------------------------------------------
# Read-only probes
# ---------------------------------------------------------------------------

def health_check(daemon_url: str, timeout: float = 5.0) -> dict:
    r = _get(f"{daemon_url.rstrip('/')}/api/health", timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_agents(daemon_url: str) -> list[dict]:
    r = _get(f"{daemon_url.rstrip('/')}/api/agents", timeout=10)
    r.raise_for_status()
    return r.json().get("agents", [])


def list_skills(daemon_url: str) -> list[dict]:
    r = _get(f"{daemon_url.rstrip('/')}/api/skills", timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("skills", data.get("items", data if isinstance(data, list) else []))


def list_design_systems(daemon_url: str) -> list[dict]:
    r = _get(f"{daemon_url.rstrip('/')}/api/design-systems", timeout=10)
    r.raise_for_status()
    return r.json().get("designSystems", [])


def pick_available_agent(daemon_url: str, preferred_order: list[str]) -> str:
    agents = list_agents(daemon_url)
    by_id = {a["id"]: a for a in agents}
    for pref in preferred_order:
        a = by_id.get(pref)
        if a and a.get("available"):
            return pref
    raise RuntimeError(
        f"none of {preferred_order} available; got: "
        f"{[a['id'] for a in agents if a.get('available')]}"
    )


# ---------------------------------------------------------------------------
# Project lifecycle
# ---------------------------------------------------------------------------

def _fresh_project_id(prefix: str = "video-promo") -> str:
    """Project id must match ^[A-Za-z0-9._-]{1,128}$."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def create_project(daemon_url: str,
                   name: str,
                   skill_id: Optional[str] = None,
                   design_system_id: Optional[str] = None,
                   pending_prompt: Optional[str] = None,
                   project_id: Optional[str] = None) -> dict:
    """POST /api/projects — daemon also seeds a default conversation.

    Returns the daemon's response payload, augmented with `conversation_id`
    obtained via a follow-up GET /api/projects/:id (since POST may not
    surface it directly in older daemon builds).
    """
    pid = project_id or _fresh_project_id()
    body = {"id": pid, "name": name}
    if skill_id: body["skillId"] = skill_id
    if design_system_id: body["designSystemId"] = design_system_id
    if pending_prompt: body["pendingPrompt"] = pending_prompt

    r = _post(f"{daemon_url.rstrip('/')}/api/projects",
                    json=body, timeout=30)
    r.raise_for_status()
    project = r.json()

    # Daemon seeds a default conversation in the same handler — fetch it.
    convs = _get(
        f"{daemon_url.rstrip('/')}/api/projects/{pid}/conversations",
        timeout=10,
    ).json()
    conv_list = convs.get("conversations", convs.get("items", convs))
    if not conv_list:
        raise RuntimeError("daemon did not auto-create a default conversation")
    conv_id = conv_list[0]["id"]

    return {"project": project, "project_id": pid, "conversation_id": conv_id}


def get_project(daemon_url: str, project_id: str) -> dict:
    r = _get(
        f"{daemon_url.rstrip('/')}/api/projects/{project_id}",
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Chat — POST /api/chat returns SSE
# ---------------------------------------------------------------------------

def send_prompt_stream(daemon_url: str,
                       project_id: str,
                       conversation_id: str,
                       message: str,
                       agent_id: str = "opencode",
                       skill_id: Optional[str] = None,
                       design_system_id: Optional[str] = None,
                       model: Optional[str] = None,
                       client_request_id: Optional[str] = None,
                       timeout: float = 600.0) -> Iterator[dict]:
    """POST /api/chat with SSE streaming. Yields parsed event dicts.

    Each yielded dict has at minimum {"event": <type>, "data": <obj>}.
    The caller decides when to stop (typically on event types that signal
    completion: "run.complete" / "artifact.write" / similar — actual event
    names live in design.runs.stream which we infer from the daemon).
    """
    body = {
        "agentId": agent_id,
        "message": message,
        "projectId": project_id,
        "conversationId": conversation_id,
        "clientRequestId": client_request_id or uuid.uuid4().hex,
    }
    if skill_id: body["skillId"] = skill_id
    if design_system_id: body["designSystemId"] = design_system_id
    if model: body["model"] = model

    from ..observability.logger import agent_logger
    log = agent_logger("agent6_opendesigner")
    log.info(f"POST /api/chat agent={agent_id} skill={skill_id} ds={design_system_id} msg_len={len(message)}")

    headers = {"Accept": "text/event-stream"}
    with httpx.Client(trust_env=False) as client, client.stream(
        "POST", f"{daemon_url.rstrip('/')}/api/chat",
        json=body, headers=headers, timeout=timeout,
    ) as r:
        r.raise_for_status()
        event_type = "message"
        data_lines: list[str] = []
        for raw in r.iter_lines():
            if raw is None: continue
            line = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
            if line == "":
                if data_lines:
                    payload_str = "\n".join(data_lines)
                    try:
                        payload = json.loads(payload_str)
                    except json.JSONDecodeError:
                        payload = {"raw": payload_str}
                    yield {"event": event_type, "data": payload}
                event_type = "message"
                data_lines = []
                continue
            if line.startswith(":"):  # SSE comment
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())


# ---------------------------------------------------------------------------
# Artifact retrieval
# ---------------------------------------------------------------------------

def read_artifact_bytes(daemon_url: str, project_id: str,
                         file_name: str = "index.html") -> bytes:
    r = _get(
        f"{daemon_url.rstrip('/')}/api/projects/{project_id}/files/{file_name}",
        timeout=30,
    )
    r.raise_for_status()
    return r.content


def list_project_files(daemon_url: str, project_id: str) -> list[dict]:
    r = _get(
        f"{daemon_url.rstrip('/')}/api/projects/{project_id}/files",
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("files", data.get("items", data if isinstance(data, list) else []))


def download_archive(daemon_url: str, project_id: str,
                     target_dir: Path) -> dict:
    """GET /api/projects/:id/archive → unzip into target_dir.

    target_dir must NOT exist (or be empty); we replace any existing
    html_asset/ wholesale to make adoption deterministic.
    """
    r = _get(
        f"{daemon_url.rstrip('/')}/api/projects/{project_id}/archive",
        timeout=120,
    )
    r.raise_for_status()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            zf.extract(name, target_dir)
            extracted.append(name)
    return {
        "target_dir": str(target_dir),
        "n_files": len(extracted),
        "files": extracted,
        "archive_bytes": len(r.content),
    }

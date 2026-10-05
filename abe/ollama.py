"""Talk to a local Ollama server: list installed models and download new ones.

Ollama listens on http://localhost:11434 unless the OLLAMA_HOST environment
variable says otherwise. ABE reads the same variable, so both agree.

Downloads run in a background thread because a model is several gigabytes. Their
progress is kept in memory, keyed by model name, for the web workbench to show.
If a download fails, the error is kept so the page can tell the user to run
`ollama pull <model>` themselves.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field

import requests

DEFAULT_HOST = "http://localhost:11434"
LIST_TIMEOUT_SECONDS = 2


def host() -> str:
    value = os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_HOST
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.rstrip("/")


def openai_endpoint() -> str:
    """The OpenAI-compatible endpoint the agent's provider calls."""
    return f"{host()}/v1"


def host_from_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or "").rstrip("/")
    return endpoint[: -len("/v1")] if endpoint.endswith("/v1") else (endpoint or host())


def list_models(base: str | None = None) -> list[str] | None:
    """Names of installed models, sorted, or None if Ollama isn't reachable."""
    try:
        response = requests.get(f"{base or host()}/api/tags", timeout=LIST_TIMEOUT_SECONDS)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (requests.RequestException, ValueError):
        return None
    return sorted({m.get("name") or m.get("model") for m in models if m.get("name") or m.get("model")})


def is_installed(model: str, installed: list[str]) -> bool:
    """Ollama treats a name without a tag as name:latest."""
    model = model.strip()
    return model in installed or (":" not in model and f"{model}:latest" in installed)


@dataclass
class PullStatus:
    model: str
    state: str = "starting"  # starting | downloading | done | error
    message: str = ""
    completed: int = 0
    total: int = 0
    started: float = field(default_factory=time.time)

    @property
    def percent(self) -> int | None:
        return int(self.completed * 100 / self.total) if self.total else None

    @property
    def active(self) -> bool:
        return self.state in {"starting", "downloading"}


_pulls: dict[str, PullStatus] = {}
_lock = threading.Lock()


def pull_status(model: str) -> PullStatus | None:
    with _lock:
        return _pulls.get(model)


def start_pull(model: str, base: str | None = None) -> PullStatus:
    """Start downloading `model` in the background unless a download is already running."""
    model = model.strip()
    with _lock:
        current = _pulls.get(model)
        if current and current.active:
            return current
        status = PullStatus(model)
        _pulls[model] = status
    thread = threading.Thread(target=_pull, args=(status, base or host()), daemon=True)
    thread.start()
    return status


def _pull(status: PullStatus, base: str) -> None:
    try:
        with requests.post(
            f"{base}/api/pull",
            json={"model": status.model, "name": status.model, "stream": True},
            stream=True,
            timeout=(5, 600),
        ) as response:
            if response.status_code >= 400:
                _fail(status, f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
                return
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    _fail(status, event["error"])
                    return
                with _lock:
                    status.state = "downloading"
                    status.message = event.get("status", "")
                    if event.get("total"):
                        status.total = event["total"]
                        status.completed = event.get("completed", 0)
                if event.get("status") == "success":
                    with _lock:
                        status.state = "done"
                        status.message = "Installed"
                    return
        _fail(status, "The download ended without Ollama reporting success.")
    except requests.RequestException as exc:
        _fail(status, f"Could not reach Ollama at {base}: {exc}")
    except ValueError as exc:
        _fail(status, f"Unexpected reply from Ollama: {exc}")


def _fail(status: PullStatus, message: str) -> None:
    with _lock:
        status.state = "error"
        status.message = message

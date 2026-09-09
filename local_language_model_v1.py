"""Discover and prepare Hyponoia's small local comment-understanding model.

Ollama is an external local runtime.  Hyponoia never installs it silently: the
UI opens the official download page when it is absent.  Once Ollama exists,
the UI can start it and download the pinned multilingual model locally.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download/mac"


class LanguageModelSetupError(RuntimeError):
    """Raised when the local language model cannot be prepared safely."""


def _candidate_clis() -> tuple[Path, ...]:
    candidates: list[Path] = []
    discovered = shutil.which("ollama")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        (
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path.home() / "Applications/Ollama.app/Contents/Resources/ollama",
        )
    )
    return tuple(dict.fromkeys(candidates))


def find_ollama_cli(candidates: tuple[Path, ...] | None = None) -> Path | None:
    for candidate in candidates or _candidate_clis():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise LanguageModelSetupError(f"Local Ollama service is unavailable: {exc}") from exc


def installed_models(
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 0.8,
    read_json: Callable[[str, float], dict[str, Any]] = _read_json,
) -> set[str]:
    payload = read_json(f"{base_url.rstrip('/')}/api/tags", timeout)
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise LanguageModelSetupError("Local Ollama service returned an invalid model list")
    result: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                result.add(value.strip())
    return result


def language_model_status(
    *,
    model: str = DEFAULT_MODEL,
    candidates: tuple[Path, ...] | None = None,
    read_json: Callable[[str, float], dict[str, Any]] = _read_json,
) -> dict[str, Any]:
    cli = find_ollama_cli(candidates)
    try:
        models = installed_models(read_json=read_json)
        service_online = True
    except LanguageModelSetupError:
        models = set()
        service_online = False
    aliases = {model, f"{model}:latest"}
    model_installed = bool(models & aliases)
    return {
        "model": model,
        "runner_installed": cli is not None,
        "runner_path": str(cli) if cli is not None else None,
        "service_online": service_online,
        "model_installed": model_installed,
        "ready": service_online and model_installed,
    }


def _start_ollama(cli: Path) -> None:
    app = Path("/Applications/Ollama.app")
    user_app = Path.home() / "Applications/Ollama.app"
    if app.exists() or user_app.exists():
        subprocess.Popen(
            ["open", "-a", "Ollama"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            [str(cli), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def prepare_language_model(
    *,
    model: str = DEFAULT_MODEL,
    wait_seconds: float = 40.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    """Start Ollama if needed and pull the pinned model.

    The caller must obtain the listener's confirmation before invoking this
    function because the model download is approximately 2.5 GB.
    """
    status = language_model_status(model=model)
    if status["ready"]:
        return status
    cli = find_ollama_cli()
    if cli is None:
        raise LanguageModelSetupError(
            "Ollama is not installed. Install it from the official download page first."
        )
    if not status["service_online"]:
        _start_ollama(cli)
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        while time.monotonic() < deadline:
            time.sleep(0.4)
            status = language_model_status(model=model)
            if status["service_online"]:
                break
        if not status["service_online"]:
            raise LanguageModelSetupError("Ollama was found but its local service did not start")
    if not status["model_installed"]:
        completed = run(
            [str(cli), "pull", model],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown error").strip()
            raise LanguageModelSetupError(f"The local model download failed: {detail}")
    final = language_model_status(model=model)
    if not final["ready"]:
        raise LanguageModelSetupError("The local model was downloaded but is not ready")
    return final

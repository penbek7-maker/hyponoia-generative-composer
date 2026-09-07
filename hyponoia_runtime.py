"""Portable runtime checks and commands for the Hyponoia desktop shell."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from composition_preference_v1 import CompositionPreferenceAssist
from representation_assist_v1 import RepresentationAssist


PROJECT_DIR = Path(__file__).resolve().parent


def _resolve(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


def load_user_config(project_dir: str | Path = PROJECT_DIR) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    path = root / "hyponoia_user_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def runtime_status(project_dir: str | Path = PROJECT_DIR) -> dict[str, Any]:
    """Return user-facing readiness without changing any local data."""
    root = Path(project_dir).resolve()
    config = load_user_config(root)
    memory_path = _resolve(root, config.get("memory_file", "memory_index_v3.json"))
    representation_path = _resolve(
        root, config.get("representation_config", "representation_config.json")
    )
    preference_path = _resolve(
        root,
        config.get(
            "composition_preference",
            "phase2_artifacts/composition_preference_gold.json",
        ),
    )

    recordings = 0
    sound_objects = 0
    memory_error = None
    if memory_path.exists():
        try:
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            recordings = len(memory)
            sound_objects = sum(len(item.get("objects", [])) for item in memory)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            memory_error = str(exc)

    representation = RepresentationAssist.from_config(representation_path)
    preference = CompositionPreferenceAssist.from_file(preference_path)
    current_audio = root / "output" / "current.wav"
    return {
        "project_dir": str(root),
        "memory_path": str(memory_path),
        "memory_ready": recordings > 0 and sound_objects > 0 and memory_error is None,
        "memory_error": memory_error,
        "recordings": recordings,
        "sound_objects": sound_objects,
        "representation": representation.snapshot(),
        "composition_preference": preference.snapshot(),
        "current_audio": str(current_audio),
        "current_audio_ready": current_audio.exists(),
        "ready_to_generate": (
            recordings > 0
            and sound_objects > 0
            and memory_error is None
            and representation.active
            and preference.active
        ),
    }


def generator_command(
    dream_level: int,
    *,
    root_pitch: int = 0,
    scale: str = "free",
    confidence: float = 0.0,
    project_dir: str | Path = PROJECT_DIR,
) -> list[str]:
    if dream_level not in (1, 3, 5):
        raise ValueError("dream_level must be 1, 3 or 5")
    if not 0 <= int(root_pitch) <= 11:
        raise ValueError("root_pitch must be between 0 and 11")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    script = Path(project_dir).resolve() / "generator_v3_memory_bloom_smooth.py"
    return [
        sys.executable,
        str(script),
        str(dream_level),
        str(int(root_pitch)),
        str(scale),
        str(float(confidence)),
    ]

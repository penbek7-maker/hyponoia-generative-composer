"""One learning-profile source for generation, text feedback and voice feedback."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from human_feedback_v1 import DEFAULT_LEARNING_PROFILE, DEFAULT_WEIGHTS


PROJECT_DIR = Path(__file__).resolve().parent
SEED_PATH = PROJECT_DIR / "phase2_artifacts" / "learning_profile_seed.json"


def _normalise(profile: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(profile)
    result.setdefault("history", [])
    shared = result.setdefault("weights", {})
    for control, default in DEFAULT_WEIGHTS.items():
        shared.setdefault(control, default)
    levels = result.setdefault("level_weights", {})
    for level in ("D1", "D3", "D5"):
        current = levels.setdefault(level, {})
        for control, default in DEFAULT_WEIGHTS.items():
            current.setdefault(control, default)
    return result


def load_learning_seed(path: str | Path = SEED_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return _normalise(DEFAULT_LEARNING_PROFILE)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid release learning seed: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid release learning seed: {source}")
    return _normalise(payload)


def load_active_learning_profile(path: str | Path) -> dict[str, Any]:
    """Load user learning on top of the frozen, comment-free release baseline.

    Old development profiles are migrated without losing their append-only
    comment history. Their weights are replaced by the already distilled gold
    baseline because those historic sessions are already represented there.
    """
    target = Path(path)
    seed = load_learning_seed()
    if not target.exists():
        return seed
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid learning profile: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid learning profile: {target}")
    payload = _normalise(payload)

    seed_id = seed.get("baseline_id")
    canonical_user_profile = (PROJECT_DIR / "learning_profile.json").resolve()
    if (
        seed_id
        and payload.get("baseline_id") != seed_id
        and target.expanduser().resolve() == canonical_user_profile
    ):
        migrated = copy.deepcopy(seed)
        migrated["history"] = copy.deepcopy(payload.get("history", []))
        migrated["migration"] = {
            "from_profile_version": payload.get("version"),
            "preserved_history_events": len(migrated["history"]),
            "policy": "distilled baseline already includes the historic training deltas",
        }
        return _normalise(migrated)
    return payload

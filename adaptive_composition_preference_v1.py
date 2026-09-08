"""Safe incremental preference learning over complete Hyponoia renders.

The frozen gold model is never edited. Explicit listener decisions update a
private copy with a small exponential-moving-average step. The copy contains
only numeric ratings and render identifiers; free text and voice transcripts
stay in the private feedback evidence store.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from composition_preference_v1 import (
    _load_embeddings,
    _unit,
    render_prototype,
    render_structure,
)
from hyponoia_stability import atomic_write_json, utc_timestamp


RATING_WEIGHTS = {
    "musicality": 0.20,
    "material_coherence": 0.10,
    "transition_smoothness": 0.10,
    "variety_without_disconnection": 0.05,
    "synth_material_presence": 0.05,
    "development_over_repetition": 0.15,
    "overall_artistic_impression": 0.35,
}


def preference_score(ratings: dict[str, Any]) -> float:
    """Return a transparent 1–5 whole-composition preference score."""
    values = {}
    for name in RATING_WEIGHTS:
        value = float(ratings[name])
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"rating {name} must be between 1 and 5")
        values[name] = value
    return float(sum(values[name] * weight for name, weight in RATING_WEIGHTS.items()))


def learning_direction(event: dict[str, Any]) -> str:
    """Use an explicit keep/no decision; uncertain reviews do not move the head."""
    decision = str(event.get("baseline_decision", "")).strip().lower()
    if decision in {"yes", "positive"}:
        return "positive"
    if decision in {"no", "negative"}:
        return "negative"
    if decision in {"unsure", "maybe", ""}:
        return "neutral"
    raise ValueError("baseline_decision must be yes, no or unsure")


def _blend_vector(existing: list[float], observed: np.ndarray, rate: float) -> list[float]:
    return _unit((1.0 - rate) * np.asarray(existing, dtype=np.float64) + rate * observed).tolist()


def _blend_structure(existing: dict[str, Any], observed: dict[str, Any], rate: float) -> dict[str, Any]:
    updated = copy.deepcopy(existing)
    for key, value in observed.items():
        if key == "role_distribution":
            old_roles = dict(updated.get(key, {}))
            roles = set(old_roles) | set(value)
            mixed = {
                role: (1.0 - rate) * float(old_roles.get(role, 0.0))
                + rate * float(value.get(role, 0.0))
                for role in roles
            }
            total = sum(mixed.values()) or 1.0
            updated[key] = {role: amount / total for role, amount in sorted(mixed.items())}
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            old = float(updated.get(key, value))
            updated[key] = (1.0 - rate) * old + rate * float(value)
    return updated


def _resolved_embeddings(model_path: Path, model: dict[str, Any]) -> Path:
    source = Path(str(model["embeddings_path"])).expanduser()
    return source.resolve() if source.is_absolute() else (model_path.parent / source).resolve()


def update_preference_from_review(
    *,
    base_model_path: str | Path,
    user_model_path: str | Path,
    render_report_path: str | Path,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Update a private preference head from one explicitly confirmed review."""
    base_path = Path(base_model_path).expanduser().resolve()
    user_path = Path(user_model_path).expanduser().resolve()
    report_path = Path(render_report_path).expanduser().resolve()
    model_path = user_path if user_path.exists() else base_path
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read the composition preference evidence") from exc
    if model.get("schema_version") != "composition_preference_v1":
        raise ValueError("unsupported composition preference model")

    level = str(event.get("dream_level", "")).upper()
    expected_level = int(level[1:]) if level in {"D1", "D3", "D5"} else None
    if expected_level is None or int(report.get("dream_level", -1)) != expected_level:
        raise ValueError("the reviewed D-level does not match the latest render")

    direction = learning_direction(event)
    score = preference_score(dict(event.get("ratings_1_to_5", {})))
    result = {
        "updated": False,
        "direction": direction,
        "score_1_to_5": round(score, 4),
        "model_path": str(user_path),
    }
    if direction == "neutral":
        result["reason"] = "listener selected unsure; control feedback was kept but the neural head was not moved"
        return result

    embeddings_path = _resolved_embeddings(model_path, model)
    embeddings = _load_embeddings(embeddings_path)
    observed, evidence = render_prototype(report, embeddings)
    structure = render_structure(report)
    confidence = min(1.0, abs(score - 3.0) / 2.0)
    rate = float(np.clip(0.06 + 0.06 * confidence, 0.06, 0.12))
    prototype_key = "positive_prototype" if direction == "positive" else "negative_prototype"
    model[prototype_key] = _blend_vector(model[prototype_key], observed, rate)

    level_heads = model.setdefault("level_heads", {})
    head = level_heads.get(str(expected_level))
    if head:
        head_key = "positive_prototype" if direction == "positive" else "negative_prototype"
        head[head_key] = _blend_vector(head[head_key], observed, rate)
        if direction == "positive":
            head["positive_structure"] = _blend_structure(
                dict(head.get("positive_structure", {})), structure, rate
            )
    if direction == "positive":
        model["positive_structure"] = _blend_structure(
            dict(model.get("positive_structure", {})), structure, rate
        )

    model["embeddings_path"] = os.path.relpath(embeddings_path, user_path.parent)
    online = model.setdefault("online_learning", {})
    online["schema_version"] = "adaptive_composition_preference_v1"
    online["updated_at"] = utc_timestamp()
    online["review_count"] = int(online.get("review_count", 0)) + 1
    online["positive_count"] = int(online.get("positive_count", 0)) + int(direction == "positive")
    online["negative_count"] = int(online.get("negative_count", 0)) + int(direction == "negative")
    history = online.setdefault("history", [])
    history.append({
        "event_id": event.get("event_id"),
        "timestamp": event.get("timestamp"),
        "dream_level": level,
        "direction": direction,
        "score_1_to_5": round(score, 4),
        "learning_rate": round(rate, 6),
        "render": Path(str(report.get("audio_file", "current.wav"))).name,
        "covered_objects": evidence["covered_objects"],
        "policy": "explicit decision; bounded update; no raw text or voice stored in model",
    })
    atomic_write_json(user_path, model)
    result.update({"updated": True, "learning_rate": rate, "covered_objects": evidence["covered_objects"]})
    return result

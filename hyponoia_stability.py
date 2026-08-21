"""Shared reliability helpers for the Hyponoia v2 stabilisation baseline.

This module deliberately has no audio-library dependency.  It owns stable IDs,
atomic JSON persistence, deterministic grouping, feedback intent parsing and
safe migration of the original filename/index based sample-learning profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TARGET_SR = 48_000
LEARNING_SCHEMA_VERSION = 2
SAMPLE_PROFILE_VERSION = 2
DREAM_LEVELS = ("D1", "D3", "D5")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_audio_id(audio: np.ndarray, sample_rate: int, prefix: str) -> str:
    """Return a content-derived ID from canonical mono float32 audio."""
    canonical = np.ascontiguousarray(np.asarray(audio, dtype="<f4").reshape(-1))
    digest = hashlib.sha256()
    digest.update(f"hyponoia-audio-v1:{int(sample_rate)}:".encode("ascii"))
    digest.update(canonical.tobytes())
    return f"{prefix}_{digest.hexdigest()[:20]}"


def stable_object_id(fragment: np.ndarray, sample_rate: int) -> str:
    return stable_audio_id(fragment, sample_rate, "obj")


def stable_recording_id(audio: np.ndarray, sample_rate: int) -> str:
    return stable_audio_id(audio, sample_rate, "rec")


def sample_key(recording_id: str, object_id: str) -> str:
    return f"{recording_id}::{object_id}"


def deterministic_group(value: str, groups: int = 4) -> int:
    if groups < 1:
        raise ValueError("groups must be at least 1")
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).digest()
    return int.from_bytes(digest[:8], "big") % groups


def atomic_write_json(path: str | os.PathLike[str], data: Any) -> None:
    """Write JSON completely before replacing the previous file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _normalise_phrase(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


_INTENT_RULES = (
    {
        "intent": "increase_musicality",
        "patterns": (
            r"\bmore musical(?:ity)?\b",
            r"\bincrease (?:the )?musicality\b",
        ),
        "updates": {"musicality_weight": 0.06},
    },
    {
        "intent": "increase_rhythmicity",
        "patterns": (
            r"\bmore rhythm(?:ic|ical|icity)?\b",
            r"\bstronger pulse\b",
            r"\bincrease (?:the )?rhythm(?:ic|icity)?\b",
        ),
        "updates": {"activity_weight": 0.05},
        "implementation_note": "Uses bounded phrase activity and pulse attraction; it does not impose a rigid beat.",
    },
    {
        "intent": "increase_bloom",
        "patterns": (
            r"\b(?:more|greater|bigger|larger|stronger) bloom\b",
            r"\bincrease (?:the )?bloom\b",
        ),
        "updates": {"bloom_weight": 0.07},
    },
    {
        "intent": "increase_synthetic_material",
        "patterns": (r"\bmore synth(?:s|esizers?|etic)?\b", r"\bmore electronic (?:sound|material)s?\b"),
        "updates": {"synthetic_material_weight": 0.08},
        "implementation_note": "Uses a transparent weak acoustic heuristic; it is not a trained source classifier.",
    },
    {
        "intent": "increase_library_exploration",
        "patterns": (
            r"\bmore (?:library )?(?:objects|samples|sounds|variety)\b",
            r"\b(?:greater|more|increase) exploration\b",
            r"\bexplore more\b",
            r"\b(?:use|try) (?:new|different) (?:sounds|samples|materials?)\b",
            r"\bdifferent (?:sounds|samples|materials?)\b",
        ),
        "updates": {"exploration_weight": 0.10, "repetition_control": 0.05},
    },
    {
        "intent": "decrease_repetition",
        "patterns": (r"\bless repet(?:ition|itive)\b", r"\bfewer repeats?\b", r"\bdont repeat as much\b"),
        "updates": {"repetition_control": 0.10, "exploration_weight": 0.06},
    },
    {
        "intent": "increase_smoothness",
        "patterns": (r"\bsmoother\b", r"\bmore smooth(?:ness| transitions?)?\b", r"\bless abrupt\b"),
        "updates": {"transition_smoothness_weight": 0.08},
    },
    {
        "intent": "increase_richness",
        "patterns": (r"\bricher\b", r"\bmore (?:richness|layers?|texture)s?\b", r"\bdenser\b"),
        "updates": {"richness_weight": 0.07},
    },
    {
        "intent": "increase_activity",
        "patterns": (
            r"\bmore energ(?:y|etic)\b",
            r"\bmore active\b",
            r"\bincrease (?:the )?(?:energy|activity)\b",
            r"\b(?:faster|quicker)\b",
            r"\b(?:more|increase) (?:the )?speed\b",
            r"\bhigher tempo\b",
        ),
        "updates": {"activity_weight": 0.08},
    },
    {
        "intent": "increase_material_development",
        "patterns": (
            r"\bfewer (?:sounds|samples|objects)\b",
            r"\bdevelop (?:the )?(?:selected )?(?:sounds|samples|materials?)\b",
            r"\bmore coherent palette\b",
            r"\bsmaller palette\b",
        ),
        "updates": {"material_development_weight": 0.08},
    },
)


def normalise_dream_level(value: Any) -> str | None:
    """Return D1/D3/D5 for a supported level representation."""
    if value is None:
        return None
    match = re.fullmatch(r"\s*[dD]?\s*([135])\s*", str(value))
    return f"D{match.group(1)}" if match else None


def feedback_target_scope(comment: str, default_target_level: Any = None) -> dict[str, Any]:
    """Route a comment to one D-level unless it explicitly requests global scope."""
    original = unicodedata.normalize("NFKC", str(comment)).casefold()
    ascii_text = _normalise_phrase(comment)
    explicit_global = bool(
        re.search(r"\b(?:globally|global|generally|all levels|every level|across all levels)\b", ascii_text)
        or re.search(r"\bγενικ(?:ά|α)\b", original)
        or "σε όλα τα επίπεδα" in original
        or "σε ολα τα επιπεδα" in original
    )
    explicit_level_match = re.search(r"\bd\s*([135])\b", ascii_text)
    explicit_level = f"D{explicit_level_match.group(1)}" if explicit_level_match else None
    default_level = normalise_dream_level(default_target_level)

    if explicit_global:
        return {
            "scope": "global",
            "target_level": None,
            "target_levels": list(DREAM_LEVELS),
            "target_source": "explicit_global_phrase",
        }
    if explicit_level:
        return {
            "scope": "level",
            "target_level": explicit_level,
            "target_levels": [explicit_level],
            "target_source": "explicit_comment_prefix",
        }
    if default_level:
        return {
            "scope": "level",
            "target_level": default_level,
            "target_levels": [default_level],
            "target_source": "render_report",
        }
    return {
        "scope": "unscoped",
        "target_level": None,
        "target_levels": [],
        "target_source": "none",
    }


def parse_feedback_comment(comment: str, default_target_level: Any = None) -> dict[str, Any]:
    """Map the bounded English MVP vocabulary and attach safe D-level routing."""
    normalised = _normalise_phrase(comment)
    target = feedback_target_scope(comment, default_target_level)
    actions: list[dict[str, Any]] = []
    combined: dict[str, float] = {}
    for rule in _INTENT_RULES:
        matched = next((pattern for pattern in rule["patterns"] if re.search(pattern, normalised)), None)
        if not matched:
            continue
        updates = dict(rule["updates"])
        actions.append({
            "intent": rule["intent"],
            "matched_pattern": matched,
            "control_deltas": updates,
            "implementation_note": rule.get("implementation_note"),
        })
        for control, delta in updates.items():
            combined[control] = combined.get(control, 0.0) + float(delta)

    return {
        "schema_version": 1,
        "original_text": comment,
        "normalised_text": normalised,
        "status": "interpreted" if actions else ("empty" if not normalised else "unrecognised"),
        "confidence": 1.0 if actions else 0.0,
        "actions": actions,
        "combined_control_deltas": combined,
        **target,
    }


def apply_control_deltas(
    weights: dict[str, Any],
    deltas: dict[str, float],
    lo: float = 0.5,
    hi: float = 1.8,
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for control, requested_delta in sorted(deltas.items()):
        old = float(weights.get(control, 1.0))
        new = max(lo, min(hi, old + float(requested_delta)))
        weights[control] = new
        updates.append({
            "control": control,
            "old_value": round(old, 6),
            "requested_delta": round(float(requested_delta), 6),
            "applied_delta": round(new - old, 6),
            "new_value": round(new, 6),
        })
    return updates


def build_legacy_sample_key_map(memory: Iterable[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for recording in memory:
        recording_name = str(recording.get("recording", ""))
        recording_id = str(recording.get("recording_id", recording_name))
        for obj in recording.get("objects", []):
            stable_id = str(obj.get("stable_id", obj.get("id")))
            stable = sample_key(recording_id, stable_id)
            legacy_id = obj.get("legacy_id", obj.get("id"))
            mapping[f"{recording_name}::{legacy_id}"] = stable
            mapping[stable] = stable
    return mapping


def migrate_sample_profile(profile: dict[str, Any], memory: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-key v1 sample evidence without discarding unresolvable entries."""
    profile = dict(profile) if isinstance(profile, dict) else {}
    original_version = int(profile.get("version", 1))
    mapping = build_legacy_sample_key_map(memory)
    migrated: dict[str, Any] = {}
    unresolved: dict[str, Any] = {}
    moved = 0

    for old_key, raw_entry in dict(profile.get("samples", {})).items():
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else {}
        new_key = mapping.get(str(old_key))
        if not new_key:
            unresolved[str(old_key)] = entry
            continue
        moved += int(new_key != old_key)
        current = migrated.get(new_key)
        if current is None:
            recording_id, _, object_id = new_key.partition("::")
            if old_key != new_key and "legacy_object_id" not in entry:
                entry["legacy_object_id"] = entry.get("object_id", old_key.partition("::")[2])
            entry["recording_id"] = recording_id
            entry["object_id"] = object_id
            entry["stable_key"] = new_key
            entry.setdefault("legacy_keys", [])
            if old_key != new_key:
                entry["legacy_keys"] = sorted(set(entry["legacy_keys"] + [old_key]))
            migrated[new_key] = entry
            continue

        old_updates = max(0, int(current.get("feedback_updates", 0)))
        new_updates = max(0, int(entry.get("feedback_updates", 0)))
        denominator = old_updates + new_updates
        if denominator:
            learned = (
                float(current.get("learned_value", 0.0)) * old_updates
                + float(entry.get("learned_value", 0.0)) * new_updates
            ) / denominator
        else:
            learned = (float(current.get("learned_value", 0.0)) + float(entry.get("learned_value", 0.0))) / 2.0
        current["learned_value"] = round(max(-1.0, min(1.0, learned)), 6)
        current["feedback_updates"] = old_updates + new_updates
        current["times_selected"] = int(current.get("times_selected", 0)) + int(entry.get("times_selected", 0))
        current["legacy_keys"] = sorted(set(current.get("legacy_keys", []) + entry.get("legacy_keys", []) + [old_key]))

    profile["version"] = SAMPLE_PROFILE_VERSION
    profile["description"] = "Hyponoia sample-level learning profile with stable content IDs."
    profile["samples"] = migrated
    profile["unresolved_legacy_samples"] = unresolved
    report = {
        "from_version": original_version,
        "to_version": SAMPLE_PROFILE_VERSION,
        "moved_entries": moved,
        "resolved_entries": len(migrated),
        "unresolved_entries": len(unresolved),
        "timestamp": utc_timestamp(),
    }
    if original_version < SAMPLE_PROFILE_VERSION or moved or unresolved:
        profile.setdefault("migration_history", []).append(report)
    return profile, report

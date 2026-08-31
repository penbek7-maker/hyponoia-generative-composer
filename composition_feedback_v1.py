"""Unified 1–5 composition feedback for the Hyponoia Deep Learning workflow."""

from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any
from uuid import uuid4

from human_feedback_v1 import DEFAULT_LEARNING_PROFILE, DEFAULT_WEIGHTS, append_feedback_history
from hyponoia_stability import apply_control_deltas, normalise_dream_level, parse_feedback_comment, utc_timestamp


COMPOSITION_CRITERIA = (
    "musicality",
    "material_coherence",
    "transition_smoothness",
    "variety_without_disconnection",
    "synth_material_presence",
    "development_over_repetition",
    "overall_artistic_impression",
)

RATING_CONTROL_MAP = {
    "musicality": {"musicality_weight": 1.0},
    "material_coherence": {"coherence_weight": 1.0},
    "transition_smoothness": {"transition_smoothness_weight": 1.0},
    "variety_without_disconnection": {"exploration_weight": 0.45, "coherence_weight": 0.35},
    "synth_material_presence": {"synthetic_material_weight": 1.0},
    "development_over_repetition": {"material_development_weight": 0.65, "repetition_control": 0.55},
}


def _field_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).lower()
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", without_marks).strip()


def directional_field_deltas(more: str, less: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Use the meaning of the More/Less field even when users enter only nouns."""
    rules = (
        ("more", r"\b(?:musicality|μουσικοτητα)\b", "increase_musicality", {"musicality_weight": 0.06}),
        ("more", r"\b(?:synth\w*|συνθε\w*|ηλεκτρονικ\w*)\b", "increase_synthetic_material", {"synthetic_material_weight": 0.08}),
        ("more", r"\b(?:layers?|ηχητικα επιπεδα|στρωματα|υφες)\b", "increase_richness", {"richness_weight": 0.07}),
        ("more", r"\b(?:development|αναπτυξη|εξελιξη)\b", "increase_material_development", {"material_development_weight": 0.08}),
        ("more", r"\b(?:smooth\w*|ομαλ\w* μεταβα\w*)\b", "increase_smoothness", {"transition_smoothness_weight": 0.08}),
        ("less", r"\b(?:bass|low end|low frequencies|μπασ\w*|χαμηλ\w* συχνοτ\w*)\b", "reduce_low_frequency_masking", {"low_frequency_control": 0.08}),
        ("less", r"\b(?:hidden|buried|κρυμμεν\w*|θαμμεν\w*)\b", "increase_layer_clarity", {"layer_clarity_weight": 0.08}),
        ("less", r"\b(?:repetition|repeats?|επαναληψ\w*)\b", "decrease_repetition", {"repetition_control": 0.10, "exploration_weight": 0.06}),
        ("less", r"\b(?:abrupt\w*|αποτομ\w*)\b", "increase_smoothness", {"transition_smoothness_weight": 0.08}),
    )
    texts = {"more": _field_text(more), "less": _field_text(less)}
    combined: dict[str, float] = {}
    actions = []
    for field, pattern, intent, updates in rules:
        if not re.search(pattern, texts[field]):
            continue
        actions.append({"field": field, "intent": intent, "matched_pattern": pattern, "control_deltas": updates})
        for control, delta in updates.items():
            combined[control] = combined.get(control, 0.0) + float(delta)
    return combined, actions


def validate_ratings(ratings: dict[str, Any]) -> dict[str, float]:
    missing = [criterion for criterion in COMPOSITION_CRITERIA if criterion not in ratings]
    if missing:
        raise ValueError(f"Missing composition ratings: {', '.join(missing)}")
    validated = {}
    for criterion in COMPOSITION_CRITERIA:
        try:
            value = float(ratings[criterion])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Rating {criterion} must be a number from 1 to 5") from exc
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"Rating {criterion} must be between 1 and 5")
        validated[criterion] = value
    return validated


def rating_control_deltas(ratings: dict[str, float], target: float = 4.0) -> dict[str, float]:
    """Turn deficits into small corrective requests; ratings never punish blindly."""
    deltas: dict[str, float] = {}
    for criterion, mappings in RATING_CONTROL_MAP.items():
        deficit = max(0.0, min(1.0, (target - ratings[criterion]) / (target - 1.0)))
        for control, multiplier in mappings.items():
            deltas[control] = deltas.get(control, 0.0) + 0.06 * deficit * multiplier
    if ratings["overall_artistic_impression"] < 3.0:
        severity = (3.0 - ratings["overall_artistic_impression"]) / 2.0
        deltas["noise_penalty"] = deltas.get("noise_penalty", 0.0) + 0.025 * severity
        deltas["impact_penalty"] = deltas.get("impact_penalty", 0.0) + 0.025 * severity
    return {key: round(value, 6) for key, value in sorted(deltas.items()) if value > 0}


def build_composition_feedback(
    ratings: dict[str, Any],
    *,
    dream_level: Any,
    keep_as_baseline: bool,
    more: str = "",
    less: str = "",
    comment: str = "",
    render_name: str | None = None,
) -> dict[str, Any]:
    validated = validate_ratings(ratings)
    level = normalise_dream_level(dream_level)
    if level is None:
        raise ValueError("dream_level must be D1, D3 or D5")
    combined_text = ". ".join(part.strip() for part in (more, less, comment) if part.strip())
    interpretation = parse_feedback_comment(combined_text, level)
    field_deltas, field_actions = directional_field_deltas(more, less)
    numeric_deltas = rating_control_deltas(validated)
    requested = dict(numeric_deltas)
    for control, delta in interpretation["combined_control_deltas"].items():
        requested[control] = round(requested.get(control, 0.0) + float(delta), 6)
    for control, delta in field_deltas.items():
        if control not in interpretation["combined_control_deltas"]:
            requested[control] = round(requested.get(control, 0.0) + float(delta), 6)
    return {
        "schema_version": "composition_feedback_v1",
        "event_id": f"cf_{uuid4().hex}",
        "timestamp": utc_timestamp(),
        "render_name": render_name,
        "dream_level": level,
        "accepted_as_aesthetic_baseline": bool(keep_as_baseline),
        "ratings_1_to_5": validated,
        "ratings_0_to_100": {key: round(value * 20.0, 3) for key, value in validated.items()},
        "listener_text": {"more": more, "less": less, "comment": comment},
        "comment_interpretation": interpretation,
        "directional_field_interpretation": {"actions": field_actions, "combined_control_deltas": field_deltas},
        "numeric_control_deltas": numeric_deltas,
        "requested_control_deltas": requested,
        "learning_policy": {
            "scope": "dream_level_only",
            "ratings_are_primary": True,
            "critic_can_override": False,
            "bounded_update": True,
        },
    }


def apply_composition_feedback(
    profile: dict[str, Any] | None,
    event: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply one explicit event to its D-level and append immutable evidence."""
    updated = copy.deepcopy(profile if profile is not None else DEFAULT_LEARNING_PROFILE)
    weights = updated.setdefault("weights", {})
    for key, value in DEFAULT_WEIGHTS.items():
        weights.setdefault(key, value)
    level_weights = updated.setdefault("level_weights", {})
    for level in ("D1", "D3", "D5"):
        level_profile = level_weights.setdefault(level, {})
        for key, value in DEFAULT_WEIGHTS.items():
            level_profile.setdefault(key, value)

    level = normalise_dream_level(event.get("dream_level"))
    if level is None:
        raise ValueError("Composition feedback event has no valid dream_level")
    updates = apply_control_deltas(level_weights[level], event.get("requested_control_deltas", {}))
    for update in updates:
        update["target_level"] = level
    stored_event = copy.deepcopy(event)
    stored_event["applied_control_updates"] = updates
    append_feedback_history(updated, stored_event)
    return updated, updates

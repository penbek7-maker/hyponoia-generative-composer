"""Preview and apply free-text or transcribed-voice feedback safely.

Both input modes use the same local interpreter path. The UI can use a small
local language model with a deterministic rules fallback. A preview is always
produced before profile mutation so the listener can see the understood
intents, target D-level and bounded control changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from human_feedback_v1 import DEFAULT_LEARNING_PROFILE, DEFAULT_WEIGHTS, append_feedback_history
from hyponoia_stability import (
    DREAM_LEVELS,
    apply_control_deltas,
    atomic_write_json,
    parse_feedback_comment,
    utc_timestamp,
)
from local_llm_feedback_v1 import LocalLLMUnavailable, interpret_with_local_llm


INPUT_SOURCES = ("text", "voice")

INTENT_LABELS_EL = {
    "increase_musicality": "περισσότερη μουσικότητα",
    "increase_rhythmicity": "περισσότερη ρυθμική κίνηση",
    "increase_bloom": "μεγαλύτερη ανάπτυξη/bloom",
    "increase_synthetic_material": "περισσότερο συνθετικό υλικό",
    "increase_arpeggios": "περισσότερα arpeggios",
    "increase_layer_clarity": "καθαρότερα ηχητικά επίπεδα",
    "diversify_long_layers": "διαφορετικά μεγάλα layers",
    "increase_library_exploration": "μεγαλύτερη εξερεύνηση της βιβλιοθήκης",
    "increase_palette_variety": "περισσότερη ποικιλία",
    "decrease_repetition": "λιγότερη επανάληψη",
    "increase_smoothness": "ομαλότερες μεταβάσεις",
    "increase_richness": "πλουσιότερα ηχητικά επίπεδα",
    "increase_activity": "περισσότερη ενέργεια",
    "increase_material_development": "περισσότερη ανάπτυξη του υλικού",
    "reduce_low_frequency_masking": "λιγότερη κάλυψη από χαμηλές συχνότητες",
}


def _normalise_profile(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    profile = copy.deepcopy(data)
    profile.setdefault("history", [])
    shared = profile.setdefault("weights", {})
    for control, default in DEFAULT_WEIGHTS.items():
        shared.setdefault(control, default)
    levels = profile.setdefault("level_weights", {})
    for level in DREAM_LEVELS:
        current = levels.setdefault(level, {})
        for control, default in DEFAULT_WEIGHTS.items():
            current.setdefault(control, default)
    return profile


def _load_profile(profile_path: str | Path | None) -> dict[str, Any]:
    if profile_path is None or not Path(profile_path).exists():
        return _normalise_profile(DEFAULT_LEARNING_PROFILE)
    try:
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid learning profile: {profile_path}") from exc
    return _normalise_profile(profile)


def build_feedback_preview(
    text: str,
    *,
    dream_level: Any,
    source: str = "text",
    locale: str | None = None,
    profile: dict[str, Any] | None = None,
    interpreter: str = "rules",
    llm_interpreter: Any = interpret_with_local_llm,
) -> dict[str, Any]:
    """Return an explainable preview without mutating the learning profile."""
    source = str(source).strip().lower()
    if source not in INPUT_SOURCES:
        raise ValueError("source must be text or voice")
    transcript = str(text).strip()
    if interpreter not in {"rules", "local_llm", "auto"}:
        raise ValueError("interpreter must be rules, local_llm or auto")
    fallback_reason = None
    if interpreter in {"local_llm", "auto"}:
        try:
            interpretation = llm_interpreter(transcript, dream_level)
        except LocalLLMUnavailable as exc:
            if interpreter == "local_llm":
                raise ValueError(str(exc)) from exc
            interpretation = parse_feedback_comment(transcript, dream_level)
            interpretation["interpreter"] = "rules_fallback"
            interpretation["model"] = None
            fallback_reason = str(exc)
    else:
        interpretation = parse_feedback_comment(transcript, dream_level)
        interpretation["interpreter"] = "rules"
        interpretation["model"] = None
    current_profile = (
        _normalise_profile(DEFAULT_LEARNING_PROFILE)
        if profile is None
        else _normalise_profile(profile)
    )
    proposed = copy.deepcopy(current_profile)
    changes: list[dict[str, Any]] = []

    if interpretation["status"] == "interpreted":
        for level in interpretation.get("target_levels", []):
            if level not in DREAM_LEVELS:
                continue
            for change in apply_control_deltas(
                proposed["level_weights"][level],
                interpretation["combined_control_deltas"],
            ):
                change["target_level"] = level
                changes.append(change)

    actions = []
    for action in interpretation.get("actions", []):
        intent = action["intent"]
        actions.append({
            "intent": intent,
            "label_el": INTENT_LABELS_EL.get(intent, intent),
            "control_deltas": dict(action.get("control_deltas", {})),
        })

    return {
        "schema_version": "feedback_input_preview_v1",
        "source": source,
        "locale": locale,
        "transcript": transcript,
        "status": interpretation["status"],
        "can_apply": interpretation["status"] == "interpreted" and bool(changes),
        "scope": interpretation["scope"],
        "target_levels": list(interpretation.get("target_levels", [])),
        "actions": actions,
        "control_changes": changes,
        "interpretation": interpretation,
        "interpreter": interpretation.get("interpreter", "rules"),
        "model": interpretation.get("model"),
        "confidence": float(interpretation.get("confidence", 0.0)),
        "summary_el": interpretation.get("summary_el"),
        "ambiguities": list(interpretation.get("ambiguities", [])),
        "fallback_reason": fallback_reason,
    }


def apply_feedback_preview(
    preview: dict[str, Any],
    *,
    profile_path: str | Path = "learning_profile.json",
    evidence_dir: str | Path = "human_feedback",
    confirmed: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist a previously shown preview after explicit listener confirmation."""
    if not confirmed:
        raise ValueError("Feedback must be previewed and explicitly confirmed before applying")
    if preview.get("schema_version") != "feedback_input_preview_v1":
        raise ValueError("Unsupported feedback preview")
    if not preview.get("can_apply"):
        raise ValueError("This feedback has no recognised changes to apply")

    profile_target = Path(profile_path)
    profile = _load_profile(profile_target)
    applied: list[dict[str, Any]] = []
    interpretation = preview.get("interpretation", {})
    for level in preview.get("target_levels", []):
        if level not in DREAM_LEVELS:
            continue
        for change in apply_control_deltas(
            profile["level_weights"][level],
            interpretation.get("combined_control_deltas", {}),
        ):
            change["target_level"] = level
            applied.append(change)

    event = {
        "schema_version": "feedback_input_event_v1",
        "event_id": f"fi_{uuid4().hex}",
        "timestamp": utc_timestamp(),
        "source": preview["source"],
        "locale": preview.get("locale"),
        "transcript": preview["transcript"],
        "scope": preview["scope"],
        "target_levels": list(preview["target_levels"]),
        "actions": copy.deepcopy(preview["actions"]),
        "interpreter": preview.get("interpreter", "rules"),
        "model": preview.get("model"),
        "confidence": preview.get("confidence"),
        "summary_el": preview.get("summary_el"),
        "ambiguities": copy.deepcopy(preview.get("ambiguities", [])),
        "applied_control_updates": applied,
        "policy": {
            "preview_required": True,
            "explicit_confirmation": True,
            "bounded_update": True,
            "shared_text_voice_interpreter": True,
        },
    }
    append_feedback_history(profile, event)
    atomic_write_json(profile_target, profile)
    evidence_target = Path(evidence_dir) / f"{event['event_id']}_free_feedback.json"
    atomic_write_json(evidence_target, event)
    return profile, event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Greek or English feedback text/transcript")
    parser.add_argument("--level", required=True, choices=DREAM_LEVELS)
    parser.add_argument("--source", choices=INPUT_SOURCES, default="text")
    parser.add_argument("--locale")
    parser.add_argument("--interpreter", choices=("rules", "local_llm", "auto"), default="auto")
    parser.add_argument("--profile", type=Path, default=Path("learning_profile.json"))
    parser.add_argument("--apply", action="store_true", help="Apply after printing the preview")
    args = parser.parse_args()
    profile = _load_profile(args.profile)
    preview = build_feedback_preview(
        args.text,
        dream_level=args.level,
        source=args.source,
        locale=args.locale,
        profile=profile,
        interpreter=args.interpreter,
    )
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    if args.apply:
        _profile, event = apply_feedback_preview(
            preview,
            profile_path=args.profile,
            confirmed=True,
        )
        print(json.dumps({"applied_event": event["event_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

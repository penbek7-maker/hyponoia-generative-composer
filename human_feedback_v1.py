# Hyponoia Human Feedback v2 stabilisation baseline.
# Human ratings are primary; Critic disagreement remains diagnostic evidence.

import os
import sys
import json
from uuid import uuid4

from hyponoia_stability import (
    DREAM_LEVELS,
    LEARNING_SCHEMA_VERSION,
    SAMPLE_PROFILE_VERSION,
    apply_control_deltas,
    atomic_write_json,
    migrate_sample_profile,
    parse_feedback_comment,
    utc_timestamp,
)


LEARNING_FILE = "learning_profile.json"
FEEDBACK_FOLDER = "human_feedback"
SAMPLE_LEARNING_FILE = "sample_learning_profile.json"
RENDER_REPORT_FILE = "render_report.json"
RENDER_REPORT_FOLDER = "render_reports"
MEMORY_FILE = "memory_index_v3.json"


CRITERIA = [
    "musicality",
    "coherence",
    "richness",
    "transitions",
    "bloom_quality",
    "overall",
]


DEFAULT_WEIGHTS = {
    "musicality_weight": 1.0,
    "coherence_weight": 1.0,
    "richness_weight": 1.0,
    "transition_smoothness_weight": 1.0,
    "bloom_weight": 1.0,
    "ambient_weight": 1.0,
    "gesture_weight": 1.0,
    "noise_penalty": 1.0,
    "impact_penalty": 1.0,
    "exploration_weight": 1.0,
    "repetition_control": 1.0,
    "synthetic_material_weight": 1.0,
    "arpeggio_weight": 1.0,
    "long_layer_diversity_weight": 1.0,
    "activity_weight": 1.0,
    "material_development_weight": 1.0,
    "low_frequency_control": 1.0,
    "layer_clarity_weight": 1.0,
}


DEFAULT_LEARNING_PROFILE = {
    "version": LEARNING_SCHEMA_VERSION,
    "description": "Hyponoia explainable learning controls. Human ratings are primary; Critic residuals are diagnostic.",
    "weights": dict(DEFAULT_WEIGHTS),
    "level_weights": {level: dict(DEFAULT_WEIGHTS) for level in DREAM_LEVELS},
    "history": []
}


def clamp(x, lo=0.5, hi=1.8):
    return float(max(lo, min(hi, x)))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_learning_profile():
    if not os.path.exists(LEARNING_FILE):
        return json.loads(json.dumps(DEFAULT_LEARNING_PROFILE))
    try:
        with open(LEARNING_FILE, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_LEARNING_PROFILE))
    profile.setdefault("history", [])
    weights = profile.setdefault("weights", {})
    for key, value in DEFAULT_WEIGHTS.items():
        weights.setdefault(key, value)
    level_weights = profile.setdefault("level_weights", {})
    for level in DREAM_LEVELS:
        level_profile = level_weights.setdefault(level, {})
        for key, value in DEFAULT_WEIGHTS.items():
            level_profile.setdefault(key, value)
    profile["version"] = LEARNING_SCHEMA_VERSION
    return profile


def save_learning_profile(profile):
    atomic_write_json(LEARNING_FILE, profile)


def load_sample_learning_profile():
    default = {
        "version": SAMPLE_PROFILE_VERSION,
        "description": "Hyponoia sample-level learning profile with stable content IDs.",
        "total_render_selections": 0,
        "samples": {},
    }
    if not os.path.exists(SAMPLE_LEARNING_FILE):
        return default
    try:
        with open(SAMPLE_LEARNING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        data.setdefault("samples", {})
        data.setdefault("total_render_selections", 0)
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as memory_handle:
                memory = json.load(memory_handle)
            data, _ = migrate_sample_profile(data, memory)
        return data
    except (OSError, json.JSONDecodeError):
        return default


def save_sample_learning_profile(profile):
    atomic_write_json(SAMPLE_LEARNING_FILE, profile)


def find_render_report(audio_file):
    if audio_file:
        base = os.path.splitext(os.path.basename(audio_file))[0]
        candidate = os.path.join(RENDER_REPORT_FOLDER, f"{base}_render_report.json")
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(RENDER_REPORT_FILE):
        try:
            latest = load_json(RENDER_REPORT_FILE)
            if not audio_file or os.path.basename(latest.get("audio_file", "")) == os.path.basename(audio_file):
                return RENDER_REPORT_FILE
        except (OSError, json.JSONDecodeError):
            pass
    return None


def update_sample_values(sample_profile, render_report, human, internal):
    """Assign signed, exposure- and role-aware credit to material in one render."""
    samples_used = render_report.get("samples", {})
    if not samples_used:
        return {}

    details = render_report.get("sample_usage_details", {})
    overall_signal = max(-1.0, min(1.0, (float(human["overall"]) - 50.0) / 50.0))
    critic_residual = max(-1.0, min(1.0, (float(human["overall"]) - float(internal["overall"])) / 50.0))
    learning_rate = 0.035
    max_count = max(int(v) for v in samples_used.values())
    max_exposure = max(
        [float(item.get("exposure_sec", 0.0)) for item in details.values()] + [1.0]
    )
    max_average_gain = max(
        [
            float(item.get("gain_sum", 0.0)) / max(1, int(item.get("selection_count", 1)))
            for item in details.values()
        ]
        + [0.01]
    )

    role_dimension = {
        "gesture": "musicality",
        "texture": "richness",
        "resonance": "bloom_quality",
        "noise": "coherence",
        "impact": "transitions",
    }

    updates = {}
    profile_samples = sample_profile.setdefault("samples", {})
    for key, count in samples_used.items():
        detail = details.get(key, {})
        role_counts = detail.get("role_counts", {})
        role_total = sum(max(0, int(v)) for v in role_counts.values())
        if role_total:
            role_signal = sum(
                max(0, int(role_count))
                * ((float(human[role_dimension.get(role, "overall")]) - 50.0) / 50.0)
                for role, role_count in role_counts.items()
            ) / role_total
        else:
            role_signal = overall_signal

        count_component = (max(1, int(count)) / max_count) ** 0.5
        exposure_component = min(1.0, float(detail.get("exposure_sec", 0.0)) / max_exposure)
        average_gain = float(detail.get("gain_sum", 0.0)) / max(1, int(detail.get("selection_count", count)))
        gain_component = min(1.0, average_gain / max_average_gain)
        credit = max(0.15, min(1.0, 0.45 * count_component + 0.40 * exposure_component + 0.15 * gain_component))
        signal = max(-1.0, min(1.0, 0.72 * overall_signal + 0.23 * role_signal + 0.05 * critic_residual))

        recording_id, _, object_id = key.partition("::")
        entry = profile_samples.setdefault(key, {
            "recording_id": recording_id,
            "object_id": object_id or key,
            "learned_value": 0.0,
            "times_selected": 0,
            "feedback_updates": 0,
            "last_used": None,
        })
        previous_updates = int(entry.get("feedback_updates", 0))
        uncertainty = max(0.08, min(1.0, (1.0 / (previous_updates + 1) ** 0.5) * (1.10 - 0.35 * credit)))
        delta = learning_rate * signal * credit
        old_value = float(entry.get("learned_value", 0.0))
        new_value = max(-1.0, min(1.0, old_value + delta))
        entry["learned_value"] = round(new_value, 6)
        entry["feedback_updates"] = previous_updates + 1
        entry["uncertainty"] = round(uncertainty, 6)
        entry["last_feedback"] = utc_timestamp()
        updates[key] = {
            "count_in_render": int(count),
            "exposure_sec": round(float(detail.get("exposure_sec", 0.0)), 6),
            "roles": role_counts,
            "overall_signal": round(overall_signal, 6),
            "role_signal": round(role_signal, 6),
            "critic_residual": round(critic_residual, 6),
            "credit": round(credit, 6),
            "uncertainty": round(uncertainty, 6),
            "old_value": round(old_value, 6),
            "delta": round(delta, 6),
            "new_value": round(new_value, 6),
        }
    return updates


def ask_score(name):
    while True:
        raw = input(f"{name} 0-100: ").strip()
        try:
            value = float(raw)
            if 0 <= value <= 100:
                return value
        except ValueError:
            pass
        print("Please enter a number between 0 and 100.")


def update_weights(profile, internal, human):
    """Apply directional corrective controls; keep Critic residuals diagnostic.

    A low human quality rating now increases the control that seeks that quality
    (for example, low transition quality requests more smoothing).  The old
    implementation used human-minus-Critic residual directly, which could move
    a quality control in the wrong musical direction.
    """
    weights = profile["weights"]
    diffs = {k: human[k] - internal[k] for k in CRITERIA}
    target = 75.0
    base_rate = 0.025
    mappings = {
        "musicality": "musicality_weight",
        "coherence": "coherence_weight",
        "richness": "richness_weight",
        "transitions": "transition_smoothness_weight",
        "bloom_quality": "bloom_weight",
    }
    requested = {}
    reasons = {}
    for criterion, control in mappings.items():
        deficit = max(0.0, min(1.0, (target - float(human[criterion])) / target))
        if deficit > 0:
            requested[control] = base_rate * deficit
            reasons[control] = f"human {criterion} score {human[criterion]:.2f} below target {target:.0f}"

    if float(human["overall"]) < 65.0:
        severity = min(1.0, (65.0 - float(human["overall"])) / 65.0)
        requested["noise_penalty"] = requested.get("noise_penalty", 0.0) + 0.015 * severity
        requested["impact_penalty"] = requested.get("impact_penalty", 0.0) + 0.015 * severity
        reasons["noise_penalty"] = "low overall human rating: reduce uncontrolled noise"
        reasons["impact_penalty"] = "low overall human rating: reduce abrupt impacts"

    control_updates = apply_control_deltas(weights, requested)
    for update in control_updates:
        update["reason"] = reasons.get(update["control"], "human-rating corrective control")

    return {
        "signed_residuals_human_minus_critic": {k: round(v, 6) for k, v in diffs.items()},
        "corrective_control_updates": control_updates,
        "target_score": target,
    }


def apply_comment_to_profile(profile, interpretation):
    """Apply text intent only to its routed D-level profile.

    Numerical ratings continue to update the shared ``weights`` profile. Text
    intent never changes shared weights, and only an explicit global phrase may
    update all three level profiles.
    """
    level_weights = profile.setdefault("level_weights", {})
    for level in DREAM_LEVELS:
        level_profile = level_weights.setdefault(level, {})
        for key, value in DEFAULT_WEIGHTS.items():
            level_profile.setdefault(key, value)

    updates = []
    for level in interpretation.get("target_levels", []):
        if level not in DREAM_LEVELS:
            continue
        for update in apply_control_deltas(
            level_weights[level],
            interpretation.get("combined_control_deltas", {}),
        ):
            update["target_level"] = level
            updates.append(update)
    return updates


def save_feedback_entry(entry):
    os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
    base = os.path.splitext(os.path.basename(entry["critic_report"]))[0]
    out = os.path.join(FEEDBACK_FOLDER, f"{base}_{entry['event_id']}_human_feedback.json")
    if os.path.exists(out):
        raise FileExistsError(f"Feedback event already exists: {out}")
    atomic_write_json(out, entry)
    return out


def append_feedback_history(profile, entry):
    """Append one unique event without replacing earlier history evidence."""
    history = profile.setdefault("history", [])
    event_id = entry.get("event_id")
    if not event_id:
        raise ValueError("Feedback event requires a non-empty event_id")
    if any(item.get("event_id") == event_id for item in history if isinstance(item, dict)):
        raise ValueError(f"Duplicate feedback event_id: {event_id}")
    history.append(entry)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 human_feedback_v1.py critic_reports/render_critic.json")
        sys.exit(1)

    critic_path = sys.argv[1]
    if not os.path.exists(critic_path):
        print("Critic report not found:", critic_path)
        sys.exit(1)

    critic = load_json(critic_path)
    internal = critic["internal_scores"]

    print()
    print("Internal Critic Scores")
    print("----------------------")
    for k in CRITERIA:
        print(f"{k}: {internal[k]}")

    print()
    print("Human Feedback")
    print("Give 0-100 for the SAME criteria.")
    print()

    human = {}
    for k in CRITERIA:
        human[k] = ask_score(k)

    comment = input("Optional comment: ").strip()

    render_report_path = find_render_report(critic.get("audio_file"))
    render_report = load_json(render_report_path) if render_report_path else None
    render_level = render_report.get("dream_level") if render_report else None

    profile = load_learning_profile()
    rating_update = update_weights(profile, internal, human)
    comment_interpretation = parse_feedback_comment(comment, render_level)
    phrase_control_updates = apply_comment_to_profile(profile, comment_interpretation)

    sample_updates = {}
    if render_report is not None:
        sample_profile = load_sample_learning_profile()
        sample_updates = update_sample_values(sample_profile, render_report, human, internal)
        save_sample_learning_profile(sample_profile)
    else:
        print("No matching render report found; global weights will still be updated.")

    event_id = f"fb_{uuid4().hex}"
    entry = {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp": utc_timestamp(),
        "critic_report": critic_path,
        "audio_file": critic.get("audio_file"),
        "rating_schema": {
            "scale_min": 0,
            "scale_max": 100,
            "primary_target": "human_scores",
            "critic_role": "auxiliary_diagnostic_signal",
            "criteria": CRITERIA,
        },
        "internal_scores": internal,
        "human_scores": human,
        "differences_human_minus_internal": rating_update["signed_residuals_human_minus_critic"],
        "comment": comment,
        "comment_interpretation": comment_interpretation,
        "learning_update": rating_update,
        "phrase_control_updates": phrase_control_updates,
        "updated_weights": dict(profile["weights"]),
        "updated_level_weights": {
            level: dict(profile["level_weights"][level]) for level in DREAM_LEVELS
        },
        "render_report": render_report_path,
        "sample_updates": sample_updates,
    }

    append_feedback_history(profile, entry)
    save_learning_profile(profile)
    feedback_path = save_feedback_entry(entry)

    print()
    print("Human - Internal differences")
    print("----------------------------")
    for k, v in rating_update["signed_residuals_human_minus_critic"].items():
        print(f"{k}: {round(v, 2)}")

    print()
    print("Comment interpretation")
    print("----------------------")
    print(comment_interpretation["status"])
    for action in comment_interpretation["actions"]:
        print(f"- {action['intent']}: {action['control_deltas']}")
    print(
        "scope:", comment_interpretation["scope"],
        "targets:", comment_interpretation["target_levels"],
    )

    print()
    print("Updated learning weights")
    print("------------------------")
    for k, v in profile["weights"].items():
        print(f"{k}: {round(v, 3)}")

    print()
    print("Saved:")
    print(feedback_path)
    print(LEARNING_FILE)
    if sample_updates:
        print(SAMPLE_LEARNING_FILE)
        print(f"Updated sample values: {len(sample_updates)}")


if __name__ == "__main__":
    main()

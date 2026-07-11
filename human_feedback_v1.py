# Hyponoia Human Feedback v1
# Human and system score the same criteria.
# This updates learning_profile.json using the disagreement between internal critic and human feedback.

import os
import sys
import json
from datetime import datetime


LEARNING_FILE = "learning_profile.json"
FEEDBACK_FOLDER = "human_feedback"
SAMPLE_LEARNING_FILE = "sample_learning_profile.json"
RENDER_REPORT_FILE = "render_report.json"
RENDER_REPORT_FOLDER = "render_reports"


CRITERIA = [
    "musicality",
    "coherence",
    "richness",
    "transitions",
    "bloom_quality",
    "overall",
]


DEFAULT_LEARNING_PROFILE = {
    "version": 1,
    "description": "Hyponoia learning profile. Updated from human/system critic agreement.",
    "weights": {
        "musicality_weight": 1.0,
        "coherence_weight": 1.0,
        "richness_weight": 1.0,
        "transition_smoothness_weight": 1.0,
        "bloom_weight": 1.0,
        "ambient_weight": 1.0,
        "gesture_weight": 1.0,
        "noise_penalty": 1.0,
        "impact_penalty": 1.0,
    },
    "history": []
}


def clamp(x, lo=0.5, hi=1.8):
    return float(max(lo, min(hi, x)))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_learning_profile():
    if not os.path.exists(LEARNING_FILE):
        return dict(DEFAULT_LEARNING_PROFILE)
    with open(LEARNING_FILE, "r") as f:
        return json.load(f)


def save_learning_profile(profile):
    with open(LEARNING_FILE, "w") as f:
        json.dump(profile, f, indent=4)


def load_sample_learning_profile():
    default = {
        "version": 1,
        "description": "Hyponoia sample-level learning profile.",
        "total_render_selections": 0,
        "samples": {},
    }
    if not os.path.exists(SAMPLE_LEARNING_FILE):
        return default
    try:
        with open(SAMPLE_LEARNING_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("samples", {})
        data.setdefault("total_render_selections", 0)
        return data
    except (OSError, json.JSONDecodeError):
        return default


def save_sample_learning_profile(profile):
    with open(SAMPLE_LEARNING_FILE, "w") as f:
        json.dump(profile, f, indent=4)


def find_render_report(audio_file):
    if audio_file:
        base = os.path.splitext(os.path.basename(audio_file))[0]
        candidate = os.path.join(RENDER_REPORT_FOLDER, f"{base}_render_report.json")
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(RENDER_REPORT_FILE):
        return RENDER_REPORT_FILE
    return None


def update_sample_values(sample_profile, render_report, human, internal):
    """Apply a small, usage-weighted reward to every sample used in the render."""
    samples_used = render_report.get("samples", {})
    if not samples_used:
        return {}

    # Human overall quality is primary; disagreement with the critic adds a smaller correction.
    quality_signal = (float(human["overall"]) - 50.0) / 50.0
    disagreement_signal = (float(human["overall"]) - float(internal["overall"])) / 50.0
    signal = max(-1.0, min(1.0, 0.75 * quality_signal + 0.25 * disagreement_signal))
    learning_rate = 0.020
    max_count = max(int(v) for v in samples_used.values())

    updates = {}
    profile_samples = sample_profile.setdefault("samples", {})
    for key, count in samples_used.items():
        entry = profile_samples.setdefault(key, {
            "recording": key.split("::", 1)[0],
            "object_id": key.split("::", 1)[1] if "::" in key else key,
            "learned_value": 0.0,
            "times_selected": 0,
            "feedback_updates": 0,
            "last_used": None,
        })
        usage_weight = (max(1, int(count)) / max_count) ** 0.5
        delta = learning_rate * signal * usage_weight
        old_value = float(entry.get("learned_value", 0.0))
        new_value = max(-1.0, min(1.0, old_value + delta))
        entry["learned_value"] = round(new_value, 6)
        entry["feedback_updates"] = int(entry.get("feedback_updates", 0)) + 1
        entry["last_feedback"] = datetime.now().isoformat(timespec="seconds")
        updates[key] = {
            "count_in_render": int(count),
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
    weights = profile["weights"]

    diffs = {k: human[k] - internal[k] for k in CRITERIA}
    lr = 0.015

    weights["musicality_weight"] = clamp(weights["musicality_weight"] + lr * diffs["musicality"] / 10.0)
    weights["coherence_weight"] = clamp(weights["coherence_weight"] + lr * diffs["coherence"] / 10.0)
    weights["richness_weight"] = clamp(weights["richness_weight"] + lr * diffs["richness"] / 10.0)
    weights["transition_smoothness_weight"] = clamp(weights["transition_smoothness_weight"] + lr * diffs["transitions"] / 10.0)
    weights["bloom_weight"] = clamp(weights["bloom_weight"] + lr * diffs["bloom_quality"] / 10.0)

    if human["richness"] > internal["richness"] + 10:
        weights["ambient_weight"] = clamp(weights["ambient_weight"] + 0.015)

    if human["transitions"] < internal["transitions"] - 10:
        weights["transition_smoothness_weight"] = clamp(weights["transition_smoothness_weight"] + 0.020)
        weights["impact_penalty"] = clamp(weights["impact_penalty"] + 0.015)

    if human["overall"] < internal["overall"] - 12:
        weights["noise_penalty"] = clamp(weights["noise_penalty"] + 0.015)
        weights["impact_penalty"] = clamp(weights["impact_penalty"] + 0.015)

    if human["musicality"] > internal["musicality"] + 10:
        weights["gesture_weight"] = clamp(weights["gesture_weight"] + 0.010)

    return diffs


def save_feedback_entry(entry):
    os.makedirs(FEEDBACK_FOLDER, exist_ok=True)
    base = os.path.splitext(os.path.basename(entry["critic_report"]))[0]
    out = os.path.join(FEEDBACK_FOLDER, f"{base}_human_feedback.json")
    with open(out, "w") as f:
        json.dump(entry, f, indent=4)
    return out


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

    profile = load_learning_profile()
    diffs = update_weights(profile, internal, human)

    render_report_path = find_render_report(critic.get("audio_file"))
    sample_updates = {}
    if render_report_path:
        render_report = load_json(render_report_path)
        sample_profile = load_sample_learning_profile()
        sample_updates = update_sample_values(sample_profile, render_report, human, internal)
        save_sample_learning_profile(sample_profile)
    else:
        print("No matching render report found; global weights will still be updated.")

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "critic_report": critic_path,
        "audio_file": critic.get("audio_file"),
        "internal_scores": internal,
        "human_scores": human,
        "differences_human_minus_internal": {k: round(v, 2) for k, v in diffs.items()},
        "comment": comment,
        "updated_weights": profile["weights"],
        "render_report": render_report_path,
        "sample_updates": sample_updates,
    }

    profile["history"].append(entry)
    save_learning_profile(profile)
    feedback_path = save_feedback_entry(entry)

    print()
    print("Human - Internal differences")
    print("----------------------------")
    for k, v in diffs.items():
        print(f"{k}: {round(v, 2)}")

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

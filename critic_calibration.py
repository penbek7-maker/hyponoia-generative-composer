"""Build transparent Critic-vs-human calibration metrics from feedback history."""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from hyponoia_stability import atomic_write_json, utc_timestamp


CRITERIA = ("musicality", "coherence", "richness", "transitions", "bloom_quality", "overall")


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def _rank_correlation(left, right):
    if len(left) < 3:
        return None
    left_rank = _rank(left)
    right_rank = _rank(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def build_calibration(feedback_folder):
    pairs = {criterion: {"critic": [], "human": []} for criterion in CRITERIA}
    files_read = 0
    skipped_files = []

    for path in sorted(glob.glob(os.path.join(feedback_folder, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                entry = json.load(handle)
            internal = entry.get("internal_scores", {})
            human = entry.get("human_scores", {})
            paired_dimensions = 0
            for criterion in CRITERIA:
                if criterion not in internal or criterion not in human:
                    continue
                critic_value = float(internal[criterion])
                human_value = float(human[criterion])
                if not np.isfinite(critic_value) or not np.isfinite(human_value):
                    continue
                pairs[criterion]["critic"].append(critic_value)
                pairs[criterion]["human"].append(human_value)
                paired_dimensions += 1
            if paired_dimensions:
                files_read += 1
            else:
                skipped_files.append({"file": path, "error": "No paired Critic/human dimensions"})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            skipped_files.append({"file": path, "error": f"{type(exc).__name__}: {exc}"})

    metrics = {}
    for criterion, values in pairs.items():
        critic = np.asarray(values["critic"], dtype=np.float64)
        human = np.asarray(values["human"], dtype=np.float64)
        if len(critic) == 0:
            metrics[criterion] = {
                "n": 0,
                "critic_variance": None,
                "human_variance": None,
                "mae": None,
                "mean_signed_error_critic_minus_human": None,
                "rank_correlation": None,
            }
            continue
        error = critic - human
        metrics[criterion] = {
            "n": int(len(critic)),
            "critic_mean": round(float(np.mean(critic)), 6),
            "human_mean": round(float(np.mean(human)), 6),
            "critic_variance": round(float(np.var(critic)), 6),
            "human_variance": round(float(np.var(human)), 6),
            "mae": round(float(np.mean(np.abs(error))), 6),
            "mean_signed_error_critic_minus_human": round(float(np.mean(error)), 6),
            "rank_correlation": (
                None
                if _rank_correlation(critic, human) is None
                else round(float(_rank_correlation(critic, human)), 6)
            ),
        }

    return {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "feedback_folder": feedback_folder,
        "feedback_events": files_read,
        "dimension_event_counts": {
            criterion: len(values["critic"]) for criterion, values in pairs.items()
        },
        "metrics": metrics,
        "skipped_files": skipped_files,
        "interpretation": {
            "mae": "Lower is better.",
            "rank_correlation": "Higher positive values indicate better agreement in ordering; at least three non-constant pairs are required.",
            "variance": "Near-zero Critic variance indicates score compression or a ceiling/floor problem.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-folder", default="human_feedback")
    parser.add_argument("--output", default="critic_calibration_report.json")
    args = parser.parse_args()
    report = build_calibration(args.feedback_folder)
    atomic_write_json(args.output, report)
    print(f"Saved: {args.output}")
    print(f"Feedback events: {report['feedback_events']}")


if __name__ == "__main__":
    main()

"""Read-only GATE 1 checks for a composer's private Hyponoia project data."""

from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

import generator_v3_memory_bloom_smooth as generator
from human_feedback_v1 import (
    CRITERIA,
    DEFAULT_LEARNING_PROFILE,
    apply_comment_to_profile,
    update_sample_values,
)
from hyponoia_stability import TARGET_SR, atomic_write_json, parse_feedback_comment, utc_timestamp


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _score_set(value):
    return {criterion: float(value) for criterion in CRITERIA}


def _latest_reports_by_level(project_dir: Path):
    reports = {}
    paths = [Path(path) for path in glob.glob(str(project_dir / "render_reports" / "*_render_report.json"))]
    latest = project_dir / "render_report.json"
    if latest.exists():
        paths.append(latest)
    for path in paths:
        report = _read_json(path)
        if not isinstance(report, dict):
            continue
        try:
            level = f"D{int(report.get('dream_level'))}"
        except (TypeError, ValueError):
            continue
        if level not in ("D1", "D3", "D5"):
            continue
        current = reports.get(level)
        if current is None or path.stat().st_mtime > current[0].stat().st_mtime:
            reports[level] = (path, report)
    return reports


def _controlled_real_report_cycle(sample_profile, render_report):
    positive = copy.deepcopy(sample_profile)
    negative = copy.deepcopy(sample_profile)
    before = copy.deepcopy(sample_profile)
    internal = _score_set(60)
    positive_updates = update_sample_values(positive, render_report, _score_set(90), internal)
    negative_updates = update_sample_values(negative, render_report, _score_set(10), internal)
    if not positive_updates or set(positive_updates) != set(negative_updates):
        return {"passed": False, "reason": "No matching sample updates were produced."}

    key = sorted(positive_updates)[0]
    recording_id, _, object_id = key.partition("::")
    obj = {
        "recording": "private-source",
        "recording_id": recording_id,
        "object_id": object_id,
        "legacy_id": None,
    }
    before_probability = float(generator.selection_probabilities([
        generator.sample_learning_factor(obj, before), 1.0
    ])[0])
    positive_probability = float(generator.selection_probabilities([
        generator.sample_learning_factor(obj, positive), 1.0
    ])[0])
    negative_probability = float(generator.selection_probabilities([
        generator.sample_learning_factor(obj, negative), 1.0
    ])[0])
    passed = (
        all(update["delta"] > 0 for update in positive_updates.values())
        and all(update["delta"] < 0 for update in negative_updates.values())
        and positive_probability > before_probability > negative_probability
    )
    return {
        "passed": passed,
        "samples_checked": len(positive_updates),
        "selection_probability_direction": {
            "negative": round(negative_probability, 8),
            "before": round(before_probability, 8),
            "positive": round(positive_probability, 8),
        },
    }


def _level_isolation_cycle(learning_profile):
    profile = copy.deepcopy(learning_profile) if isinstance(learning_profile, dict) else copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    profile.setdefault("weights", copy.deepcopy(DEFAULT_LEARNING_PROFILE["weights"]))
    shared_before = copy.deepcopy(profile["weights"])
    cases = (
        ("D1: smoother", 1, "transition_smoothness_weight"),
        ("D3: more active", 3, "activity_weight"),
        ("D5: less repetition", 5, "repetition_control"),
    )
    case_results = []
    for comment, level, control in cases:
        levels_before = copy.deepcopy(profile.get("level_weights", {}))
        apply_comment_to_profile(profile, parse_feedback_comment(comment, level))
        target = f"D{level}"
        levels_after = profile["level_weights"]
        target_increased = (
            float(levels_after[target][control])
            > float(levels_before[target][control])
        )
        other_levels_unchanged = all(
            levels_after[name] == levels_before[name]
            for name in ("D1", "D3", "D5")
            if name != target
        )
        case_results.append({
            "comment": comment,
            "target": target,
            "control": control,
            "target_increased": target_increased,
            "other_levels_unchanged": other_levels_unchanged,
        })

    passed = (
        profile["weights"] == shared_before
        and all(
            result["target_increased"] and result["other_levels_unchanged"]
            for result in case_results
        )
    )
    return {
        "passed": passed,
        "shared_weights_unchanged": profile["weights"] == shared_before,
        "cases": case_results,
    }


def validate_private_project(project_dir: str | Path):
    project = Path(project_dir).resolve()
    checks = {}
    memory = _read_json(project / "memory_index_v3.json", [])
    if not isinstance(memory, list) or not memory:
        checks["memory_index"] = {"status": "BLOCKING", "detail": "memory_index_v3.json is missing or empty."}
        memory = []
    else:
        objects = [obj for recording in memory for obj in recording.get("objects", [])]
        invalid_recording_ids = sum(not str(item.get("recording_id", "")).startswith("rec_") for item in memory)
        invalid_object_ids = sum(
            not str(obj.get("stable_id", obj.get("id", ""))).startswith("obj_") for obj in objects
        )
        non_48k = sum(int(item.get("sample_rate", 0)) != TARGET_SR for item in memory)
        missing_sources = sum(
            not (project / "alpha_memory" / str(item.get("recording", ""))).exists() for item in memory
        )
        blockers = invalid_recording_ids + invalid_object_ids + non_48k + missing_sources
        checks["memory_index"] = {
            "status": "PASS" if blockers == 0 else "BLOCKING",
            "recordings": len(memory),
            "objects": len(objects),
            "invalid_recording_ids": invalid_recording_ids,
            "invalid_object_ids": invalid_object_ids,
            "non_48000_hz_records": non_48k,
            "missing_source_files": missing_sources,
        }

    reports = _latest_reports_by_level(project)
    missing_levels = [level for level in ("D1", "D3", "D5") if level not in reports]
    invalid_report_rates = sum(
        int(report.get("target_sample_rate", 0)) != TARGET_SR for _, report in reports.values()
    )
    checks["render_reports"] = {
        "status": "PASS" if not missing_levels and invalid_report_rates == 0 else "BLOCKING",
        "levels_found": sorted(reports),
        "missing_levels": missing_levels,
        "non_48000_hz_reports": invalid_report_rates,
    }

    sample_profile = _read_json(project / "sample_learning_profile.json", {
        "version": 2,
        "samples": {},
        "total_render_selections": 0,
    })
    source_report = next((reports[level][1] for level in ("D5", "D3", "D1") if level in reports), None)
    if source_report:
        cycle = _controlled_real_report_cycle(sample_profile, source_report)
        checks["controlled_feedback_cycle"] = {
            "status": "PASS" if cycle.pop("passed") else "BLOCKING",
            **cycle,
        }
    else:
        checks["controlled_feedback_cycle"] = {
            "status": "BLOCKING",
            "detail": "No real render report was available for controlled credit testing.",
        }

    learning_profile = _read_json(project / "learning_profile.json", DEFAULT_LEARNING_PROFILE)
    isolation = _level_isolation_cycle(learning_profile)
    checks["level_specific_text"] = {
        "status": "PASS" if isolation.pop("passed") else "BLOCKING",
        **isolation,
    }

    event_ids = []
    invalid_feedback_files = 0
    for path_text in glob.glob(str(project / "human_feedback" / "*.json")):
        entry = _read_json(Path(path_text))
        if not isinstance(entry, dict) or not entry.get("event_id"):
            invalid_feedback_files += 1
            continue
        event_ids.append(entry["event_id"])
    duplicate_events = len(event_ids) - len(set(event_ids))
    checks["feedback_history"] = {
        "status": "PASS" if invalid_feedback_files == 0 and duplicate_events == 0 else "BLOCKING",
        "events": len(event_ids),
        "invalid_event_files": invalid_feedback_files,
        "duplicate_event_ids": duplicate_events,
    }

    blockers = [name for name, result in checks.items() if result["status"] == "BLOCKING"]
    return {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "scope": "private real-data baseline only",
        "project_directory": str(project),
        "status": "PASS" if not blockers else "BLOCKED",
        "checks": checks,
        "blocking_checks": blockers,
        "profiles_modified": False,
        "audio_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--output", default="reports/gate1_private_baseline.json")
    args = parser.parse_args()
    report = validate_private_project(args.project_dir)
    output = Path(args.project_dir) / args.output
    atomic_write_json(output, report)
    print(f"Private baseline: {report['status']}")
    print(f"Saved: {output}")
    if report["blocking_checks"]:
        print("Blocking checks:", ", ".join(report["blocking_checks"]))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

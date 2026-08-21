"""Reanalyse existing rated audio with Critic v2.1 without changing learning."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import critic_v2 as critic
from critic_calibration import build_calibration
from hyponoia_stability import atomic_write_json, utc_timestamp


def reanalyse(project_dir="."):
    project = Path(project_dir).resolve()
    original_cwd = Path.cwd()
    reanalysis_dir = project / "reports" / "critic_v2_1_reanalysis"
    paired_dir = project / "reports" / "critic_v2_1_paired"
    reanalysis_dir.mkdir(parents=True, exist_ok=True)
    paired_dir.mkdir(parents=True, exist_ok=True)

    analysed = []
    skipped = []
    try:
        os.chdir(project)
        for feedback_path_text in sorted(glob.glob("human_feedback/*.json")):
            feedback_path = Path(feedback_path_text)
            try:
                feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
                audio_value = feedback.get("audio_file")
                if not audio_value:
                    skipped.append({"feedback": str(feedback_path), "reason": "missing audio_file"})
                    continue
                audio_path = Path(audio_value)
                if not audio_path.is_absolute():
                    audio_path = project / audio_path
                if not audio_path.exists():
                    skipped.append({"feedback": str(feedback_path), "reason": "audio file not present"})
                    continue

                relative_audio = os.path.relpath(audio_path, project)
                report = critic.analyse_audio(relative_audio)
                base = audio_path.stem
                report_path = reanalysis_dir / f"{base}_critic_v2_1.json"
                atomic_write_json(report_path, report)

                paired = {
                    "schema_version": 1,
                    "source_feedback": str(feedback_path),
                    "audio_file": relative_audio,
                    "critic_version": critic.CRITIC_VERSION,
                    "internal_scores": report["internal_scores"],
                    "human_scores": feedback.get("human_scores", {}),
                }
                paired_path = paired_dir / f"{base}_paired.json"
                atomic_write_json(paired_path, paired)
                analysed.append({
                    "audio_file": relative_audio,
                    "feedback": str(feedback_path),
                    "critic_report": str(report_path),
                    "paired_file": str(paired_path),
                })
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                skipped.append({
                    "feedback": str(feedback_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
    finally:
        os.chdir(original_cwd)

    calibration = build_calibration(str(paired_dir))
    calibration["critic_version"] = critic.CRITIC_VERSION
    calibration["source"] = "read-only reanalysis of existing rated audio"
    calibration_path = project / "reports" / "critic_calibration_v2_1_existing_pairs.json"
    atomic_write_json(calibration_path, calibration)

    summary = {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "critic_version": critic.CRITIC_VERSION,
        "audio_or_learning_modified": False,
        "analysed_pairs": len(analysed),
        "analysed": analysed,
        "skipped": skipped,
        "calibration_report": str(calibration_path),
    }
    summary_path = project / "reports" / "critic_recalibration_v2_1_summary.json"
    atomic_write_json(summary_path, summary)
    return summary, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()
    summary, calibration = reanalyse(args.project_dir)
    print("Critic version:", summary["critic_version"])
    print("Existing rated audio reanalysed:", summary["analysed_pairs"])
    print("Missing/unavailable pairs skipped:", len(summary["skipped"]))
    print("Saved:", summary["calibration_report"])
    coherence = calibration["metrics"]["coherence"]
    print("Coherence n:", coherence["n"])
    print("Coherence variance:", coherence["critic_variance"])
    print("Coherence MAE:", coherence["mae"])
    print("Coherence rank correlation:", coherence["rank_correlation"])


if __name__ == "__main__":
    main()

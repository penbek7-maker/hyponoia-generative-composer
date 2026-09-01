"""Safely update Hyponoia's active WAV library and musical memory.

The updater never moves, deletes or rewrites source audio.  Unchanged and
renamed recordings reuse their analysed memory entries; only new or modified
recordings are decoded and analysed.  Existing generated data is backed up
before the new index and manifest are promoted atomically.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

from hyponoia_stability import (
    TARGET_SR,
    atomic_write_json,
    stable_object_id,
    stable_recording_id,
    utc_timestamp,
)
from library_manager_v1 import load_library_manifest, scan_library, update_library_manifest
from memory_builder_v3 import MIN_OBJECT, analyse_object, detect_sound_objects, load_audio


def _safe_timestamp() -> str:
    return utc_timestamp().replace(":", "-")


def _load_previous_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid memory index: {path}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Invalid memory index: {path}")
    return [item for item in data if isinstance(item, dict)]


def _analyse_recording(root: Path, relative_path: str) -> dict[str, Any]:
    audio_path = root / relative_path
    audio, sr = load_audio(audio_path)
    recording_id = stable_recording_id(audio, sr)
    recording: dict[str, Any] = {
        "schema_version": 2,
        "recording": relative_path,
        "recording_id": recording_id,
        "sample_rate": int(sr),
        "duration": len(audio) / sr,
        "objects": [],
    }
    for legacy_id, (start, end) in enumerate(detect_sound_objects(audio, sr)):
        fragment = audio[int(start * sr) : int(end * sr)]
        if len(fragment) < sr * MIN_OBJECT:
            continue
        object_id = stable_object_id(fragment, sr)
        recording["objects"].append(
            {
                "id": object_id,
                "stable_id": object_id,
                "legacy_id": legacy_id,
                "start": float(start),
                "end": float(end),
                "start_sample": int(round(start * sr)),
                "end_sample": int(round(end * sr)),
                "duration": float(end - start),
                "features": analyse_object(fragment, sr),
                "times_used": 0,
            }
        )
    return recording


def build_incremental_memory(
    library_dir: str | Path,
    previous_index_path: str | Path,
    previous_manifest: dict[str, Any],
    current_files: dict[str, dict[str, Any]],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a complete index while analysing only genuinely changed audio."""
    root = Path(library_dir).expanduser().resolve()
    previous_records = {
        str(record.get("recording")): record
        for record in _load_previous_index(Path(previous_index_path))
        if record.get("recording")
    }
    previous_files = dict(previous_manifest.get("active_files", {}))
    rename_sources = {item["to"]: item["from"] for item in plan.get("renamed", [])}
    modified_or_added = set(plan.get("modified", [])) | set(plan.get("added", []))

    memory: list[dict[str, Any]] = []
    reused: list[str] = []
    analysed: list[str] = []
    failures: list[dict[str, str]] = []

    for relative_path in sorted(current_files, key=str.casefold):
        source_path = rename_sources.get(relative_path, relative_path)
        old_record = previous_records.get(source_path)
        old_digest = previous_files.get(source_path, {}).get("content_sha256")
        current_digest = current_files[relative_path].get("content_sha256")
        can_reuse = (
            relative_path not in modified_or_added
            and old_record is not None
            and old_digest == current_digest
        ) or (relative_path in rename_sources and old_record is not None and old_digest == current_digest)

        if can_reuse:
            record = copy.deepcopy(old_record)
            record["recording"] = relative_path
            memory.append(record)
            reused.append(relative_path)
            continue

        try:
            memory.append(_analyse_recording(root, relative_path))
            analysed.append(relative_path)
        except Exception as exc:
            failures.append(
                {"recording": relative_path, "error": f"{type(exc).__name__}: {exc}"}
            )

    report = {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "target_sample_rate": TARGET_SR,
        "library_path": str(root),
        "recordings": len(memory),
        "sound_objects": sum(len(item.get("objects", [])) for item in memory),
        "reused_recordings": reused,
        "analysed_recordings": analysed,
        "failed_recordings": failures,
        "source_audio_modified": False,
        "stable_id_scheme": "sha256(canonical mono float32 audio at 48 kHz)",
    }
    return memory, report


def _backup_generated_files(paths: list[Path], backup_root: Path) -> list[str]:
    copied: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        backup_root.mkdir(parents=True, exist_ok=True)
        destination = backup_root / path.name
        shutil.copy2(path, destination)
        copied.append(str(destination))
    return copied


def update_library(
    library_dir: str | Path,
    project_dir: str | Path,
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Preview or safely apply a complete, reversible library update."""
    library_root = Path(library_dir).expanduser().resolve()
    project_root = Path(project_dir).expanduser().resolve()
    manifest_path = project_root / "library_manifest_v1.json"
    index_path = project_root / "memory_index_v3.json"
    report_path = project_root / "memory_build_report.json"
    user_config_path = project_root / "hyponoia_user_config.json"

    previous_manifest = load_library_manifest(manifest_path)
    current_files = scan_library(library_root)
    proposed_manifest, plan = update_library_manifest(
        library_root, manifest_path, dry_run=True
    )
    result: dict[str, Any] = {
        "status": "preview" if not apply else "pending",
        "library_path": str(library_root),
        "project_path": str(project_root),
        "plan": plan,
        "recommended_recordings": 100,
        "minimum_is_recommendation_not_a_lock": True,
        "source_audio_modified": False,
    }
    if not current_files:
        result["status"] = "blocked"
        result["error"] = "No WAV files were found; the current working memory was preserved."
        return result
    if not apply:
        return result

    memory, build_report = build_incremental_memory(
        library_root,
        index_path,
        previous_manifest,
        current_files,
        plan,
    )
    if build_report["failed_recordings"]:
        result["status"] = "blocked"
        result["error"] = "At least one WAV could not be analysed; the current working memory was preserved."
        result["build_report"] = build_report
        return result
    if not memory or build_report["sound_objects"] < 1:
        result["status"] = "blocked"
        result["error"] = "The selected WAVs produced no usable sound objects; the current working memory was preserved."
        result["build_report"] = build_report
        return result

    backup_root = project_root / "library_backups" / _safe_timestamp()
    backups = _backup_generated_files(
        [manifest_path, index_path, report_path, user_config_path], backup_root
    )
    atomic_write_json(index_path, memory)
    atomic_write_json(report_path, build_report)
    atomic_write_json(manifest_path, proposed_manifest)
    atomic_write_json(
        user_config_path,
        {
            "schema_version": 1,
            "updated_at": utc_timestamp(),
            "memory_folder": str(library_root),
            "memory_file": str(index_path),
            "representation_config": str(project_root / "representation_config.json"),
        },
    )
    result.update(
        {
            "status": "updated",
            "memory_index": str(index_path),
            "manifest": str(manifest_path),
            "user_config": str(user_config_path),
            "backup_folder": str(backup_root) if backups else None,
            "backed_up_files": backups,
            "build_report": build_report,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    result = update_library(args.library, args.project_dir, apply=not args.preview)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

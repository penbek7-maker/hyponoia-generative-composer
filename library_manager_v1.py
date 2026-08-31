"""Content-aware, reversible inventory for user-managed Hyponoia WAVs.

This module records what changed in a library. It deliberately does not train
or render: the UI can show the proposed update before starting expensive work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from hyponoia_stability import atomic_write_json, utc_timestamp


LIBRARY_MANIFEST_VERSION = 1
SUPPORTED_SUFFIXES = frozenset({".wav"})


def _file_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_library(library_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Return a deterministic inventory without decoding or changing audio."""
    root = Path(library_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Library folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Library path is not a folder: {root}")

    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        inventory[relative] = {
            "relative_path": relative,
            "content_sha256": _file_digest(path),
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }
    return inventory


def load_library_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {
            "schema_version": LIBRARY_MANIFEST_VERSION,
            "active_files": {},
            "archived_files": {},
            "history": [],
        }
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid library manifest: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid library manifest: {manifest_path}")
    data.setdefault("active_files", {})
    data.setdefault("archived_files", {})
    data.setdefault("history", [])
    return data


def plan_library_update(
    previous_files: dict[str, dict[str, Any]],
    current_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Classify additions, replacements, removals, renames and unchanged WAVs."""
    previous_paths = set(previous_files)
    current_paths = set(current_files)
    shared_paths = previous_paths & current_paths
    unchanged = sorted(
        path
        for path in shared_paths
        if previous_files[path].get("content_sha256") == current_files[path].get("content_sha256")
    )
    modified = sorted(shared_paths - set(unchanged))
    removed_candidates = set(previous_paths - current_paths)
    added_candidates = set(current_paths - previous_paths)

    old_by_digest: dict[str, list[str]] = {}
    new_by_digest: dict[str, list[str]] = {}
    for path in removed_candidates:
        old_by_digest.setdefault(str(previous_files[path].get("content_sha256")), []).append(path)
    for path in added_candidates:
        new_by_digest.setdefault(str(current_files[path].get("content_sha256")), []).append(path)

    renamed = []
    for digest in sorted(set(old_by_digest) & set(new_by_digest)):
        old_paths = sorted(old_by_digest[digest])
        new_paths = sorted(new_by_digest[digest])
        if len(old_paths) == len(new_paths) == 1:
            old_path, new_path = old_paths[0], new_paths[0]
            renamed.append({"from": old_path, "to": new_path, "content_sha256": digest})
            removed_candidates.remove(old_path)
            added_candidates.remove(new_path)

    result = {
        "added": sorted(added_candidates),
        "modified": modified,
        "removed": sorted(removed_candidates),
        "renamed": renamed,
        "unchanged": unchanged,
    }
    result["changed"] = any(result[key] for key in ("added", "modified", "removed", "renamed"))
    result["summary"] = {
        "active_wavs": len(current_files),
        "added": len(result["added"]),
        "modified": len(result["modified"]),
        "removed": len(result["removed"]),
        "renamed": len(result["renamed"]),
        "unchanged": len(result["unchanged"]),
    }
    return result


def update_library_manifest(
    library_dir: str | Path,
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scan, plan and atomically save a reversible active-library manifest."""
    root = Path(library_dir).expanduser().resolve()
    previous = load_library_manifest(manifest_path)
    previous_files = dict(previous.get("active_files", {}))
    current_files = scan_library(root)
    plan = plan_library_update(previous_files, current_files)
    timestamp = utc_timestamp()

    rename_sources = {item["to"]: item["from"] for item in plan["renamed"]}
    active_files: dict[str, dict[str, Any]] = {}
    for path, current in current_files.items():
        previous_path = path if path in previous_files else rename_sources.get(path)
        previous_entry = previous_files.get(previous_path, {}) if previous_path else {}
        active_files[path] = {
            **current,
            "first_seen": previous_entry.get("first_seen", timestamp),
            "last_seen": timestamp,
            "previous_relative_path": previous_path if previous_path and previous_path != path else None,
        }

    archived_files = dict(previous.get("archived_files", {}))
    for path in plan["removed"] + plan["modified"]:
        old = dict(previous_files[path])
        digest = str(old.get("content_sha256", "unknown"))
        archive_key = f"{path}::{digest[:16]}"
        archived_files[archive_key] = {
            **old,
            "archived_at": timestamp,
            "archive_reason": "replaced" if path in plan["modified"] else "removed",
        }

    history = list(previous.get("history", []))
    if plan["changed"] or not history:
        history.append({"timestamp": timestamp, **plan["summary"]})

    manifest = {
        "schema_version": LIBRARY_MANIFEST_VERSION,
        "library_path": str(root),
        "updated_at": timestamp,
        "recommended_recordings": 100,
        "active_files": active_files,
        "archived_files": archived_files,
        "history": history,
        "last_update": plan,
    }
    if not dry_run:
        atomic_write_json(manifest_path, manifest)
    return manifest, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("alpha_memory"))
    parser.add_argument("--manifest", type=Path, default=Path("library_manifest_v1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _manifest, plan = update_library_manifest(args.library, args.manifest, dry_run=args.dry_run)
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

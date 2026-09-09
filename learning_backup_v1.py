"""Export and recoverably reset Hyponoia's private learning state."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from hyponoia_runtime import update_user_config
from hyponoia_stability import utc_timestamp


PRIVATE_FILES = (
    "learning_profile.json",
    "sample_learning_profile.json",
    "composition_preference_v1.json",
    "hyponoia_user_config.json",
)


def _stamp() -> str:
    return utc_timestamp().replace(":", "-")


def export_learning_backup(project_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    target = Path(destination).expanduser()
    if target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    included = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in PRIVATE_FILES:
            source = root / name
            if source.is_file():
                archive.write(source, f"learning/{name}")
                included.append(name)
        evidence = root / "human_feedback"
        if evidence.is_dir():
            for source in sorted(evidence.glob("*.json")):
                archive.write(source, f"learning/human_feedback/{source.name}")
                included.append(f"human_feedback/{source.name}")
        manifest = {
            "schema_version": "hyponoia_learning_backup_v1",
            "created_at": utc_timestamp(),
            "included": included,
            "excludes": ["source WAV library", "generated audio", "local voice model"],
        }
        archive.writestr("learning/backup_manifest.json", json.dumps(manifest, indent=2))
    return {"path": str(target.resolve()), "included": included}


def archive_and_reset_learning(project_dir: str | Path) -> dict[str, Any]:
    """Move private learning to a timestamped archive; never delete it."""
    root = Path(project_dir).expanduser().resolve()
    archive_root = root / "learning_backups" / _stamp()
    moved = []
    for name in PRIVATE_FILES[:-1]:
        source = root / name
        if source.exists():
            archive_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive_root / name))
            moved.append(name)
    evidence = root / "human_feedback"
    if evidence.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(evidence), str(archive_root / "human_feedback"))
        moved.append("human_feedback")

    update_user_config(root, composition_preference=None)
    return {"archive": str(archive_root), "moved": moved, "library_preserved": True}

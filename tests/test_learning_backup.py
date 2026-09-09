import json
import zipfile
from pathlib import Path

from learning_backup_v1 import archive_and_reset_learning, export_learning_backup


def test_export_contains_private_learning_but_not_audio(tmp_path):
    (tmp_path / "learning_profile.json").write_text(json.dumps({"history": []}))
    (tmp_path / "composition_preference_v1.json").write_text(json.dumps({"model": "user"}))
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "private.wav").write_bytes(b"audio")
    destination = tmp_path / "backup"

    result = export_learning_backup(tmp_path, destination)

    with zipfile.ZipFile(result["path"]) as archive:
        names = set(archive.namelist())
    assert "learning/learning_profile.json" in names
    assert "learning/composition_preference_v1.json" in names
    assert not any(name.endswith(".wav") for name in names)


def test_reset_is_recoverable_and_preserves_library_config(tmp_path):
    (tmp_path / "learning_profile.json").write_text("{}")
    (tmp_path / "composition_preference_v1.json").write_text("{}")
    (tmp_path / "hyponoia_user_config.json").write_text(json.dumps({
        "memory_folder": "/sounds",
        "composition_preference": "composition_preference_v1.json",
    }))

    result = archive_and_reset_learning(tmp_path)

    assert not (tmp_path / "learning_profile.json").exists()
    assert (Path(result["archive"]) / "learning_profile.json").exists()
    config = json.loads((tmp_path / "hyponoia_user_config.json").read_text())
    assert config["memory_folder"] == "/sounds"
    assert "composition_preference" not in config
    assert result["library_preserved"] is True

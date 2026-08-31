from pathlib import Path

from library_manager_v1 import plan_library_update, update_library_manifest


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_manifest_detects_add_modify_remove_and_preserves_archive(tmp_path):
    library = tmp_path / "library"
    manifest_path = tmp_path / "library_manifest_v1.json"
    _write(library / "a.wav", b"first")
    _write(library / "b.wav", b"second")
    first, first_plan = update_library_manifest(library, manifest_path)
    assert first_plan["summary"]["added"] == 2
    assert len(first["active_files"]) == 2

    _write(library / "a.wav", b"replacement")
    (library / "b.wav").unlink()
    _write(library / "nested" / "c.WAV", b"third")
    second, plan = update_library_manifest(library, manifest_path)
    assert plan["modified"] == ["a.wav"]
    assert plan["removed"] == ["b.wav"]
    assert plan["added"] == ["nested/c.WAV"]
    assert plan["summary"]["active_wavs"] == 2
    assert {entry["archive_reason"] for entry in second["archived_files"].values()} == {"removed", "replaced"}


def test_unique_content_move_is_a_rename_and_keeps_first_seen(tmp_path):
    library = tmp_path / "library"
    manifest_path = tmp_path / "manifest.json"
    _write(library / "old.wav", b"same audio bytes")
    first, _ = update_library_manifest(library, manifest_path)
    first_seen = first["active_files"]["old.wav"]["first_seen"]
    (library / "old.wav").rename(library / "new.wav")
    second, plan = update_library_manifest(library, manifest_path)
    assert plan["added"] == []
    assert plan["removed"] == []
    assert plan["renamed"][0]["from"] == "old.wav"
    assert plan["renamed"][0]["to"] == "new.wav"
    assert second["active_files"]["new.wav"]["first_seen"] == first_seen
    assert second["archived_files"] == {}


def test_dry_run_does_not_write_manifest(tmp_path):
    library = tmp_path / "library"
    manifest_path = tmp_path / "manifest.json"
    _write(library / "sound.wav", b"audio")
    _manifest, plan = update_library_manifest(library, manifest_path, dry_run=True)
    assert plan["changed"] is True
    assert not manifest_path.exists()


def test_ambiguous_duplicate_content_is_not_guessed_as_rename():
    previous = {
        "one.wav": {"content_sha256": "duplicate"},
        "two.wav": {"content_sha256": "duplicate"},
    }
    current = {
        "three.wav": {"content_sha256": "duplicate"},
        "four.wav": {"content_sha256": "duplicate"},
    }
    plan = plan_library_update(previous, current)
    assert plan["renamed"] == []
    assert plan["removed"] == ["one.wav", "two.wav"]
    assert plan["added"] == ["four.wav", "three.wav"]

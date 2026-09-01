import json
from pathlib import Path

import update_library_v1 as updater


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _fake_record(_root: Path, relative_path: str) -> dict:
    token = Path(relative_path).stem
    return {
        "schema_version": 2,
        "recording": relative_path,
        "recording_id": f"rec_{token}",
        "sample_rate": 48_000,
        "duration": 1.0,
        "objects": [
            {
                "id": f"obj_{token}",
                "stable_id": f"obj_{token}",
                "start_sample": 0,
                "end_sample": 48_000,
                "features": {},
            }
        ],
    }


def test_update_builds_memory_config_and_never_changes_source_audio(tmp_path, monkeypatch):
    library = tmp_path / "my sounds"
    project = tmp_path / "project"
    project.mkdir()
    _write(library / "tone.wav", b"original source bytes")
    before = (library / "tone.wav").read_bytes()
    monkeypatch.setattr(updater, "_analyse_recording", _fake_record)

    result = updater.update_library(library, project)

    assert result["status"] == "updated"
    assert result["source_audio_modified"] is False
    assert (library / "tone.wav").read_bytes() == before
    memory = json.loads((project / "memory_index_v3.json").read_text())
    assert memory[0]["recording"] == "tone.wav"
    config = json.loads((project / "hyponoia_user_config.json").read_text())
    assert config["memory_folder"] == str(library.resolve())
    assert config["memory_file"] == str((project / "memory_index_v3.json").resolve())


def test_unchanged_and_renamed_wav_reuse_previous_analysis(tmp_path, monkeypatch):
    library = tmp_path / "library"
    project = tmp_path / "project"
    project.mkdir()
    _write(library / "old.wav", b"same source")
    monkeypatch.setattr(updater, "_analyse_recording", _fake_record)
    updater.update_library(library, project)

    def unexpected_analysis(_root, _relative_path):
        raise AssertionError("unchanged content must reuse the previous analysis")

    monkeypatch.setattr(updater, "_analyse_recording", unexpected_analysis)
    unchanged = updater.update_library(library, project)
    assert unchanged["build_report"]["reused_recordings"] == ["old.wav"]

    (library / "old.wav").rename(library / "renamed.wav")
    renamed = updater.update_library(library, project)
    assert renamed["plan"]["summary"]["renamed"] == 1
    assert renamed["build_report"]["reused_recordings"] == ["renamed.wav"]
    memory = json.loads((project / "memory_index_v3.json").read_text())
    assert memory[0]["recording"] == "renamed.wav"


def test_failed_or_empty_update_preserves_working_memory(tmp_path, monkeypatch):
    library = tmp_path / "library"
    project = tmp_path / "project"
    project.mkdir()
    _write(library / "good.wav", b"good")
    monkeypatch.setattr(updater, "_analyse_recording", _fake_record)
    updater.update_library(library, project)
    existing = (project / "memory_index_v3.json").read_bytes()

    (library / "good.wav").unlink()
    empty = updater.update_library(library, project)
    assert empty["status"] == "blocked"
    assert (project / "memory_index_v3.json").read_bytes() == existing

    _write(library / "broken.wav", b"broken")

    def fail(_root, _relative_path):
        raise RuntimeError("decoder rejected file")

    monkeypatch.setattr(updater, "_analyse_recording", fail)
    broken = updater.update_library(library, project)
    assert broken["status"] == "blocked"
    assert "preserved" in broken["error"]
    assert (project / "memory_index_v3.json").read_bytes() == existing


def test_preview_reports_changes_without_writing_generated_files(tmp_path):
    library = tmp_path / "library"
    project = tmp_path / "project"
    project.mkdir()
    _write(library / "new.wav", b"new")

    result = updater.update_library(library, project, apply=False)

    assert result["status"] == "preview"
    assert result["plan"]["summary"]["added"] == 1
    assert not (project / "memory_index_v3.json").exists()
    assert not (project / "library_manifest_v1.json").exists()

import json
import sys

import pytest

from composition_preference_v1 import CompositionPreferenceAssist
from hyponoia_app import ROOT_NOTE_PITCHES, format_runtime_summary
from hyponoia_runtime import (
    PROJECT_DIR,
    generator_command,
    latest_composition_path,
    load_user_config,
    runtime_status,
    update_user_config,
)
from representation_assist_v1 import RepresentationAssist


def test_packaged_deep_learning_assets_are_active():
    representation = RepresentationAssist.from_config(PROJECT_DIR / "representation_config.json")
    preference = CompositionPreferenceAssist.from_file(
        PROJECT_DIR / "phase2_artifacts" / "composition_preference_gold.json"
    )

    assert representation.active is True
    assert representation.snapshot()["embedding_count"] == 2566
    assert preference.active is True
    assert preference.snapshot()["level_specific_heads"] == [1, 3, 5]


def test_runtime_status_requires_memory_but_reports_models(tmp_path):
    (tmp_path / "phase2_artifacts").mkdir()
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    (tmp_path / "phase2_artifacts" / "embeddings.json").write_text(json.dumps(embeddings))
    (tmp_path / "representation_config.json").write_text(json.dumps({
        "mode": "assist",
        "embeddings_path": "phase2_artifacts/embeddings.json",
        "strength": 0.35,
    }))
    (tmp_path / "phase2_artifacts" / "preference.json").write_text(json.dumps({
        "schema_version": "composition_preference_v1",
        "mode": "assist",
        "strength": 0.3,
        "embeddings_path": "embeddings.json",
        "positive_prototype": [1.0, 0.0],
        "negative_prototype": [0.0, 1.0],
        "positive_evidence": [{"dream_level": 1}],
    }))
    (tmp_path / "hyponoia_user_config.json").write_text(json.dumps({
        "composition_preference": "phase2_artifacts/preference.json",
    }))

    status = runtime_status(tmp_path)

    assert status["representation"]["active"] is True
    assert status["composition_preference"]["active"] is True
    assert status["memory_ready"] is False
    assert status["source_audio_ready"] is False
    assert status["ready_to_generate"] is False


def test_generator_command_is_safe_and_uses_current_python(tmp_path):
    command = generator_command(3, root_pitch=2, scale="minor", confidence=0.8, project_dir=tmp_path)
    assert command == [
        sys.executable,
        str(tmp_path / "generator_v3_memory_bloom_smooth.py"),
        "3",
        "2",
        "minor",
        "0.8",
    ]
    with pytest.raises(ValueError):
        generator_command(2, project_dir=tmp_path)


def test_latest_composition_uses_unique_master_and_ignores_legacy_current(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "current.wav").write_bytes(b"legacy")
    first = output / "Hyponoia_D1_20260918_100000.wav"
    latest = output / "Hyponoia_D3_20260918_110000.wav"
    stem = output / "Hyponoia_D3_20260918_110000_HIGH.wav"
    first.write_bytes(b"first")
    latest.write_bytes(b"latest")
    stem.write_bytes(b"stem")

    first.touch()
    latest.touch()
    stem.touch()

    assert latest_composition_path(tmp_path) == latest


def test_user_config_update_keeps_paths_portable(tmp_path):
    model = tmp_path / "composition_preference_v1.json"
    update_user_config(tmp_path, composition_preference=model, ui_language="el")
    assert load_user_config(tmp_path) == {
        "composition_preference": "composition_preference_v1.json",
        "ui_language": "el",
    }
    update_user_config(tmp_path, composition_preference=None)
    assert load_user_config(tmp_path) == {"ui_language": "el"}


def test_final_ui_uses_musical_root_names():
    assert ROOT_NOTE_PITCHES["C"] == 0
    assert ROOT_NOTE_PITCHES["F♯ / G♭"] == 6
    assert ROOT_NOTE_PITCHES["B"] == 11


def test_final_ui_runtime_summary_is_clear_without_private_detail():
    ready = {
        "ready_to_generate": True,
        "recordings": 34,
        "preference_review_count": 0,
    }
    assert format_runtime_summary(ready) == (
        "Ready to compose · 34 recordings · deep listening and learning active"
    )
    assert format_runtime_summary({"source_audio_ready": False}) == (
        "Start here: choose a folder containing your WAV sounds, then update the library."
    )

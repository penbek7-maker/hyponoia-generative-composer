import json

from composition_preference_v1 import (
    CompositionPreferenceAssist,
    train_composition_preference,
)


def _report(path, audio, objects, dream_level=None):
    details = {}
    for index, (object_id, recording) in enumerate(objects):
        details[f"sample-{index}"] = {
            "object_id": object_id,
            "recording": recording,
            "selection_count": index + 1,
            "exposure_sec": 2.0 + index,
        }
    path.write_text(json.dumps({
        "audio_file": audio,
        "dream_level": dream_level,
        "sample_usage_details": details,
        "total_sample_selections": 84,
        "recordings": {recording: 1 for _, recording in objects},
        "role_counts": {"resonance": 36, "texture": 27, "gesture": 14, "noise": 5, "impact": 2},
        "temporal_metrics": {
            "event_rate_per_minute": 28.0,
            "foreground_event_rate_per_minute": 7.0,
            "average_event_duration_sec": 18.4,
            "quick_succession_count": 26,
        },
    }))


def test_whole_render_preference_favours_positive_embedding_region(tmp_path):
    embeddings = {
        "positive-a": [1.0, 0.0],
        "positive-b": [0.9, 0.1],
        "water-a": [0.0, 1.0],
        "water-b": [0.1, 0.9],
    }
    embedding_path = tmp_path / "embeddings.json"
    embedding_path.write_text(json.dumps(embeddings))
    positive_report = tmp_path / "positive.json"
    contrast_report = tmp_path / "contrast.json"
    _report(positive_report, "D1.wav", [("positive-a", "synth-a.wav"), ("positive-b", "synth-b.wav")], dream_level=1)
    _report(contrast_report, "D3.wav", [("water-a", "water-a.wav"), ("water-b", "water-b.wav")], dream_level=3)
    model_path = tmp_path / "preference.json"

    model = train_composition_preference(
        embeddings_path=embedding_path,
        positive_reports=[positive_report],
        negative_reports=[contrast_report],
        output_path=model_path,
        positive_target=0.98,
    )
    assist = CompositionPreferenceAssist.from_file(model_path)
    assert assist.active is True
    assert assist.object_factor("positive-b", "new-synth.wav", 1) < 1.0
    assert assist.object_factor("water-b", "water-b.wav", 1) > 1.0
    assert assist.object_factor("water-b", "water-b.wav", 5) == 1.0
    assert model["positive_target"] == 0.98
    assert model["embeddings_path"] == "embeddings.json"
    assert not model["positive_evidence"][0]["audio_file"].startswith("/")
    assert model["training_diagnostics"]["covered_positive_objects"] == 2
    assert model["training_diagnostics"]["positive_render_count"] == 1
    assert model["training_diagnostics"]["contrast_render_count"] == 1
    assert model["training_diagnostics"]["mean_training_margin_gap"] > 0
    assert assist.target_event_count(1, 180.0) == 84
    assert assist.target_event_count(3, 180.0) == 91
    assert assist.target_event_count(5, 180.0) == 104
    assert assist.target_average_event_duration(1) == 18.4
    assert assist.target_unique_recordings(1) == 2
    assert assist.target_unique_objects(1) == 2
    assert abs(sum(assist.target_role_distribution(1).values()) - 1.0) < 1e-9
    assert assist.role_factor("resonance", 3) > assist.role_factor("noise", 3)


def test_missing_preference_model_fails_safe(tmp_path):
    assist = CompositionPreferenceAssist.from_file(tmp_path / "missing.json")
    assert assist.active is False
    assert assist.object_factor("anything") == 1.0
    assert assist.target_average_event_duration(1) is None
    assert assist.target_unique_recordings(1) is None
    assert assist.target_unique_objects(1) is None
    assert assist.target_role_distribution(1) == {}


def test_level_specific_heads_preserve_each_accepted_form(tmp_path):
    embeddings = {
        "d1-good": [1.0, 0.0, 0.0],
        "d1-bad": [0.0, 1.0, 0.0],
        "d3-good": [0.0, 0.0, 1.0],
        "d3-bad": [0.0, 0.8, 0.2],
    }
    embedding_path = tmp_path / "embeddings.json"
    embedding_path.write_text(json.dumps(embeddings))
    reports = {}
    for name, object_id, level in (
        ("d1-positive", "d1-good", 1),
        ("d1-negative", "d1-bad", 1),
        ("d3-positive", "d3-good", 3),
        ("d3-negative", "d3-bad", 3),
    ):
        reports[name] = tmp_path / f"{name}.json"
        _report(reports[name], f"{name}.wav", [(object_id, f"{object_id}.wav")], dream_level=level)
    model_path = tmp_path / "preference.json"

    model = train_composition_preference(
        embeddings_path=embedding_path,
        positive_reports=[reports["d1-positive"], reports["d3-positive"]],
        negative_reports=[reports["d1-negative"], reports["d3-negative"]],
        output_path=model_path,
    )
    assist = CompositionPreferenceAssist.from_file(model_path)

    assert sorted(model["level_heads"]) == ["1", "3"]
    assert assist.snapshot()["level_specific_heads"] == [1, 3]
    assert assist.object_factor("d1-good", dream_level=1) < assist.object_factor("d1-bad", dream_level=1)
    assert assist.object_factor("d3-good", dream_level=3) < assist.object_factor("d3-bad", dream_level=3)
    assert assist.object_factor("d3-good", "d3-good.wav", 3) < 1.0
    assert assist.target_event_count(1, 180.0) == 84
    assert assist.target_event_count(3, 180.0) == 84

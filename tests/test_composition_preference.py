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
    assert model["training_diagnostics"]["covered_positive_objects"] == 2
    assert assist.target_event_count(1, 180.0) == 84
    assert assist.target_event_count(3, 180.0) == 91
    assert assist.target_event_count(5, 180.0) == 104
    assert assist.role_factor("resonance", 3) > assist.role_factor("noise", 3)


def test_missing_preference_model_fails_safe(tmp_path):
    assist = CompositionPreferenceAssist.from_file(tmp_path / "missing.json")
    assert assist.active is False
    assert assist.object_factor("anything") == 1.0

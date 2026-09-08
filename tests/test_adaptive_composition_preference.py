import json

import numpy as np

from adaptive_composition_preference_v1 import (
    learning_direction,
    preference_score,
    update_preference_from_review,
)


RATINGS = {
    "musicality": 4.0,
    "material_coherence": 4.0,
    "transition_smoothness": 3.5,
    "variety_without_disconnection": 4.0,
    "synth_material_presence": 4.0,
    "development_over_repetition": 3.5,
    "overall_artistic_impression": 4.5,
}


def _model(tmp_path):
    embeddings = {"a": [1.0, 0.0], "b": [0.8, 0.2], "c": [0.0, 1.0]}
    (tmp_path / "embeddings.json").write_text(json.dumps(embeddings))
    payload = {
        "schema_version": "composition_preference_v1",
        "mode": "assist",
        "strength": 0.3,
        "embeddings_path": "embeddings.json",
        "positive_prototype": [0.0, 1.0],
        "negative_prototype": [1.0, 0.0],
        "positive_structure": {"event_rate_per_minute": 10.0, "role_distribution": {"texture": 1.0}},
        "level_heads": {
            "5": {
                "positive_prototype": [0.0, 1.0],
                "negative_prototype": [1.0, 0.0],
                "positive_structure": {"event_rate_per_minute": 10.0, "role_distribution": {"texture": 1.0}},
                "contrast_structure": {},
                "reference_recordings": [],
            }
        },
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(payload))
    return path


def _report(tmp_path):
    payload = {
        "audio_file": "/private/example.wav",
        "dream_level": 5,
        "total_sample_selections": 3,
        "sample_usage_details": {
            "one": {"object_id": "a", "selection_count": 2, "exposure_sec": 4, "recording": "r1"},
            "two": {"object_id": "b", "selection_count": 1, "exposure_sec": 2, "recording": "r2"},
        },
        "role_counts": {"gesture": 2, "texture": 1},
        "temporal_metrics": {"event_rate_per_minute": 20.0},
        "recordings": {"r1": 2, "r2": 1},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload))
    return path


def test_score_and_explicit_direction_are_transparent():
    assert 1.0 <= preference_score(RATINGS) <= 5.0
    assert learning_direction({"baseline_decision": "yes"}) == "positive"
    assert learning_direction({"baseline_decision": "no"}) == "negative"
    assert learning_direction({"baseline_decision": "unsure"}) == "neutral"


def test_positive_review_updates_private_copy_without_raw_comment(tmp_path):
    gold = _model(tmp_path)
    user = tmp_path / "user.json"
    result = update_preference_from_review(
        base_model_path=gold,
        user_model_path=user,
        render_report_path=_report(tmp_path),
        event={
            "event_id": "cf_test",
            "timestamp": "now",
            "dream_level": "D5",
            "baseline_decision": "yes",
            "ratings_1_to_5": RATINGS,
            "listener_text": {"comment": "private words"},
        },
    )
    assert result["updated"] is True
    updated = json.loads(user.read_text())
    assert np.asarray(updated["positive_prototype"])[0] > 0
    assert updated["online_learning"]["positive_count"] == 1
    assert "private words" not in user.read_text()
    assert json.loads(gold.read_text())["positive_prototype"] == [0.0, 1.0]


def test_unsure_review_does_not_create_or_move_user_head(tmp_path):
    gold = _model(tmp_path)
    user = tmp_path / "user.json"
    result = update_preference_from_review(
        base_model_path=gold,
        user_model_path=user,
        render_report_path=_report(tmp_path),
        event={"dream_level": "D5", "baseline_decision": "unsure", "ratings_1_to_5": RATINGS},
    )
    assert result["updated"] is False
    assert not user.exists()

import json

import learning_profile_store_v1 as store
from learning_profile_store_v1 import load_active_learning_profile, load_learning_seed


def test_clean_install_starts_from_trained_comment_free_seed():
    profile = load_learning_seed()
    assert profile["baseline_id"] == "hyponoia-gold-2026-09-07-v1"
    assert profile["training_summary"]["supervised_feedback_events"] == 16
    assert profile["history"] == []
    assert profile["level_weights"]["D5"]["material_development_weight"] > 1.0
    assert profile["level_weights"]["D1"]["structured_granulation_weight"] > 1.0


def test_legacy_user_comments_are_preserved_when_baseline_is_migrated(tmp_path, monkeypatch):
    event = {"event_id": "kept-comment", "transcript": "περισσότερη εξέλιξη"}
    legacy = {
        "version": 2,
        "weights": {},
        "level_weights": {},
        "history": [event],
    }
    path = tmp_path / "learning_profile.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(store, "PROJECT_DIR", tmp_path)
    migrated = load_active_learning_profile(path)
    assert migrated["history"] == [event]
    assert migrated["baseline_id"] == "hyponoia-gold-2026-09-07-v1"
    assert migrated["migration"]["preserved_history_events"] == 1


def test_existing_release_profile_keeps_new_user_learning(tmp_path):
    profile = load_learning_seed()
    profile["level_weights"]["D3"]["activity_weight"] += 0.05
    profile["history"].append({"event_id": "new-feedback"})
    path = tmp_path / "learning_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = load_active_learning_profile(path)
    assert loaded["history"][-1]["event_id"] == "new-feedback"
    assert loaded["level_weights"]["D3"]["activity_weight"] == profile["level_weights"]["D3"]["activity_weight"]

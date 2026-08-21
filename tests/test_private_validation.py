import json

from gate1_private_validation import validate_private_project
from human_feedback_v1 import DEFAULT_LEARNING_PROFILE


def test_private_validator_passes_complete_fixture_without_mutating_profiles(tmp_path):
    memory_dir = tmp_path / "alpha_memory"
    reports_dir = tmp_path / "render_reports"
    feedback_dir = tmp_path / "human_feedback"
    memory_dir.mkdir()
    reports_dir.mkdir()
    feedback_dir.mkdir()
    (memory_dir / "source.wav").touch()

    memory = [{
        "recording": "source.wav",
        "recording_id": "rec_1234567890abcdef1234",
        "sample_rate": 48_000,
        "objects": [{"stable_id": "obj_1234567890abcdef1234"}],
    }]
    (tmp_path / "memory_index_v3.json").write_text(json.dumps(memory))
    sample_key = "rec_1234567890abcdef1234::obj_1234567890abcdef1234"
    for level in (1, 3, 5):
        report = {
            "dream_level": level,
            "target_sample_rate": 48_000,
            "samples": {sample_key: 2},
            "sample_usage_details": {
                sample_key: {
                    "selection_count": 2,
                    "exposure_sec": 4.0,
                    "gain_sum": 1.0,
                    "role_counts": {"texture": 2},
                }
            },
        }
        (reports_dir / f"D{level}_render_report.json").write_text(json.dumps(report))

    sample_profile = {"version": 2, "samples": {}, "total_render_selections": 6}
    learned_profile = json.loads(json.dumps(DEFAULT_LEARNING_PROFILE))
    learned_profile["level_weights"]["D1"]["transition_smoothness_weight"] = 1.18
    learned_profile["level_weights"]["D3"]["activity_weight"] = 1.24
    learned_profile["level_weights"]["D5"]["musicality_weight"] = 1.12
    learned_profile["level_weights"]["D5"]["repetition_control"] = 1.15
    learning_text = json.dumps(learned_profile)
    sample_text = json.dumps(sample_profile)
    (tmp_path / "learning_profile.json").write_text(learning_text)
    (tmp_path / "sample_learning_profile.json").write_text(sample_text)

    result = validate_private_project(tmp_path)
    assert result["status"] == "PASS"
    assert result["blocking_checks"] == []
    assert result["profiles_modified"] is False
    assert (tmp_path / "learning_profile.json").read_text() == learning_text
    assert (tmp_path / "sample_learning_profile.json").read_text() == sample_text

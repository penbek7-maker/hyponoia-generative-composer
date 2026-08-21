import json

import critic_recalibration_v21 as recalibration


def test_reanalysis_preserves_feedback_and_builds_paired_calibration(tmp_path, monkeypatch):
    (tmp_path / "human_feedback").mkdir()
    (tmp_path / "output").mkdir()
    audio = tmp_path / "output" / "rated.wav"
    audio.write_bytes(b"test-placeholder")
    feedback = {
        "audio_file": "output/rated.wav",
        "human_scores": {
            "musicality": 70,
            "coherence": 72,
            "richness": 68,
            "transitions": 75,
            "bloom_quality": 71,
            "overall": 70,
        },
    }
    feedback_path = tmp_path / "human_feedback" / "rated_feedback.json"
    feedback_path.write_text(json.dumps(feedback))

    internal = {
        "musicality": 66,
        "coherence": 70,
        "richness": 64,
        "transitions": 74,
        "bloom_quality": 69,
        "overall": 68,
    }
    monkeypatch.setattr(
        recalibration.critic,
        "analyse_audio",
        lambda path: {"critic_version": "2.1", "internal_scores": internal},
    )
    before = feedback_path.read_text()
    summary, calibration = recalibration.reanalyse(tmp_path)
    assert summary["analysed_pairs"] == 1
    assert summary["audio_or_learning_modified"] is False
    assert calibration["metrics"]["coherence"]["n"] == 1
    assert feedback_path.read_text() == before

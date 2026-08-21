import json

from critic_calibration import build_calibration


CRITERIA = ("musicality", "coherence", "richness", "transitions", "bloom_quality", "overall")


def test_calibration_reports_variance_mae_and_rank_correlation(tmp_path):
    for index, (critic_value, human_value) in enumerate(((40, 50), (60, 65), (80, 90))):
        entry = {
            "internal_scores": {criterion: critic_value for criterion in CRITERIA},
            "human_scores": {criterion: human_value for criterion in CRITERIA},
        }
        (tmp_path / f"feedback_{index}.json").write_text(json.dumps(entry))
    report = build_calibration(str(tmp_path))
    overall = report["metrics"]["overall"]
    assert report["feedback_events"] == 3
    assert overall["critic_variance"] > 0
    assert overall["mae"] > 0
    assert overall["rank_correlation"] == 1.0


def test_calibration_uses_only_dimensions_with_matching_human_ratings(tmp_path):
    entry = {
        "internal_scores": {criterion: 60 for criterion in CRITERIA},
        "human_scores": {"musicality": 70, "overall": 75},
    }
    (tmp_path / "partial_feedback.json").write_text(json.dumps(entry))
    report = build_calibration(str(tmp_path))
    assert report["metrics"]["musicality"]["n"] == 1
    assert report["metrics"]["overall"]["n"] == 1
    assert report["metrics"]["coherence"]["n"] == 0
    assert report["dimension_event_counts"]["richness"] == 0

import json

import numpy as np
import pytest

from representation_feedback_v2 import adapt_embeddings_from_feedback, load_feedback, load_feedback_history


def test_feedback_adapter_writes_isolated_normalized_artifacts(tmp_path):
    embeddings = {
        "a": [1.0, 0.0, 0.1, 0.0],
        "b": [0.9, 0.1, 0.0, 0.0],
        "c": [0.0, 1.0, 0.1, 0.0],
        "d": [0.1, 0.9, 0.0, 0.0],
        "e": [0.7, 0.7, 0.0, 0.1],
        "f": [0.6, 0.7, 0.1, 0.0],
    }
    feedback = {
        "version": 1,
        "answers": {
            "a__b": "related",
            "a__c": "different",
            "c__d": "related",
            "c__a": "different",
            "e__f": "related",
            "e__a": "different",
            "b__d": "unsure",
        },
    }
    embeddings_path = tmp_path / "embeddings_v1.json"
    feedback_path = tmp_path / "feedback.json"
    embeddings_path.write_text(json.dumps(embeddings))
    feedback_path.write_text(json.dumps(feedback))
    output = tmp_path / "v2"

    manifest = adapt_embeddings_from_feedback(
        embeddings_path, feedback_path, output, steps=20, learning_rate=0.02
    )

    assert manifest["gate1_integration"] is False
    assert manifest["generator_integration"] is False
    assert manifest["training_pairs"] == 6
    assert manifest["uncertain_pairs_ignored_for_training"] == 1
    assert manifest["leave_one_anchor_out"]["pairs"] == 6
    assert (output / "metric_adapter_v2.pt").exists()
    assert (output / "feedback_training_manifest_v2.json").exists()
    adapted = json.loads((output / "embeddings_v2.json").read_text())
    assert set(adapted) == set(embeddings)
    assert all(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5) for vector in adapted.values())


def test_feedback_rejects_unknown_stable_ids(tmp_path):
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text(json.dumps({"answers": {"known__missing": "related"}}))
    with pytest.raises(ValueError, match="unknown stable IDs"):
        load_feedback(feedback_path, {"known"})


def test_feedback_history_keeps_latest_answer_for_same_pair(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"answers": {"a__b": "unsure", "a__c": "different"}}))
    second.write_text(json.dumps({"answers": {"a__b": "related"}}))
    pairs, paths = load_feedback_history([first, second], {"a", "b", "c"})
    assert paths == [first, second]
    assert {(pair.left, pair.right): pair.label for pair in pairs} == {
        ("a", "b"): "related",
        ("a", "c"): "different",
    }

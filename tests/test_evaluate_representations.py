import json

from evaluate_representations_v1 import evaluate_embeddings, select_balanced_review_anchors


def test_evaluation_reports_neighbor_structure(tmp_path):
    records = []
    embeddings = {}
    for index in range(6):
        stable_id = f"obj_{index}"
        group = index // 3
        item = {
            "stable_id": stable_id,
            "duration": 1.0 + group,
            "features": {
                "energy": 0.1 + group,
                "register": "low" if group == 0 else "high",
                "gesture_type": "texture" if group == 0 else "burst",
                "phrase_role": "opening" if group == 0 else "ending",
                "emotional_function": "memory" if group == 0 else "rupture",
                "synthetic_score": index / 6,
                "drone_score": (index + 1) / 7,
                "harmonicity": 0.2,
                "pitch_confidence": 0.1,
                "transient_score": (6 - index) / 6,
                "gesture_strength": 0.3,
                "texture_score": (index % 2) * 0.8,
                "ambient_score": 0.4,
            },
        }
        records.append({"recording": f"source_{index}.wav", "objects": [item]})
        embeddings[stable_id] = [1.0, index * 0.01] if group == 0 else [index * 0.01, 1.0]
    index_path = tmp_path / "memory_index_v3.json"
    index_path.write_text(json.dumps(records))
    embeddings_path = tmp_path / "embeddings.json"
    embeddings_path.write_text(json.dumps(embeddings))
    output = tmp_path / "evaluation.json"
    result = evaluate_embeddings(index_path, embeddings_path, output, neighbors=2, sampled_pairs=1000, seed=3)
    assert result["neighbor_policy"] == "cross_recording_only"
    assert result["matched_embeddings"] == 6
    assert result["neighbor_distance_ratio"] < 1.0
    assert result["categorical_neighbor_agreement"]["register"]["lift"] > 1.0
    for example in result["example_neighbors"]:
        assert all(neighbor["recording"] != example["anchor"]["recording"] for neighbor in example["neighbors"])
    assert result["review_sampling_policy"] == "balanced_four_sound_families"
    assert {example["anchor"]["review_category_key"] for example in result["example_neighbors"]} == {
        "synthetic", "tonal_drone", "transient_gesture", "texture_ambient"
    }
    assert output.exists()


def test_balanced_review_selection_round_robins_categories_and_recordings():
    ids = [f"obj_{index}" for index in range(12)]
    metadata = {
        stable_id: {
            "recording": f"sample_{index}.wav",
            "features": {
                "synthetic_score": index / 11,
                "drone_score": (11 - index) / 11,
                "harmonicity": 0.1,
                "pitch_confidence": 0.1,
                "transient_score": (index % 4) / 3,
                "gesture_strength": 0.1,
                "texture_score": ((index + 2) % 4) / 3,
                "ambient_score": 0.1,
            },
        }
        for index, stable_id in enumerate(ids)
    }
    selected = select_balanced_review_anchors(ids, metadata, limit=8)
    assert [row[1] for row in selected[:4]] == [
        "synthetic", "tonal_drone", "transient_gesture", "texture_ambient"
    ]
    assert len({metadata[ids[row[0]]]["recording"] for row in selected}) == 8

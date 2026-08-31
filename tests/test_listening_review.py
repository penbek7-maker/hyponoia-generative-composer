import json

import numpy as np
import soundfile as sf

from build_listening_review_v1 import build_listening_review


def test_builds_local_review_with_audio_and_downloadable_feedback(tmp_path):
    memory = tmp_path / "alpha_memory"
    memory.mkdir()
    sf.write(memory / "source.wav", np.ones(4_800, dtype=np.float32) * 0.1, 48_000)
    objects = [
        {"stable_id": "obj_a", "start_sample": 0, "end_sample": 2_400},
        {"stable_id": "obj_b", "start_sample": 2_400, "end_sample": 4_800},
    ]
    index_path = tmp_path / "memory_index_v3.json"
    index_path.write_text(json.dumps([{"recording": "source.wav", "objects": objects}]))
    evaluation = {
        "example_neighbors": [{
            "anchor": {
                "stable_id": "obj_a",
                "recording": "source.wav",
                "review_category": "Συνθετικός / synth χαρακτήρας",
            },
            "neighbors": [{"stable_id": "obj_b", "recording": "source.wav", "cosine_similarity": 0.8}],
        }]
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation))
    output = tmp_path / "review"
    manifest = build_listening_review(index_path, memory, evaluation_path, output, trials=1)
    html = (output / "index.html").read_text()
    assert manifest == {
        "title": "Hyponoia Listening Review v1",
        "review_id": "hyponoia-listening-review-v1",
        "trials": 1,
        "comparisons": 1,
        "unique_clips": 2,
        "sample_rate": 48_000,
        "entrypoint": "index.html",
    }
    assert "hyponoia_listening_feedback.json" in html
    assert "Συνθετικός / synth χαρακτήρας" in html
    assert (output / "clips" / "obj_a.wav").exists()
    assert (output / "clips" / "obj_b.wav").exists()

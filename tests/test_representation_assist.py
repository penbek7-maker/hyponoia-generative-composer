import json

import pytest
import numpy as np

from representation_assist_v1 import RepresentationAssist
import generator_v3_memory_bloom_smooth as generator


def test_off_mode_is_exactly_neutral_even_when_path_is_present(tmp_path):
    config = tmp_path / "representation_config.json"
    config.write_text(json.dumps({
        "mode": "off",
        "embeddings_path": "missing.json",
        "strength": 0.35,
    }))
    assist = RepresentationAssist.from_config(config)
    assert assist.active is False
    assert assist.continuity_factor("a", "b") == 1.0


def test_assist_prefers_related_but_remains_bounded(tmp_path):
    embeddings = tmp_path / "embeddings.json"
    embeddings.write_text(json.dumps({
        "anchor": [1.0, 0.0],
        "related": [0.95, 0.05],
        "different": [0.0, 1.0],
    }))
    config = tmp_path / "representation_config.json"
    config.write_text(json.dumps({
        "mode": "assist",
        "embeddings_path": "embeddings.json",
        "strength": 0.35,
    }))
    assist = RepresentationAssist.from_config(config)
    assert assist.active is True
    related = assist.continuity_factor("anchor", "related")
    different = assist.continuity_factor("anchor", "different")
    assert 0.82 <= related < 1.0 < different <= 1.18
    assert assist.continuity_factor("anchor", "anchor") == 1.0
    assert assist.continuity_factor("anchor", "unknown") == 1.0


def test_invalid_config_fails_closed_without_crashing(tmp_path):
    config = tmp_path / "representation_config.json"
    config.write_text(json.dumps({"mode": "assist", "strength": 0.9}))
    assist = RepresentationAssist.from_config(config)
    assert assist.mode == "off"
    assert assist.active is False
    assert assist.error
    assert assist.continuity_factor("a", "b") == pytest.approx(1.0)


def test_generator_assist_is_limited_to_same_validated_role(monkeypatch):
    assist = RepresentationAssist(mode="assist", strength=0.35)
    assist.embeddings = {
        "a": np.asarray([1.0, 0.0]),
        "b": np.asarray([0.95, 0.05]) / np.linalg.norm([0.95, 0.05]),
    }
    monkeypatch.setattr(generator, "REPRESENTATION_ASSIST", assist)
    texture_a = {"object_id": "a", "role": "texture"}
    texture_b = {"object_id": "b", "role": "texture"}
    gesture_b = {"object_id": "b", "role": "gesture"}
    resonance_a = {"object_id": "a", "role": "resonance"}
    resonance_b = {"object_id": "b", "role": "resonance"}
    assert generator.representation_continuity_factor(texture_a, texture_b) < 1.0
    assert generator.representation_continuity_factor(texture_a, gesture_b) == 1.0
    assert generator.representation_continuity_factor(resonance_a, resonance_b) == 1.0

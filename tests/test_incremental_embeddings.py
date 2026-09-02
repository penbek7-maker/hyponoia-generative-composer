import json

import numpy as np
import pytest
import soundfile as sf
import torch

from incremental_embeddings_v1 import prepare_from_config, prepare_incremental_embeddings
from representation_feedback_v2 import DiagonalMetricAdapter
from representation_learning_v1 import ContrastiveEncoder


def _make_dataset(tmp_path):
    memory = tmp_path / "library"
    memory.mkdir()
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    audio = np.concatenate(
        [
            0.2 * np.sin(2 * np.pi * 220 * time[:sample_rate]),
            0.2 * np.sin(2 * np.pi * 660 * time[:sample_rate]),
        ]
    ).astype(np.float32)
    sf.write(memory / "source.wav", audio, sample_rate)
    index = [
        {
            "recording": "source.wav",
            "objects": [
                {"stable_id": "obj_keep", "start_sample": 0, "end_sample": sample_rate},
                {
                    "stable_id": "obj_new",
                    "start_sample": sample_rate,
                    "end_sample": sample_rate * 2,
                },
            ],
        }
    ]
    index_path = tmp_path / "memory_index_v3.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return memory, index_path


def _write_models(tmp_path):
    encoder = ContrastiveEncoder(32)
    encoder_path = tmp_path / "encoder_v1.pt"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "embedding_dim": 32,
            "frames": 32,
        },
        encoder_path,
    )
    adapter = DiagonalMetricAdapter(32)
    with torch.no_grad():
        adapter.log_scale.copy_(torch.linspace(-0.2, 0.2, 32))
    adapter_path = tmp_path / "metric_adapter_v2.pt"
    torch.save(
        {
            "model_state_dict": adapter.state_dict(),
            "embedding_dim": 32,
            "adapter_type": "positive_diagonal",
        },
        adapter_path,
    )
    return encoder_path, adapter_path


def test_incremental_refresh_reuses_adds_and_prunes_with_frozen_models(tmp_path):
    memory, index_path = _make_dataset(tmp_path)
    encoder_path, adapter_path = _write_models(tmp_path)
    keep_vector = np.zeros(32, dtype=np.float32)
    keep_vector[0] = 1.0
    old_vector = np.zeros(32, dtype=np.float32)
    old_vector[1] = 1.0
    embeddings_path = tmp_path / "embeddings_v2.json"
    embeddings_path.write_text(
        json.dumps({"obj_keep": keep_vector.tolist(), "obj_removed": old_vector.tolist()}),
        encoding="utf-8",
    )

    payload, report = prepare_incremental_embeddings(
        index_path,
        memory,
        embeddings_path,
        encoder_path=encoder_path,
        adapter_path=adapter_path,
        batch_size=2,
    )

    assert set(payload) == {"obj_keep", "obj_new"}
    assert payload["obj_keep"] == keep_vector.tolist()
    assert np.linalg.norm(payload["obj_new"]) == pytest.approx(1.0, abs=1e-5)
    assert report["reused_embeddings"] == 1
    assert report["created_embeddings"] == 1
    assert report["removed_embeddings"] == 1
    assert report["encoder_used"] is True
    assert report["adapter_used"] is True
    assert report["full_retraining"] is False


def test_removal_only_needs_no_encoder(tmp_path):
    memory, index_path = _make_dataset(tmp_path)
    index = json.loads(index_path.read_text())
    index[0]["objects"] = index[0]["objects"][:1]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    embeddings_path = tmp_path / "embeddings.json"
    vector = [1.0] + [0.0] * 31
    embeddings_path.write_text(
        json.dumps({"obj_keep": vector, "obj_removed": vector}), encoding="utf-8"
    )

    payload, report = prepare_incremental_embeddings(
        index_path,
        memory,
        embeddings_path,
        encoder_path=None,
    )

    assert set(payload) == {"obj_keep"}
    assert report["removed_embeddings"] == 1
    assert report["created_embeddings"] == 0
    assert report["encoder_used"] is False


def test_new_objects_fail_safely_when_encoder_is_not_configured(tmp_path):
    memory, index_path = _make_dataset(tmp_path)
    embeddings_path = tmp_path / "embeddings.json"
    embeddings_path.write_text(json.dumps({"obj_keep": [1.0] + [0.0] * 31}))

    with pytest.raises(FileNotFoundError, match="encoder_path"):
        prepare_incremental_embeddings(
            index_path,
            memory,
            embeddings_path,
            encoder_path=None,
        )


def test_off_config_reports_disabled_without_creating_embeddings(tmp_path):
    memory, index_path = _make_dataset(tmp_path)
    config_path = tmp_path / "representation_config.json"
    config_path.write_text(json.dumps({"mode": "off"}), encoding="utf-8")

    path, payload, report = prepare_from_config(index_path, memory, config_path)

    assert path is None
    assert payload is None
    assert report["status"] == "disabled"
    assert report["full_retraining"] is False


def test_assist_config_resolves_relative_artifact_paths(tmp_path):
    memory, index_path = _make_dataset(tmp_path)
    encoder_path, adapter_path = _write_models(tmp_path)
    embeddings_path = tmp_path / "embeddings_v2.json"
    embeddings_path.write_text(json.dumps({"obj_keep": [1.0] + [0.0] * 31}))
    config_path = tmp_path / "representation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "assist",
                "embeddings_path": embeddings_path.name,
                "encoder_path": encoder_path.name,
                "metric_adapter_path": adapter_path.name,
                "strength": 0.35,
            }
        ),
        encoding="utf-8",
    )

    path, payload, report = prepare_from_config(index_path, memory, config_path)

    assert path == embeddings_path.resolve()
    assert set(payload) == {"obj_keep", "obj_new"}
    assert report["created_embeddings"] == 1
    assert report["config_path"] == str(config_path.resolve())

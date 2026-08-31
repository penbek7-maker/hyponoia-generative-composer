import json

import numpy as np
import soundfile as sf
import torch

from representation_learning_v1 import (
    ContrastiveEncoder,
    contrastive_loss,
    export_embeddings,
    load_sound_objects,
    log_mel,
    read_fragment,
)


def test_stable_ids_resolve_to_audio_fragments(tmp_path):
    memory = tmp_path / "alpha_memory"
    memory.mkdir()
    audio = np.linspace(-0.5, 0.5, 4_800, dtype=np.float32)
    sf.write(memory / "source.wav", audio, 48_000)
    index = [{"recording": "source.wav", "objects": [{"stable_id": "obj_abc", "start_sample": 100, "end_sample": 1100}]}]
    index_path = tmp_path / "memory_index_v3.json"
    index_path.write_text(json.dumps(index))
    objects = load_sound_objects(index_path, memory)
    assert objects[0].stable_id == "obj_abc"
    assert read_fragment(objects[0]).shape == (1000,)


def test_fragment_boundaries_use_canonical_48khz_grid(tmp_path):
    memory = tmp_path / "alpha_memory"
    memory.mkdir()
    sf.write(memory / "source.wav", np.ones(44_100, dtype=np.float32), 44_100)
    index = [{"recording": "source.wav", "objects": [{"stable_id": "obj_resampled", "start_sample": 24_000, "end_sample": 48_000}]}]
    index_path = tmp_path / "memory_index_v3.json"
    index_path.write_text(json.dumps(index))
    item = load_sound_objects(index_path, memory)[0]
    assert read_fragment(item).shape == (24_000,)


def test_log_mel_is_normalized_and_finite():
    t = np.arange(9_600, dtype=np.float32) / 48_000
    representation = log_mel(np.sin(2 * np.pi * 440 * t).astype(np.float32))
    assert representation.shape[0:2] == (1, 64)
    assert torch.isfinite(representation).all()
    assert abs(float(representation.mean())) < 1e-4


def test_encoder_and_contrastive_loss_are_well_formed():
    model = ContrastiveEncoder(embedding_dim=32)
    first = model(torch.randn(3, 1, 64, 32))
    second = model(torch.randn(3, 1, 64, 32))
    loss = contrastive_loss(first, second)
    assert first.shape == (3, 32)
    assert torch.allclose(first.norm(dim=1), torch.ones(3), atol=1e-5)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_embeddings_export_by_stable_id(tmp_path):
    output = tmp_path / "embeddings.json"
    export_embeddings(["obj_a", "obj_b"], torch.zeros(2, 32), output)
    payload = json.loads(output.read_text())
    assert set(payload) == {"obj_a", "obj_b"}
    assert all(len(vector) == 32 for vector in payload.values())

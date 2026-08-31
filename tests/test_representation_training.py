import json

import numpy as np
import soundfile as sf

from representation_training_v1 import fixed_frames, train_representation_model
from representation_learning_v1 import log_mel


def make_dataset(tmp_path):
    memory = tmp_path / "alpha_memory"
    memory.mkdir()
    sr = 48_000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.concatenate([np.sin(2 * np.pi * hz * t / 4) for hz in (220, 330, 440, 550)]).astype(np.float32)
    sf.write(memory / "source.wav", audio, sr)
    objects = []
    for index in range(4):
        objects.append({"stable_id": f"obj_{index:020x}", "start_sample": index * sr, "end_sample": (index + 1) * sr})
    (tmp_path / "memory_index_v3.json").write_text(json.dumps([{"recording": "source.wav", "objects": objects}]))


def test_fixed_frames_crops_and_pads():
    short = fixed_frames(log_mel(np.ones(2_000, dtype=np.float32)), 32)
    long = fixed_frames(log_mel(np.ones(20_000, dtype=np.float32)), 32)
    assert short.shape == long.shape == (1, 64, 32)


def test_smoke_training_writes_isolated_artifacts(tmp_path):
    make_dataset(tmp_path)
    output = tmp_path / "run"
    manifest = train_representation_model(
        tmp_path / "memory_index_v3.json",
        tmp_path / "alpha_memory",
        output,
        epochs=1,
        batch_size=2,
        frames=32,
        seed=7,
        device="cpu",
    )
    assert manifest["objects_seen"] == 4
    assert manifest["gate1_integration"] is False
    assert (output / "encoder_v1.pt").exists()
    embeddings = json.loads((output / "embeddings_v1.json").read_text())
    assert len(embeddings) == 4
    assert all(len(vector) == 32 for vector in embeddings.values())


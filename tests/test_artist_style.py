import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from artist_style_v1 import ArtistStyleAssist, train_artist_style
from representation_learning_v1 import EMBEDDING_DIM, ContrastiveEncoder


def _work(sample_rate, seconds, frequency, motion):
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    envelope = 0.35 + 0.65 * np.sin(np.pi * time / seconds) ** 2
    carrier = np.sin(2.0 * np.pi * frequency * time)
    detail = 0.32 * np.sin(2.0 * np.pi * (frequency * 2.41) * time)
    pan = motion * np.sin(2.0 * np.pi * 0.37 * time)
    left = (carrier + detail) * envelope * np.sqrt((1.0 - pan) * 0.5) * 0.2
    right = (carrier + detail) * envelope * np.sqrt((1.0 + pan) * 0.5) * 0.2
    return np.stack([left, right], axis=1).astype(np.float32)


def test_artist_style_training_is_isolated_and_loadable(tmp_path):
    audio_dir = tmp_path / "owned"
    audio_dir.mkdir()
    sf.write(audio_dir / "first.wav", _work(48_000, 2.2, 173.0, 0.55), 48_000)
    sf.write(audio_dir / "second.wav", _work(44_100, 2.4, 281.0, 0.75), 44_100)
    encoder_path = tmp_path / "encoder.pt"
    torch.save(
        {
            "model_state_dict": ContrastiveEncoder(EMBEDDING_DIM).state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "frames": 128,
        },
        encoder_path,
    )
    output = tmp_path / "artist_style.json"
    profile = train_artist_style(
        audio_dir,
        output,
        encoder_path=encoder_path,
        strength=0.20,
    )

    assert output.exists()
    assert profile["work_count"] == 2
    assert profile["training_diagnostics"]["raw_audio_copied"] is False
    assert profile["training_diagnostics"]["encoder_fine_tuned"] is False
    assert len(profile["artist_prototype"]) == EMBEDDING_DIM
    assert 1.05 <= profile["controls"]["expressive_drive"] <= 1.30
    assert all(len(work["descriptors"]["energy_curve"]) == 12 for work in profile["works"])
    assert all(len(work["descriptors"]["tonal_curve"]) == 12 for work in profile["works"])
    assert all(len(work["descriptors"]["pan_curve"]) == 12 for work in profile["works"])
    assert not list(tmp_path.glob("**/*.wav")) == []
    assist = ArtistStyleAssist.from_file(output)
    assert assist.active
    assert assist.work_count == 2


def test_artist_style_controls_are_bounded_and_level_sensitive(tmp_path):
    prototype = np.zeros(4, dtype=float)
    prototype[0] = 1.0
    payload = {
        "schema_version": "artist_style_v1",
        "mode": "assist",
        "strength": 0.30,
        "work_count": 2,
        "artist_prototype": prototype.tolist(),
        "controls": {"pan_motion": 1.34},
        "works": [
            {"descriptors": {"energy_curve": [0.2, 0.8] * 6, "density_curve": [0.8, 0.2] * 6, "pan_curve": [-0.6, 0.4] * 6}},
            {"descriptors": {"energy_curve": [0.8, 0.2] * 6, "density_curve": [0.2, 0.8] * 6, "pan_curve": [0.3, -0.5] * 6}},
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assist = ArtistStyleAssist.from_file(path)

    assert 1.0 < assist.control("pan_motion", 1) < assist.control("pan_motion", 5) <= 1.38
    assert assist.object_factor(np.asarray([1.0, 0.0, 0.0, 0.0]), 5) < 1.0
    assert assist.object_factor(np.asarray([-1.0, 0.0, 0.0, 0.0]), 5) > 1.0
    for level in (1, 3, 5):
        assert 0.78 <= assist.form_factor(0.65, level, 1234) <= 1.30
    positions = np.linspace(0.0, 1.0, 19)
    first = assist.trajectory_curve("energy", positions, 5, 1234)
    second = assist.trajectory_curve("energy", positions, 5, 1234)
    assert np.allclose(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))
    pan = assist.trajectory_curve("pan", positions, 5, 1234)
    assert np.max(np.abs(pan)) > 0.10
    assert np.all((pan >= -1.0) & (pan <= 1.0))


def test_artist_style_loader_fails_closed(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    assist = ArtistStyleAssist.from_file(path)
    assert not assist.active
    assert assist.error


def test_release_artist_style_baseline_is_loadable():
    path = Path(__file__).resolve().parents[1] / "phase2_artifacts" / "artist_style_baseline_v1.json"
    assist = ArtistStyleAssist.from_file(path)

    assert assist.active
    assert assist.work_count == 5
    assert assist.strength == 0.32
    assert assist.control("granulation", 5) == 1.34

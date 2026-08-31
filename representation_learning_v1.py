"""Experimental Hyponoia Phase 2 representation learning.

This module is deliberately independent from the Gate 1 generator, critic,
feedback and OSC path. It does not train or mutate project data on import.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F


TARGET_SR = 48_000
N_MELS = 64
EMBEDDING_DIM = 32


@dataclass(frozen=True)
class SoundObject:
    stable_id: str
    audio_path: Path
    start_sample: int
    end_sample: int


def load_sound_objects(index_path: str | Path, memory_dir: str | Path) -> list[SoundObject]:
    """Read Gate 1 stable IDs and resolve their source-audio slices."""
    index_path, memory_dir = Path(index_path), Path(memory_dir)
    records = json.loads(index_path.read_text(encoding="utf-8"))
    objects: list[SoundObject] = []
    for record in records:
        audio_path = memory_dir / record["recording"]
        for item in record.get("objects", []):
            stable_id = item.get("stable_id")
            if not stable_id:
                raise ValueError("Every sound object must have a stable_id")
            objects.append(SoundObject(stable_id, audio_path, int(item["start_sample"]), int(item["end_sample"])))
    return objects


def read_fragment(item: SoundObject, target_sr: int = TARGET_SR) -> np.ndarray:
    info = sf.info(item.audio_path)
    # Gate 1 object boundaries index canonical 48 kHz audio, not necessarily the
    # source file's native sample grid. Resample first whenever those grids differ.
    if info.samplerate == target_sr:
        audio, sr = sf.read(
            item.audio_path,
            start=item.start_sample,
            stop=item.end_sample,
            dtype="float32",
            always_2d=False,
        )
    else:
        audio, sr = sf.read(item.audio_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = audio[item.start_sample : item.end_sample]
    return np.asarray(audio, dtype=np.float32)


def log_mel(audio: np.ndarray, sr: int = TARGET_SR, n_mels: int = N_MELS) -> torch.Tensor:
    """Return a finite, normalized [1, mel, time] representation."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Cannot represent an empty sound object")
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels, power=2.0)
    x = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.from_numpy(x).unsqueeze(0)


def augment(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Lightweight spectrogram masking/noise for two contrastive views."""
    y = x.clone()
    noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype) * 0.01
    y = y + noise
    if y.shape[-1] > 4:
        width = max(1, y.shape[-1] // 12)
        start = int(torch.randint(0, y.shape[-1] - width + 1, (1,), generator=generator).item())
        y[..., start : start + width] = 0
    return y


class ContrastiveEncoder(nn.Module):
    """Small convolutional encoder producing L2-normalized embeddings."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        z = self.features(x).flatten(1)
        return F.normalize(self.projection(z), dim=1)


def contrastive_loss(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric in-batch InfoNCE loss for paired views."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Contrastive batches must have equal [batch, embedding] shapes")
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_embeddings(stable_ids: Iterable[str], embeddings: torch.Tensor, output_path: str | Path) -> None:
    """Persist embeddings keyed only by stable ID; never alter Gate 1 indexes."""
    ids = list(stable_ids)
    values = embeddings.detach().cpu().numpy()
    if len(ids) != len(values):
        raise ValueError("stable_ids and embeddings must have the same length")
    payload = {key: [float(v) for v in row] for key, row in zip(ids, values)}
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

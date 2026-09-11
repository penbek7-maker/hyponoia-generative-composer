"""Train and apply a bounded artist-style head from complete owned works.

The frozen Hyponoia encoder is never fine-tuned here. Training stores only
aggregate trajectories, descriptors and an embedding prototype; it never
copies source audio or makes the works part of the generation library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import find_peaks
from torch.nn import functional as F

from hyponoia_stability import TARGET_SR, atomic_write_json, utc_timestamp
from incremental_embeddings_v1 import _load_adapter, _load_encoder
from representation_learning_v1 import log_mel
from representation_training_v1 import fixed_frames


SUPPORTED_AUDIO = frozenset({".wav", ".aif", ".aiff", ".flac"})
TRAJECTORY_BINS = 12


def _unit(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-12:
        raise ValueError("invalid artist-style vector")
    return vector / norm


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_stereo(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != TARGET_SR:
        channels = [
            librosa.resample(audio[:, channel], orig_sr=sample_rate, target_sr=TARGET_SR)
            for channel in range(audio.shape[1])
        ]
        shortest = min(map(len, channels))
        audio = np.stack([channel[:shortest] for channel in channels], axis=1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if len(audio) < TARGET_SR:
        raise ValueError(f"Artist work must be at least one second: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Artist work contains non-finite samples: {path}")
    return np.asarray(audio, dtype=np.float32)


def _bin_means(values: np.ndarray, bins: int = TRAJECTORY_BINS) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0] * bins
    chunks = np.array_split(values, bins)
    return [float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks]


def _relative_curve(values: np.ndarray) -> list[float]:
    curve = np.asarray(_bin_means(values), dtype=np.float64)
    low, high = np.percentile(curve, [10, 90])
    if high - low < 1e-9:
        return [0.5] * len(curve)
    return np.clip((curve - low) / (high - low), 0.0, 1.0).tolist()


def analyse_artist_work(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """Measure a complete work without treating mastering level as style."""
    source = Path(path).expanduser().resolve()
    audio = _load_stereo(source)
    mono = np.mean(audio, axis=1)
    hop = 1024
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=mono, sr=TARGET_SR, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=mono, hop_length=hop)[0]
    chroma = librosa.feature.chroma_stft(y=mono, sr=TARGET_SR, hop_length=hop)
    onset = librosa.onset.onset_strength(y=mono, sr=TARGET_SR, hop_length=hop)
    onset_scale = float(np.percentile(onset, 90)) + 1e-9
    onset_relative = np.clip(onset / onset_scale, 0.0, 2.0)
    peaks, _ = find_peaks(
        onset_relative,
        height=0.72,
        distance=max(1, int(0.22 * TARGET_SR / hop)),
    )

    left_rms = librosa.feature.rms(y=audio[:, 0], frame_length=2048, hop_length=hop)[0]
    right_rms = librosa.feature.rms(y=audio[:, 1], frame_length=2048, hop_length=hop)[0]
    count = min(
        len(left_rms), len(right_rms), len(rms), len(centroid),
        len(flatness), len(onset_relative), chroma.shape[1],
    )
    left_rms, right_rms = left_rms[:count], right_rms[:count]
    rms, centroid, flatness, onset_relative = (
        rms[:count], centroid[:count], flatness[:count], onset_relative[:count]
    )
    pan = (right_rms - left_rms) / (left_rms + right_rms + 1e-8)
    pan_smooth = np.convolve(pan, np.ones(9) / 9.0, mode="same")
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5
    mid_rms = float(np.sqrt(np.mean(mid * mid))) + 1e-9
    side_rms = float(np.sqrt(np.mean(side * side)))
    rms_reference = float(np.percentile(rms, 75)) + 1e-9
    energy_relative = np.clip(rms / rms_reference, 0.0, 2.5)
    spectral_relative = centroid / (float(np.median(centroid)) + 1e-9)
    chroma = chroma[:, :count]
    chroma /= np.sum(chroma, axis=0, keepdims=True) + 1e-9
    pitch_angles = np.exp(2j * np.pi * np.arange(12, dtype=np.float64) / 12.0)
    tonal_vector = np.sum(chroma * pitch_angles[:, None], axis=0)
    tonal_confidence_curve = np.abs(tonal_vector)
    tonal_phase = np.unwrap(np.angle(tonal_vector)) * 6.0 / np.pi
    tonal_phase -= float(np.median(tonal_phase))
    tonal_position = np.clip(tonal_phase / 6.0, -1.0, 1.0)
    # Macro-phrase boundaries must not collapse into every small acousmatic
    # attack. Smooth the novelty trace and keep boundaries several seconds
    # apart; micro-activity remains represented by onset_rate_per_minute.
    macro_frames = max(3, int(1.25 * TARGET_SR / hop))
    macro_kernel = np.ones(macro_frames, dtype=np.float64) / macro_frames
    smooth_energy = np.convolve(energy_relative, macro_kernel, mode="same")
    smooth_spectral = np.convolve(spectral_relative, macro_kernel, mode="same")
    smooth_onset = np.convolve(onset_relative, macro_kernel, mode="same")
    novelty = (
        0.38 * np.abs(np.diff(smooth_energy, prepend=smooth_energy[0]))
        + 0.40 * np.abs(np.diff(smooth_spectral, prepend=smooth_spectral[0]))
        + 0.22 * smooth_onset
    )
    phrase_peaks, _ = find_peaks(
        novelty,
        height=float(np.percentile(novelty, 70)),
        distance=max(1, int(3.0 * TARGET_SR / hop)),
    )
    phrase_times = phrase_peaks * hop / TARGET_SR
    phrase_intervals = np.diff(phrase_times)

    descriptors = {
        "duration_sec": float(len(audio) / TARGET_SR),
        "onset_rate_per_minute": float(len(peaks) / (len(audio) / TARGET_SR) * 60.0),
        "energy_contrast": float(
            np.percentile(rms, 90) / (float(np.percentile(rms, 20)) + 1e-9)
        ),
        "dynamic_motion": float(np.mean(np.abs(np.diff(energy_relative)))),
        "spectral_development": float(np.mean(np.abs(np.diff(spectral_relative)))),
        "tonal_development": float(
            np.mean(np.abs(np.diff(tonal_position)))
            * np.mean(tonal_confidence_curve)
        ),
        "tonal_confidence": float(np.mean(tonal_confidence_curve)),
        "spectral_flatness": float(np.mean(flatness)),
        "stereo_width": float(np.clip(side_rms / mid_rms, 0.0, 2.0)),
        "pan_motion": float(np.mean(np.abs(np.diff(pan_smooth)))),
        "phrase_interval_sec": float(np.median(phrase_intervals)) if len(phrase_intervals) else 8.0,
        "sustained_fraction": float(np.mean(onset_relative < 0.32)),
        "energy_curve": _relative_curve(energy_relative),
        "density_curve": _relative_curve(onset_relative),
        "brightness_curve": _relative_curve(centroid),
        "tonal_curve": _bin_means(tonal_position),
        "pan_curve": _bin_means(pan_smooth),
    }
    evidence = {
        "filename": source.name,
        "sha256": _file_hash(source),
        "duration_sec": descriptors["duration_sec"],
        "sample_rate": TARGET_SR,
        "channels": 2,
    }
    return {"evidence": evidence, "descriptors": descriptors}, audio


def _segment_embeddings(
    audio: np.ndarray,
    encoder: torch.nn.Module,
    frames: int,
    adapter: torch.nn.Module | None,
    *,
    max_segments: int = 24,
    segment_sec: float = 8.0,
) -> np.ndarray:
    mono = np.mean(audio, axis=1)
    size = max(1, int(segment_sec * TARGET_SR))
    if len(mono) <= size:
        starts = np.asarray([0], dtype=int)
    else:
        starts = np.linspace(
            0,
            len(mono) - size,
            min(max_segments, 1 + len(mono) // size),
        ).astype(int)
    features = []
    for start in starts:
        fragment = mono[start : start + size]
        features.append(fixed_frames(log_mel(fragment), frames))
    with torch.no_grad():
        vectors = encoder(torch.stack(features))
        if adapter is not None:
            vectors = adapter(vectors)
        vectors = F.normalize(vectors, dim=1)
    return vectors.cpu().numpy().astype(np.float64)


def _bounded_controls(descriptors: list[dict[str, Any]]) -> dict[str, float]:
    median = lambda key: float(np.median([float(item[key]) for item in descriptors]))
    phrase_interval = median("phrase_interval_sec")
    sustained = median("sustained_fraction")
    spectral = median("spectral_development")
    pan = median("pan_motion")
    width = median("stereo_width")
    recurrence = median("embedding_recurrence")
    contrast = median("energy_contrast")
    dynamic_motion = median("dynamic_motion")
    tonal_development = median("tonal_development")
    return {
        "phrase_presence": float(np.clip(1.00 + 0.030 * phrase_interval, 1.00, 1.28)),
        "phrase_persistence": float(np.clip(0.98 + 0.040 * phrase_interval, 1.00, 1.34)),
        "recurrence": float(np.clip(0.90 + 0.38 * recurrence, 1.00, 1.30)),
        "granulation": float(np.clip(1.00 + 2.8 * spectral, 0.94, 1.34)),
        "drone_presence": float(np.clip(1.00 + 0.22 * sustained, 1.00, 1.25)),
        "spatial_width": float(np.clip(0.94 + 0.18 * width, 0.94, 1.28)),
        "pan_motion": float(np.clip(0.98 + 7.0 * pan, 0.98, 1.34)),
        # Corpus-derived expressive pressure. This is deliberately bounded: it
        # changes how confidently source phrases develop, but cannot turn the
        # five works into copied templates or bypass the user's D-level choice.
        "expressive_drive": float(
            np.clip(0.96 + 0.025 * contrast + 1.8 * dynamic_motion, 1.05, 1.30)
        ),
        "tonal_development": float(
            np.clip(1.00 + 1.8 * tonal_development, 1.00, 1.30)
        ),
    }


def train_artist_style(
    audio_dir: str | Path,
    output_path: str | Path,
    *,
    encoder_path: str | Path,
    adapter_path: str | Path | None = None,
    strength: float = 0.24,
) -> dict[str, Any]:
    """Build an isolated style head from a directory of complete works."""
    if not 0.0 <= strength <= 0.35:
        raise ValueError("artist-style strength must be between 0 and 0.35")
    root = Path(audio_dir).expanduser().resolve()
    paths = sorted(path for path in root.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO)
    if len(paths) < 2:
        raise ValueError("artist-style training requires at least two owned works")
    encoder, embedding_dim, frames = _load_encoder(Path(encoder_path).expanduser().resolve())
    adapter = _load_adapter(
        Path(adapter_path).expanduser().resolve() if adapter_path else None,
        embedding_dim,
    )
    work_records = []
    work_vectors = []
    for path in paths:
        record, audio = analyse_artist_work(path)
        vectors = _segment_embeddings(audio, encoder, frames, adapter)
        work_vectors.append(_unit(np.mean(vectors, axis=0)))
        if len(vectors) >= 4:
            similarity = vectors @ vectors.T
            np.fill_diagonal(similarity, -1.0)
            for index in range(len(vectors) - 1):
                similarity[index, index + 1] = -1.0
                similarity[index + 1, index] = -1.0
            recurrence = float(np.mean(np.max(similarity, axis=1)))
        else:
            recurrence = 0.5
        record["descriptors"]["embedding_recurrence"] = float(
            np.clip(recurrence, 0.0, 1.0)
        )
        record["evidence"]["embedded_segments"] = int(len(vectors))
        work_records.append(record)
    prototype = _unit(np.mean(np.stack(work_vectors), axis=0))
    similarities = [float(np.dot(vector, prototype)) for vector in work_vectors]
    descriptors = [record["descriptors"] for record in work_records]
    profile = {
        "schema_version": "artist_style_v1",
        "model_type": "frozen_encoder_aggregate_trajectory_head",
        "trained_at": utc_timestamp(),
        "mode": "assist",
        "strength": float(strength),
        "work_count": len(work_records),
        "embedding_dimensions": int(len(prototype)),
        "artist_prototype": prototype.tolist(),
        "controls": _bounded_controls(descriptors),
        "works": work_records,
        "training_diagnostics": {
            "work_to_prototype_similarity": similarities,
            "mean_similarity": float(np.mean(similarities)),
            "minimum_similarity": float(np.min(similarities)),
            "raw_audio_copied": False,
            "encoder_fine_tuned": False,
            "policy": "owned works; aggregate learning only; bounded assist; no reproduction claim",
        },
    }
    atomic_write_json(Path(output_path).expanduser().resolve(), profile)
    return profile


@dataclass
class ArtistStyleAssist:
    mode: str = "off"
    strength: float = 0.0
    prototype: np.ndarray | None = None
    controls: dict[str, float] = field(default_factory=dict)
    energy_curves: list[np.ndarray] = field(default_factory=list)
    density_curves: list[np.ndarray] = field(default_factory=list)
    brightness_curves: list[np.ndarray] = field(default_factory=list)
    tonal_curves: list[np.ndarray] = field(default_factory=list)
    pan_curves: list[np.ndarray] = field(default_factory=list)
    work_count: int = 0
    source_path: str | None = None
    error: str | None = None

    @classmethod
    def disabled(cls) -> "ArtistStyleAssist":
        return cls()

    @classmethod
    def from_file(cls, path: str | Path | None) -> "ArtistStyleAssist":
        if not path:
            return cls.disabled()
        source = Path(path).expanduser()
        if not source.exists():
            return cls(mode="off", source_path=str(source), error=f"Artist style not found: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "artist_style_v1":
                raise ValueError("unsupported artist-style schema")
            strength = float(payload.get("strength", 0.0))
            if not 0.0 <= strength <= 0.35:
                raise ValueError("artist-style strength must be between 0 and 0.35")
            works = list(payload.get("works", []))
            return cls(
                mode=str(payload.get("mode", "off")),
                strength=strength,
                prototype=_unit(np.asarray(payload["artist_prototype"], dtype=np.float64)),
                controls={str(k): float(v) for k, v in dict(payload.get("controls", {})).items()},
                energy_curves=[np.asarray(w["descriptors"]["energy_curve"], dtype=np.float64) for w in works],
                density_curves=[np.asarray(w["descriptors"]["density_curve"], dtype=np.float64) for w in works],
                brightness_curves=[
                    np.asarray(
                        w["descriptors"].get(
                            "brightness_curve", w["descriptors"]["energy_curve"]
                        ),
                        dtype=np.float64,
                    )
                    for w in works
                ],
                tonal_curves=[
                    np.asarray(
                        w["descriptors"].get("tonal_curve", [0.0] * TRAJECTORY_BINS),
                        dtype=np.float64,
                    )
                    * min(
                        1.0,
                        max(0.0, float(w["descriptors"].get("tonal_confidence", 0.0)))
                        / 0.35,
                    )
                    for w in works
                ],
                pan_curves=[
                    np.asarray(
                        w["descriptors"].get("pan_curve", [0.0] * TRAJECTORY_BINS),
                        dtype=np.float64,
                    )
                    for w in works
                ],
                work_count=int(payload.get("work_count", len(works))),
                source_path=str(source.resolve()),
            )
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return cls(mode="off", source_path=str(source), error=str(exc))

    @property
    def active(self) -> bool:
        return self.mode == "assist" and self.prototype is not None and self.work_count >= 2 and self.error is None

    def control(self, name: str, dream_level: int, default: float = 1.0) -> float:
        if not self.active:
            return float(default)
        learned = float(self.controls.get(name, default))
        blend = {1: 0.42, 3: 0.68, 5: 1.0}.get(int(dream_level), 0.68)
        return float(np.clip(1.0 + (learned - 1.0) * blend, 0.82, 1.38))

    def object_factor(self, vector: np.ndarray | None, dream_level: int) -> float:
        """Lower favours library objects near the artist's aggregate timbral region."""
        if not self.active or vector is None:
            return 1.0
        values = _unit(np.asarray(vector, dtype=np.float64))
        similarity = float(np.clip(np.dot(values, self.prototype), -1.0, 1.0))
        level = {1: 0.48, 3: 0.72, 5: 1.0}.get(int(dream_level), 0.72)
        factor = float(np.exp(-self.strength * level * (similarity - 0.45)))
        return float(np.clip(factor, 0.84, 1.16))

    def trajectory_curve(
        self,
        name: str,
        positions: np.ndarray,
        dream_level: int,
        seed: int,
    ) -> np.ndarray:
        """Return a trajectory sampled from the owned-work aggregate."""
        positions = np.asarray(positions, dtype=np.float64)
        if not self.active:
            neutral = 0.0 if name == "tonal" else 0.5
            return np.full_like(positions, neutral)
        curves = {
            "energy": self.energy_curves,
            "density": self.density_curves,
            "brightness": self.brightness_curves,
            "tonal": self.tonal_curves,
            "pan": self.pan_curves,
        }.get(str(name), [])
        if not curves:
            neutral = 0.0 if name in {"tonal", "pan"} else 0.5
            return np.full_like(positions, neutral)
        index = (int(seed) // 997 + int(dream_level)) % len(curves)
        other = (index + 1 + int(seed) % max(1, len(curves))) % len(curves)
        curve = 0.68 * curves[index] + 0.32 * curves[other]
        return np.interp(
            np.clip(positions, 0.0, 1.0),
            np.linspace(0.0, 1.0, len(curve)),
            curve,
        )

    def form_factor(self, position: float, dream_level: int, seed: int) -> float:
        """Sample corpus-derived trajectories while retaining render variation."""
        if not self.active or not self.energy_curves:
            return 1.0
        index = (int(seed) // 997 + int(dream_level)) % len(self.energy_curves)
        other = (index + 1 + int(seed) % max(1, len(self.energy_curves))) % len(self.energy_curves)
        curve = 0.68 * self.energy_curves[index] + 0.32 * self.density_curves[other]
        curve = curve / (float(np.mean(curve)) + 1e-9)
        point = float(
            np.interp(
                np.clip(position, 0.0, 1.0),
                np.linspace(0.0, 1.0, len(curve)),
                curve,
            )
        )
        blend = {1: 0.12, 3: 0.20, 5: 0.34}.get(int(dream_level), 0.20)
        return float(np.clip(1.0 + (point - 1.0) * blend, 0.78, 1.30))

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active": self.active,
            "strength": self.strength,
            "work_count": self.work_count,
            "controls": self.controls if self.active else {},
            "source_path": self.source_path,
            "error": self.error,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--strength", type=float, default=0.24)
    args = parser.parse_args()
    profile = train_artist_style(
        args.audio_dir,
        args.output,
        encoder_path=args.encoder,
        adapter_path=args.adapter,
        strength=args.strength,
    )
    print(json.dumps(profile["training_diagnostics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Build a configurable Hyponoia acoustic target profile from any WAV file."""

from __future__ import annotations

import argparse

import librosa
import numpy as np

from hyponoia_stability import TARGET_SR, atomic_write_json, stable_recording_id, utc_timestamp


def analyse_reference(reference):
    audio, sample_rate = librosa.load(reference, sr=TARGET_SR, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    rms = librosa.feature.rms(y=audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    return {
        "schema_version": 2,
        "timestamp": utc_timestamp(),
        "reference": reference,
        "reference_id": stable_recording_id(audio, sample_rate),
        "sample_rate": int(sample_rate),
        "energy": float(np.mean(rms)),
        "brightness": float(np.mean(centroid)),
        "noise": float(np.mean(flatness)),
        "attack": float(np.mean(onset)) if len(onset) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", nargs="?", default="output/alpha_memory_drone.wav")
    parser.add_argument("--output", default="alpha_profile.json")
    args = parser.parse_args()
    profile = analyse_reference(args.reference)
    atomic_write_json(args.output, profile)
    print(f"Created {args.output}")
    print(profile)


if __name__ == "__main__":
    main()

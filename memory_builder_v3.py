# Hyponoia Memory Builder v3
# Builds a musical memory, not only an acoustic index.
# Output: memory_index_v3.json

import os
import json
import math

import librosa
import numpy as np

from hyponoia_stability import (
    TARGET_SR,
    atomic_write_json,
    stable_object_id,
    stable_recording_id,
    utc_timestamp,
)

MEMORY_FOLDER = "alpha_memory"
OUTPUT_FILE = "memory_index_v3.json"
REPORT_FILE = "memory_build_report.json"

MIN_OBJECT = 0.7
MAX_OBJECT = 15.0
def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def safe_mean(x, default=0.0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(default)
    return float(np.mean(x))


def safe_std(x, default=0.0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(default)
    return float(np.std(x))


def hz_to_midi_safe(hz):
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return None
    return float(librosa.hz_to_midi(hz))


def load_audio(path):
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32), sr


def detect_sound_objects(audio, sr):
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        backtrack=True,
        delta=0.18,
        wait=int(0.7 * sr / 512)
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    starts = [0.0]

    for t in onset_times:
        if t - starts[-1] >= MIN_OBJECT:
            starts.append(float(t))

    duration = len(audio) / sr
    if starts[-1] < duration:
        starts.append(duration)

    objects = []
    for i in range(len(starts) - 1):
        start = starts[i]
        end = starts[i + 1]

        if end - start > MAX_OBJECT:
            current = start
            while current < end:
                next_end = min(current + MAX_OBJECT, end)
                objects.append((current, next_end))
                current = next_end
        elif end - start >= MIN_OBJECT:
            objects.append((start, end))

    return objects


def estimate_pitch(fragment, sr):
    """Fast-ish pitch estimate. Works best for synth/resonant material; returns confidence too."""
    if len(fragment) < 1024:
        return {
            "pitch_mean": None,
            "pitch_midi": None,
            "pitch_std": None,
            "pitch_range": None,
            "pitch_confidence": 0.0,
            "pitch_motion": 0.0,
        }

    try:
        f0 = librosa.yin(
            fragment,
            fmin=librosa.note_to_hz("C1"),
            fmax=librosa.note_to_hz("C8"),
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        valid = f0[np.isfinite(f0)]
        valid = valid[(valid > 25) & (valid < 5000)]

        if len(valid) < 3:
            return {
                "pitch_mean": None,
                "pitch_midi": None,
                "pitch_std": None,
                "pitch_range": None,
                "pitch_confidence": 0.0,
                "pitch_motion": 0.0,
            }

        pitch_mean = float(np.median(valid))
        midi_vals = librosa.hz_to_midi(valid)
        pitch_midi = float(np.median(midi_vals))
        pitch_std = float(np.std(midi_vals))
        pitch_range = float(np.percentile(midi_vals, 90) - np.percentile(midi_vals, 10))

        # Confidence: stable pitch + enough voiced frames.
        voiced_ratio = len(valid) / max(1, len(f0))
        stability = 1.0 / (1.0 + pitch_std / 3.0)
        confidence = clamp(voiced_ratio * stability)
        motion = clamp(pitch_range / 18.0)

        return {
            "pitch_mean": pitch_mean,
            "pitch_midi": pitch_midi,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "pitch_confidence": confidence,
            "pitch_motion": motion,
        }
    except Exception:
        return {
            "pitch_mean": None,
            "pitch_midi": None,
            "pitch_std": None,
            "pitch_range": None,
            "pitch_confidence": 0.0,
            "pitch_motion": 0.0,
        }


def register_from_pitch_or_brightness(pitch_midi, brightness):
    if pitch_midi is not None and np.isfinite(pitch_midi):
        if pitch_midi < 36:
            return "sub"
        if pitch_midi < 48:
            return "low"
        if pitch_midi < 64:
            return "mid"
        if pitch_midi < 78:
            return "high"
        return "air"

    # fallback for noisy/non-pitched material
    if brightness < 250:
        return "sub"
    if brightness < 900:
        return "low"
    if brightness < 2800:
        return "mid"
    if brightness < 7000:
        return "high"
    return "air"


def estimate_tail_length(rms, sr, hop_length=512):
    if len(rms) == 0:
        return 0.0
    max_rms = float(np.max(rms) + 1e-9)
    threshold = max_rms * 0.12
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return 0.0
    last = int(above[-1])
    tail_frames = max(0, len(rms) - 1 - last)
    return float(tail_frames * hop_length / sr)


def gesture_type_from_features(duration, energy, brightness, noise, attack, harmonicity, pitch_confidence, tail_length):
    if duration >= 5.0 and harmonicity > 0.55:
        return "drone"
    if duration >= 3.0 and noise < 0.035:
        return "texture"
    if pitch_confidence > 0.45 and tail_length > 0.25:
        return "resonance"
    if attack > 5.0 and duration < 2.2 and noise > 0.025:
        return "burst"
    if noise > 0.045:
        return "noise"
    if attack > 3.0 and duration < 4.5:
        return "gesture"
    if brightness > 7000 and energy < 0.035:
        return "air"
    return "texture"


def phrase_role_from_scores(duration, attack, resonance_score, ambient_score, transient_score):
    if transient_score > 0.72:
        return "opening"
    if resonance_score > 0.70 and duration >= 3.0:
        return "ending"
    if ambient_score > 0.70:
        return "continuation"
    if attack > 3.8:
        return "transition"
    return "continuation"


def analyse_object(audio, sr):
    # Basic spectral/envelope descriptors
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    duration = len(audio) / sr
    energy = safe_mean(rms)
    brightness = safe_mean(centroid)
    noise = safe_mean(flatness)
    attack = float(np.max(onset_env)) if len(onset_env) else 0.0
    spectral_bandwidth = safe_mean(bandwidth)
    spectral_rolloff = safe_mean(rolloff)
    zcr_mean = safe_mean(zcr)
    spectral_width = clamp(spectral_bandwidth / 6000.0)

    # Pitch descriptors
    pitch = estimate_pitch(audio, sr)
    pitch_mean = pitch["pitch_mean"]
    pitch_midi = pitch["pitch_midi"]
    pitch_confidence = pitch["pitch_confidence"]
    pitch_motion = pitch["pitch_motion"]

    # Musical descriptors, heuristic but useful and stable.
    harmonicity = clamp((1.0 - min(1.0, noise * 38.0)) * 0.55 + pitch_confidence * 0.45)
    inharmonicity = clamp(1.0 - harmonicity)
    tail_length = estimate_tail_length(rms, sr)
    resonance_strength = clamp((tail_length / 2.5) * 0.35 + harmonicity * 0.45 + (duration / 10.0) * 0.20)

    transient_score = clamp((attack / 7.5) * 0.60 + (1.0 - min(1.0, duration / 4.0)) * 0.40)
    ambient_score = clamp((duration / 8.0) * 0.34 + harmonicity * 0.36 + resonance_strength * 0.30)
    texture_score = clamp((duration / 6.0) * 0.42 + (1.0 - transient_score) * 0.32 + spectral_width * 0.26)
    drone_score = clamp((duration / 8.5) * 0.45 + harmonicity * 0.45 + pitch_confidence * 0.10)
    gesture_strength = clamp((attack / 7.0) * 0.55 + min(1.0, energy / 0.16) * 0.25 + (1.0 - min(1.0, duration / 5.0)) * 0.20)

    # Richness: enough spectral content and motion, but not uncontrolled noise.
    richness = clamp(
        spectral_width * 0.28
        + min(1.0, brightness / 8500.0) * 0.20
        + pitch_motion * 0.18
        + harmonicity * 0.22
        + min(1.0, energy / 0.12) * 0.12
    )

    # General musicality: good material should be usable, not merely loud/bright.
    useful_duration = 1.0 if 1.0 <= duration <= 10.5 else (0.72 if 0.7 <= duration < 1.0 else 0.58)
    harsh_short_penalty = clamp((attack / 7.5) * (noise * 28.0) * (1.0 if duration < 2.2 else 0.35))
    ultra_noisy_penalty = clamp((noise - 0.04) * 10.0)
    musicality = clamp(
        0.20 * useful_duration
        + 0.22 * harmonicity
        + 0.20 * resonance_strength
        + 0.18 * richness
        + 0.12 * ambient_score
        + 0.08 * pitch_confidence
        - 0.28 * harsh_short_penalty
        - 0.18 * ultra_noisy_penalty
    )

    foreground_probability = clamp(gesture_strength * 0.45 + richness * 0.25 + min(1.0, energy / 0.14) * 0.20 + (1.0 - ambient_score) * 0.10)
    background_probability = clamp(ambient_score * 0.45 + resonance_strength * 0.35 + (1.0 - transient_score) * 0.20)
    contrast_score = clamp(abs(brightness - 2500.0) / 9000.0 * 0.35 + gesture_strength * 0.30 + spectral_width * 0.35)
    novelty = clamp(richness * 0.45 + spectral_width * 0.25 + pitch_motion * 0.20 + transient_score * 0.10)
    # A transparent, weak heuristic used only for the deterministic
    # "more synthesizers" MVP intent.  It is not a source classifier.
    synthetic_score = clamp(
        harmonicity * 0.38
        + pitch_confidence * 0.24
        + resonance_strength * 0.18
        + (1.0 - min(1.0, noise * 24.0)) * 0.20
    )

    register = register_from_pitch_or_brightness(pitch_midi, brightness)
    gesture_type = gesture_type_from_features(duration, energy, brightness, noise, attack, harmonicity, pitch_confidence, tail_length)
    phrase_role = phrase_role_from_scores(duration, attack, resonance_strength, ambient_score, transient_score)

    # Emotional / formal function: not emotion-recognition; compositional function.
    if phrase_role == "ending" and resonance_strength > 0.70:
        emotional_function = "release"
    elif ambient_score > 0.75:
        emotional_function = "floating"
    elif transient_score > 0.75:
        emotional_function = "tension"
    elif harmonicity > 0.70 and pitch_confidence > 0.40:
        emotional_function = "arrival"
    else:
        emotional_function = "memory"

    return {
        # v2-compatible descriptors
        "energy": float(energy),
        "brightness": float(brightness),
        "noise": float(noise),
        "attack": float(attack),

        # Pitch descriptors
        "pitch_mean": pitch_mean,
        "pitch_midi": pitch_midi,
        "pitch_std": pitch["pitch_std"],
        "pitch_range": pitch["pitch_range"],
        "pitch_confidence": float(pitch_confidence),
        "pitch_motion": float(pitch_motion),
        "register": register,

        # Spectral / harmonic descriptors
        "harmonicity": float(harmonicity),
        "inharmonicity": float(inharmonicity),
        "spectral_bandwidth": float(spectral_bandwidth),
        "spectral_rolloff": float(spectral_rolloff),
        "spectral_width": float(spectral_width),
        "zero_crossing_rate": float(zcr_mean),

        # Envelope / behaviour descriptors
        "tail_length": float(tail_length),
        "resonance_strength": float(resonance_strength),
        "transient_score": float(transient_score),
        "ambient_score": float(ambient_score),
        "texture_score": float(texture_score),
        "drone_score": float(drone_score),
        "gesture_strength": float(gesture_strength),

        # Compositional descriptors
        "gesture_type": gesture_type,
        "phrase_role": phrase_role,
        "emotional_function": emotional_function,
        "musicality": float(musicality),
        "richness": float(richness),
        "foreground_probability": float(foreground_probability),
        "background_probability": float(background_probability),
        "contrast_score": float(contrast_score),
        "synthetic_score": float(synthetic_score),

        # Learning descriptors, initial values. Critic will update these later.
        "critic_score": 0.5,
        "times_liked": 0,
        "fatigue": 0.0,
        "novelty": float(novelty),
    }


def build_memory():
    memory = []
    total_objects = 0
    failures = []

    files = sorted(os.listdir(MEMORY_FOLDER))

    for filename in files:
        if not filename.lower().endswith(".wav"):
            continue

        path = os.path.join(MEMORY_FOLDER, filename)
        print("Analysing:", filename)

        try:
            audio, sr = load_audio(path)
            objects = detect_sound_objects(audio, sr)
        except Exception as exc:
            failures.append({"recording": filename, "error": f"{type(exc).__name__}: {exc}"})
            print("Failed:", filename, failures[-1]["error"])
            continue

        recording_id = stable_recording_id(audio, sr)

        recording = {
            "schema_version": 2,
            "recording": filename,
            "recording_id": recording_id,
            "sample_rate": int(sr),
            "duration": len(audio) / sr,
            "objects": []
        }

        for idx, (start, end) in enumerate(objects):
            fragment = audio[int(start * sr): int(end * sr)]

            if len(fragment) < sr * MIN_OBJECT:
                continue

            features = analyse_object(fragment, sr)

            object_id = stable_object_id(fragment, sr)
            recording["objects"].append({
                "id": object_id,
                "stable_id": object_id,
                "legacy_id": idx,
                "start": float(start),
                "end": float(end),
                "start_sample": int(round(start * sr)),
                "end_sample": int(round(end * sr)),
                "duration": float(end - start),
                "features": features,
                "times_used": 0
            })

        total_objects += len(recording["objects"])
        memory.append(recording)

    atomic_write_json(OUTPUT_FILE, memory)
    report = {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "target_sample_rate": TARGET_SR,
        "memory_folder": MEMORY_FOLDER,
        "output_file": OUTPUT_FILE,
        "recordings": len(memory),
        "sound_objects": total_objects,
        "failed_recordings": failures,
        "stable_id_scheme": "sha256(canonical mono float32 audio at 48 kHz)",
    }
    atomic_write_json(REPORT_FILE, report)

    print()
    print("Saved:", OUTPUT_FILE)
    print("Build report:", REPORT_FILE)
    print("Recordings:", len(memory))
    print("Sound objects:", total_objects)
    print("Memory v3 descriptors: pitch, register, harmonicity, musicality, richness, phrase roles, learning fields")


if __name__ == "__main__":
    build_memory()

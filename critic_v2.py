# Hyponoia Critic v2
# Internal critic for rendered Hyponoia WAV files.
# It scores the same criteria that the human user will score:
# musicality, coherence, richness, transitions, bloom_quality, overall.

import os
import sys
import json
from datetime import datetime

import librosa
import numpy as np


REPORT_FOLDER = "critic_reports"
RENDER_REPORT_FOLDER = "render_reports"
RENDER_REPORT_FILE = "render_report.json"
TARGET_SR = 44100


def clamp(x, lo=0.0, hi=100.0):
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


def moving_average(x, n=12):
    if len(x) < n:
        return x
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


def find_render_report(audio_path):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    candidate = os.path.join(RENDER_REPORT_FOLDER, f"{base}_render_report.json")
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(RENDER_REPORT_FILE):
        try:
            with open(RENDER_REPORT_FILE, "r") as f:
                latest = json.load(f)
            if os.path.basename(latest.get("audio_file", "")) == os.path.basename(audio_path):
                return RENDER_REPORT_FILE
        except (OSError, json.JSONDecodeError):
            pass
    return None


def analyse_sample_usage(audio_path):
    report_path = find_render_report(audio_path)
    if not report_path:
        return None, {
            "sample_diversity": None,
            "repetition_index": None,
            "exploration_score": None,
        }

    with open(report_path, "r") as f:
        report = json.load(f)

    counts = np.asarray(list(report.get("samples", {}).values()), dtype=np.float64)
    total = float(np.sum(counts))
    unique = int(len(counts))
    if total <= 0 or unique == 0:
        metrics = {
            "sample_diversity": 0.0,
            "repetition_index": 100.0,
            "exploration_score": 0.0,
        }
    else:
        probabilities = counts / total
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        max_entropy = float(np.log(unique)) if unique > 1 else 1.0
        entropy_norm = entropy / max_entropy if unique > 1 else 0.0
        concentration = float(np.max(counts) / total)
        diversity = clamp((0.65 * entropy_norm + 0.35 * min(1.0, unique / max(total, 1.0))) * 100.0)
        repetition = clamp(concentration * 100.0)
        exploration = clamp((unique / max(total, 1.0)) * 100.0)
        metrics = {
            "sample_diversity": round(diversity, 2),
            "repetition_index": round(repetition, 2),
            "exploration_score": round(exploration, 2),
        }

    return report_path, metrics


def analyse_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    duration = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    rms_smooth = moving_average(rms, 16)
    centroid_smooth = moving_average(centroid, 16)

    energy_mean = safe_mean(rms)
    energy_std = safe_std(rms)
    brightness_mean = safe_mean(centroid)
    brightness_std = safe_std(centroid)
    flatness_mean = safe_mean(flatness)
    bandwidth_mean = safe_mean(bandwidth)
    rolloff_mean = safe_mean(rolloff)

    dynamic_breath = clamp((energy_std / (energy_mean + 1e-6)) * 38.0)
    harsh_penalty = clamp(flatness_mean * 260.0)
    overly_flat_penalty = 20.0 if dynamic_breath < 8 else 0.0
    musicality = clamp(58 + dynamic_breath * 0.55 - harsh_penalty * 0.42 - overly_flat_penalty)

    energy_jumps = np.abs(np.diff(rms_smooth))
    spectral_jumps = np.abs(np.diff(centroid_smooth)) / 8000.0
    jump_score = safe_mean(energy_jumps) * 260 + safe_mean(spectral_jumps) * 85
    coherence = clamp(92 - jump_score)

    spectral_motion = safe_std(centroid_smooth) / 2800.0
    richness = clamp(
        48
        + min(1.0, bandwidth_mean / 4500.0) * 24
        + min(1.0, brightness_std / 2200.0) * 20
        + min(1.0, spectral_motion) * 18
        - flatness_mean * 120
    )

    if len(onset_env) > 0:
        onset_strength_mean = safe_mean(onset_env)
        onset_strength_max = float(np.max(onset_env))
    else:
        onset_strength_mean = 0.0
        onset_strength_max = 0.0

    spike_ratio = onset_strength_max / (onset_strength_mean + 1e-6)
    transitions = clamp(90 - max(0.0, spike_ratio - 8.0) * 2.8 - jump_score * 0.20)

    thirds = np.array_split(np.arange(len(rms)), 3)
    if len(thirds) == 3 and all(len(t) > 0 for t in thirds):
        early_bright = safe_mean(centroid[thirds[0]])
        mid_bright = safe_mean(centroid[thirds[1]])
        late_bright = safe_mean(centroid[thirds[2]])

        early_bw = safe_mean(bandwidth[thirds[0]])
        mid_bw = safe_mean(bandwidth[thirds[1]])

        early_energy = safe_mean(rms[thirds[0]])
        mid_energy = safe_mean(rms[thirds[1]])

        bloom_brightness = clamp((mid_bright - early_bright) / 25.0, 0, 30)
        bloom_width = clamp((mid_bw - early_bw) / 35.0, 0, 30)
        bloom_energy = clamp((mid_energy - early_energy) / (early_energy + 1e-6) * 16.0, 0, 25)
        closing_control = clamp(100 - abs(late_bright - mid_bright) / 90.0, 0, 15)

        bloom_quality = clamp(45 + bloom_brightness + bloom_width + bloom_energy + closing_control)
    else:
        bloom_quality = 50.0

    overall = clamp(
        musicality * 0.24
        + coherence * 0.22
        + richness * 0.20
        + transitions * 0.18
        + bloom_quality * 0.16
    )

    render_report_path, sample_metrics = analyse_sample_usage(path)

    comments = []
    if musicality < 65:
        comments.append("Musicality is low: render may feel static, harsh, or insufficiently shaped.")
    if coherence < 70:
        comments.append("Coherence is low: transitions or material relationships may feel disconnected.")
    if richness < 70:
        comments.append("Richness is low: spectrum may need more evolution or layered development.")
    if transitions < 72:
        comments.append("Transitions may be too abrupt.")
    if bloom_quality < 70:
        comments.append("Middle bloom could be stronger or more naturally developed.")
    if overall >= 85:
        comments.append("Strong render: good candidate for performance/testing.")

    return {
        "audio_file": path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": duration,
        "internal_scores": {
            "musicality": round(musicality, 2),
            "coherence": round(coherence, 2),
            "richness": round(richness, 2),
            "transitions": round(transitions, 2),
            "bloom_quality": round(bloom_quality, 2),
            "overall": round(overall, 2),
        },
        "sample_usage": {
            "render_report": render_report_path,
            **sample_metrics,
        },
        "analysis": {
            "energy_mean": float(energy_mean),
            "energy_std": float(energy_std),
            "brightness_mean": float(brightness_mean),
            "brightness_std": float(brightness_std),
            "flatness_mean": float(flatness_mean),
            "bandwidth_mean": float(bandwidth_mean),
            "rolloff_mean": float(rolloff_mean),
            "onset_count": int(len(onset_times)),
            "onsets_per_minute": float(len(onset_times) / max(duration / 60.0, 1e-6)),
        },
        "comments": comments,
    }


def save_report(report):
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    base = os.path.splitext(os.path.basename(report["audio_file"]))[0]
    out = os.path.join(REPORT_FOLDER, f"{base}_critic.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=4)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 critic_v2.py path/to/render.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print("File not found:", audio_path)
        sys.exit(1)

    report = analyse_audio(audio_path)
    out = save_report(report)

    print()
    print("Internal Critic Scores")
    print("----------------------")
    for k, v in report["internal_scores"].items():
        print(f"{k}: {v}")

    print()
    print("Comments:")
    for c in report["comments"]:
        print("-", c)

    print()
    print("Saved critic report:")
    print(out)


if __name__ == "__main__":
    main()

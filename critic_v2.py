# Hyponoia Critic v2
# Internal critic for rendered Hyponoia WAV files.
# It scores the same criteria that the human user will score:
# musicality, coherence, richness, transitions, bloom_quality, overall.

import os
import sys
import json
import glob

import librosa
import numpy as np

from hyponoia_stability import TARGET_SR, atomic_write_json, utc_timestamp


REPORT_FOLDER = "critic_reports"
RENDER_REPORT_FOLDER = "render_reports"
RENDER_REPORT_FILE = "render_report.json"
CRITIC_VERSION = "2.1"
def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))


def score_from_unit(value, lo=3.0, hi=97.0):
    """Map a continuous 0..1 quality estimate without creating 0/100 ceilings."""
    return clamp(lo + (hi - lo) * max(0.0, min(1.0, float(value))), lo, hi)


def sigmoid(value):
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + np.exp(-value))


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


def _metric_unit(metrics, name, default=0.55):
    value = metrics.get(name)
    if value is None:
        return float(default)
    return max(0.0, min(1.0, float(value) / 100.0))


def temporal_arc_metrics(rms, centroid, segments=12):
    """Describe macro-form motion separately from short-term discontinuity."""
    if len(rms) < segments or len(centroid) < segments:
        return {
            "arc_smoothness": 0.5,
            "arc_motion": 0.5,
            "energy_curve": [],
            "brightness_curve": [],
        }

    rms_chunks = np.array([safe_mean(chunk) for chunk in np.array_split(rms, segments)])
    centroid_chunks = np.array([safe_mean(chunk) for chunk in np.array_split(centroid, segments)])

    def normalise(values, floor):
        centre = safe_mean(values)
        return (values - centre) / (abs(centre) + floor)

    energy_curve = normalise(rms_chunks, 1e-6)
    brightness_curve = normalise(centroid_chunks, 250.0)
    curvature = 0.55 * safe_mean(np.abs(np.diff(energy_curve, n=2))) + 0.45 * safe_mean(
        np.abs(np.diff(brightness_curve, n=2))
    )
    arc_smoothness = float(np.exp(-1.8 * curvature))
    motion = 0.58 * safe_std(energy_curve) + 0.42 * safe_std(brightness_curve)
    # Some motion is necessary for development, but extreme instability is not.
    arc_motion = float(np.exp(-((motion - 0.24) / 0.26) ** 2))
    return {
        "arc_smoothness": max(0.0, min(1.0, arc_smoothness)),
        "arc_motion": max(0.0, min(1.0, arc_motion)),
        "energy_curve": [round(float(value), 6) for value in energy_curve],
        "brightness_curve": [round(float(value), 6) for value in brightness_curve],
    }


def _render_structure(report):
    roles = ("gesture", "texture", "resonance", "noise", "impact")
    sections = ("opening", "activation", "complexity", "memory", "resolution")
    role_counts = report.get("role_counts", {})
    role_values = np.asarray([max(0.0, float(role_counts.get(role, 0.0))) for role in roles])
    if role_values.sum() <= 0:
        return None
    role_values /= role_values.sum()

    section_counts = {section: 0.0 for section in sections}
    for detail in report.get("sample_usage_details", {}).values():
        for section, count in detail.get("section_counts", {}).items():
            if section in section_counts:
                section_counts[section] += max(0.0, float(count))
    section_values = np.asarray([section_counts[section] for section in sections])
    if section_values.sum() <= 0:
        section_values = np.full(len(sections), 1.0 / len(sections))
    else:
        section_values /= section_values.sum()
    return role_values, section_values


def historical_structure_novelty(render_report_path):
    """Compare role/form proportions with earlier renders at the same D-level."""
    if not render_report_path or not os.path.exists(render_report_path):
        return {
            "score": None,
            "prior_renders": 0,
            "closest_report": None,
            "closest_distance": None,
        }
    try:
        with open(render_report_path, "r", encoding="utf-8") as handle:
            current = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {
            "score": None,
            "prior_renders": 0,
            "closest_report": None,
            "closest_distance": None,
        }

    current_structure = _render_structure(current)
    if current_structure is None:
        return {
            "score": None,
            "prior_renders": 0,
            "closest_report": None,
            "closest_distance": None,
        }
    current_timestamp = str(current.get("timestamp", ""))
    current_audio = os.path.basename(str(current.get("audio_file", "")))
    candidates = []
    pattern = os.path.join(os.path.dirname(render_report_path) or ".", "*_render_report.json")
    for candidate_path in glob.glob(pattern):
        if os.path.abspath(candidate_path) == os.path.abspath(render_report_path):
            continue
        try:
            with open(candidate_path, "r", encoding="utf-8") as handle:
                candidate = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if candidate.get("dream_level") != current.get("dream_level"):
            continue
        if os.path.basename(str(candidate.get("audio_file", ""))) == current_audio:
            continue
        candidate_timestamp = str(candidate.get("timestamp", ""))
        if current_timestamp and candidate_timestamp and candidate_timestamp >= current_timestamp:
            continue
        candidate_structure = _render_structure(candidate)
        if candidate_structure is None:
            continue
        role_distance = 0.5 * float(np.abs(current_structure[0] - candidate_structure[0]).sum())
        section_distance = 0.5 * float(np.abs(current_structure[1] - candidate_structure[1]).sum())
        distance = 0.65 * role_distance + 0.35 * section_distance
        candidates.append((distance, candidate_path))

    if not candidates:
        return {
            "score": None,
            "prior_renders": 0,
            "closest_report": None,
            "closest_distance": None,
        }
    distance, closest = min(candidates, key=lambda item: item[0])
    # A 0.20 total-variation distance is treated as clearly distinct structure.
    score = max(0.0, min(100.0, distance / 0.20 * 100.0))
    return {
        "score": round(score, 2),
        "prior_renders": len(candidates),
        "closest_report": closest,
        "closest_distance": round(distance, 6),
    }


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
            "exposure_diversity": None,
            "exposure_concentration": None,
            "role_diversity": None,
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
            "exposure_diversity": 0.0,
            "exposure_concentration": 100.0,
            "role_diversity": 0.0,
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

        details = report.get("sample_usage_details", {})
        exposure = np.asarray(
            [max(0.0, float(details.get(key, {}).get("exposure_sec", 0.0))) for key in report.get("samples", {})],
            dtype=np.float64,
        )
        if exposure.sum() > 0:
            exposure_p = exposure / exposure.sum()
            exposure_entropy = -float(np.sum(exposure_p * np.log(exposure_p + 1e-12)))
            exposure_max_entropy = float(np.log(len(exposure_p))) if len(exposure_p) > 1 else 1.0
            metrics["exposure_diversity"] = round(
                clamp((exposure_entropy / exposure_max_entropy if len(exposure_p) > 1 else 0.0) * 100.0), 2
            )
            metrics["exposure_concentration"] = round(clamp(float(exposure_p.max()) * 100.0), 2)
        else:
            metrics["exposure_diversity"] = None
            metrics["exposure_concentration"] = None

        roles = report.get("role_counts", {})
        role_values = np.asarray([max(0.0, float(value)) for value in roles.values()], dtype=np.float64)
        if role_values.sum() > 0:
            role_p = role_values / role_values.sum()
            role_entropy = -float(np.sum(role_p * np.log(role_p + 1e-12)))
            role_max_entropy = float(np.log(len(role_p))) if len(role_p) > 1 else 1.0
            metrics["role_diversity"] = round(
                clamp((role_entropy / role_max_entropy if len(role_p) > 1 else 0.0) * 100.0), 2
            )
        else:
            metrics["role_diversity"] = None

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

    energy_jumps = np.abs(np.diff(rms_smooth))
    spectral_jumps = np.abs(np.diff(centroid_smooth))
    dynamic_ratio = energy_std / (energy_mean + 1e-6)
    dynamic_quality = np.tanh(dynamic_ratio * 1.55)
    tonal_balance = np.exp(-max(0.0, flatness_mean - 0.012) * 11.0)
    activity_quality = np.tanh(safe_mean(onset_env) / 1.4)
    musicality_unit = 0.12 + 0.36 * dynamic_quality + 0.34 * tonal_balance + 0.18 * activity_quality
    musicality = score_from_unit(musicality_unit)

    energy_step = float(np.percentile(energy_jumps, 75)) / (energy_mean + 1e-6) if len(energy_jumps) else 0.0
    spectral_step = float(np.percentile(spectral_jumps, 75)) / (brightness_mean + 250.0) if len(spectral_jumps) else 0.0
    discontinuity = 0.68 * energy_step + 0.32 * spectral_step
    # Kept as a diagnostic. v2.1 no longer treats absence of jumps as the
    # complete definition of coherence.
    continuity_quality = sigmoid((0.030 - discontinuity) / 0.011)

    spectral_motion = safe_std(centroid_smooth) / (brightness_mean + 300.0)
    richness_unit = (
        0.10
        + 0.32 * np.tanh(bandwidth_mean / 3800.0)
        + 0.24 * np.tanh(brightness_std / 1900.0)
        + 0.24 * np.tanh(spectral_motion * 2.2)
        + 0.10 * tonal_balance
    )
    richness = score_from_unit(richness_unit)

    if len(onset_env) > 0:
        onset_strength_mean = safe_mean(onset_env)
        onset_strength_max = float(np.max(onset_env))
    else:
        onset_strength_mean = 0.0
        onset_strength_max = 0.0

    spike_ratio = onset_strength_max / (onset_strength_mean + 1e-6)
    spike_excess = max(0.0, np.log1p(spike_ratio) - np.log(5.0))
    transitions_unit = np.exp(-0.65 * spike_excess - 1.45 * discontinuity)
    transitions = score_from_unit(transitions_unit)

    thirds = np.array_split(np.arange(len(rms)), 3)
    if len(thirds) == 3 and all(len(t) > 0 for t in thirds):
        early_bright = safe_mean(centroid[thirds[0]])
        mid_bright = safe_mean(centroid[thirds[1]])
        late_bright = safe_mean(centroid[thirds[2]])

        early_bw = safe_mean(bandwidth[thirds[0]])
        mid_bw = safe_mean(bandwidth[thirds[1]])

        early_energy = safe_mean(rms[thirds[0]])
        mid_energy = safe_mean(rms[thirds[1]])

        brightness_growth = (mid_bright - early_bright) / (abs(early_bright) + 250.0)
        width_growth = (mid_bw - early_bw) / (abs(early_bw) + 250.0)
        energy_growth = (mid_energy - early_energy) / (abs(early_energy) + 1e-6)
        brightness_settle = (mid_bright - late_bright) / (abs(mid_bright) + 250.0)
        growth = 0.36 * brightness_growth + 0.28 * width_growth + 0.36 * energy_growth
        growth_quality = sigmoid((growth - 0.04) / 0.12)
        settle_quality = sigmoid((brightness_settle + 0.02) / 0.10)
        bloom_unit = 0.12 + 0.58 * growth_quality + 0.30 * settle_quality
        bloom_quality = score_from_unit(bloom_unit)
    else:
        growth = 0.0
        brightness_settle = 0.0
        growth_quality = 0.35
        settle_quality = 0.35
        bloom_unit = 0.35
        bloom_quality = score_from_unit(0.35)

    render_report_path, sample_metrics = analyse_sample_usage(path)
    arc = temporal_arc_metrics(rms, centroid)
    development_quality = 0.62 * growth_quality + 0.38 * settle_quality
    material_quality = (
        0.30 * _metric_unit(sample_metrics, "sample_diversity")
        + 0.24 * _metric_unit(sample_metrics, "exposure_diversity")
        + 0.28 * _metric_unit(sample_metrics, "role_diversity")
        + 0.18 * (1.0 - _metric_unit(sample_metrics, "repetition_index", 0.20))
    )

    # v2.1: coherence means smooth local continuity plus an intentional arc and
    # related material. This removes the 92-94 ceiling observed in real ratings.
    coherence_unit = (
        0.46 * continuity_quality
        + 0.21 * arc["arc_smoothness"]
        + 0.18 * development_quality
        + 0.15 * material_quality
    )
    coherence = score_from_unit(coherence_unit)

    acoustic_musicality_unit = (musicality - 3.0) / 94.0
    musicality_unit_v21 = (
        0.55 * acoustic_musicality_unit
        + 0.20 * development_quality
        + 0.15 * material_quality
        + 0.10 * arc["arc_motion"]
    )
    musicality = score_from_unit(musicality_unit_v21)

    acoustic_richness_unit = (richness - 3.0) / 94.0
    richness_unit_v21 = (
        0.45 * acoustic_richness_unit
        + 0.30 * development_quality
        + 0.25 * material_quality
    )
    richness = score_from_unit(richness_unit_v21)

    structural_history = historical_structure_novelty(render_report_path)
    novelty_score = structural_history.get("score")
    novelty_penalty = 0.0 if novelty_score is None else max(0.0, 40.0 - float(novelty_score)) * 0.08

    overall = clamp(
        musicality * 0.24
        + coherence * 0.22
        + richness * 0.20
        + transitions * 0.18
        + bloom_quality * 0.16
        - novelty_penalty
    )

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
    if novelty_score is not None and novelty_score < 35:
        comments.append(
            "Structural novelty is low: the role balance and formal activity resemble an earlier render at this D-level."
        )
    if overall >= 85:
        comments.append("Strong render: good candidate for performance/testing.")

    return {
        "schema_version": 3,
        "critic_version": CRITIC_VERSION,
        "audio_file": path,
        "timestamp": utc_timestamp(),
        "duration_sec": duration,
        "analysis_sample_rate": int(sr),
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
        "structural_history": structural_history,
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
            "dynamic_ratio": float(dynamic_ratio),
            "energy_step": float(energy_step),
            "spectral_step": float(spectral_step),
            "discontinuity": float(discontinuity),
            "continuity_quality": float(continuity_quality),
            "development_quality": float(development_quality),
            "material_quality": float(material_quality),
            "arc_smoothness": float(arc["arc_smoothness"]),
            "arc_motion": float(arc["arc_motion"]),
            "energy_curve": arc["energy_curve"],
            "brightness_curve": arc["brightness_curve"],
            "onset_spike_ratio": float(spike_ratio),
            "bloom_growth": float(growth),
            "bloom_settle": float(brightness_settle),
        },
        "comments": comments,
    }


def save_report(report):
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    base = os.path.splitext(os.path.basename(report["audio_file"]))[0]
    out = os.path.join(REPORT_FOLDER, f"{base}_critic.json")
    atomic_write_json(out, report)
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

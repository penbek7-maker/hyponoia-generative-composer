# Hyponoia AI Composer v3 + Memory v3 — richer musical pass
# More active D5, phrase evolution, musical delay/reverb, and less over-soft selection.
import os
import json
import random
from pathlib import Path
from functools import lru_cache
from datetime import datetime

import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

from hyponoia_stability import (
    TARGET_SR,
    atomic_write_json,
    deterministic_group,
    migrate_sample_profile,
    sample_key as stable_sample_key,
    utc_timestamp,
)
from composition_preference_v1 import CompositionPreferenceAssist
from artist_style_v1 import ArtistStyleAssist
from library_coverage_v1 import (
    recording_coverage_factor,
    sync_recording_coverage,
    update_recording_coverage,
)
from representation_assist_v1 import RepresentationAssist
from learning_profile_store_v1 import load_active_learning_profile

USER_CONFIG_FILE = os.environ.get("HYPNOIA_USER_CONFIG", "hyponoia_user_config.json")


def load_user_paths(config_path=USER_CONFIG_FILE):
    """Load paths saved by Update Library; environment variables still win."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


USER_PATHS = load_user_paths()
MEMORY_FILE = os.environ.get(
    "HYPNOIA_MEMORY_FILE", USER_PATHS.get("memory_file", "memory_index_v3.json")
)
MEMORY_FOLDER = os.environ.get(
    "HYPNOIA_MEMORY_FOLDER", USER_PATHS.get("memory_folder", "alpha_memory")
)
OUTPUT_FOLDER = os.environ.get("HYPNOIA_OUTPUT_FOLDER", "output")
PROFILE_FILE = os.environ.get("HYPNOIA_PROFILE_FILE", "alpha_profile.json")
LEARNING_FILE = os.environ.get("HYPNOIA_LEARNING_FILE", "learning_profile.json")
SAMPLE_LEARNING_FILE = os.environ.get(
    "HYPNOIA_SAMPLE_LEARNING_FILE", "sample_learning_profile.json"
)
RENDER_REPORT_FILE = os.environ.get("HYPNOIA_RENDER_REPORT_FILE", "render_report.json")
RENDER_REPORT_FOLDER = os.environ.get("HYPNOIA_RENDER_REPORT_FOLDER", "render_reports")
GENERATOR_REVISION = "2026-09-11-anchor-morphology-learned-pan-24"
REPRESENTATION_CONFIG_FILE = os.environ.get(
    "HYPNOIA_REPRESENTATION_CONFIG",
    USER_PATHS.get("representation_config", "representation_config.json"),
)
REPRESENTATION_ASSIST = RepresentationAssist.disabled()
REPRESENTATION_ASSIST_ROLES = frozenset({"gesture", "texture", "impact", "noise"})
COMPOSITION_PREFERENCE_FILE = os.environ.get(
    "HYPNOIA_COMPOSITION_PREFERENCE",
    USER_PATHS.get(
        "composition_preference",
        "phase2_artifacts/composition_preference_gold.json",
    ),
)
COMPOSITION_PREFERENCE = CompositionPreferenceAssist.disabled()
ARTIST_STYLE_FILE = os.environ.get(
    "HYPNOIA_ARTIST_STYLE_PROFILE",
    USER_PATHS.get(
        "artist_style_profile",
        "phase2_artifacts/artist_style_baseline_v1.json",
    ),
)
ARTIST_STYLE = ArtistStyleAssist.disabled()
CURRENT_LIBRARY_COVERAGE_SNAPSHOT = {}
LEARNED_SYNTH_AFFINITY = {}
ORIGIN_LEARNING_SNAPSHOT = {
    "active": False,
    "synthetic_anchor_objects": 0,
    "contrast_anchor_objects": 0,
    "inferred_object_count": 0,
    "high_confidence_synthetic_objects": 0,
}
LIBRARY_LABELS_FILE = os.environ.get(
    "HYPNOIA_LIBRARY_LABELS",
    USER_PATHS.get("library_labels", "library_source_labels.json"),
)


def load_library_source_labels(path=LIBRARY_LABELS_FILE):
    """Load optional user-correctable natural/synthetic source labels."""
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = Path(__file__).resolve().parent / source
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        recordings = payload.get("recordings", {}) if isinstance(payload, dict) else {}
        return {
            str(name): str(label).strip().lower()
            for name, label in recordings.items()
            if str(label).strip().lower()
            in {"natural", "synthetic", "hybrid", "instrument_hybrid", "unknown"}
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


LIBRARY_SOURCE_LABELS = load_library_source_labels()


def build_embedding_origin_model(objects, representation_assist, labels=None):
    """Learn synth affinity from human-labelled recordings in embedding space.

    Explicit recording labels remain authoritative. Unlabelled sound objects
    receive a bounded probability-like affinity by comparing their nearest
    confirmed synth neighbour with their nearest confirmed non-synth neighbour.
    """
    labels = LIBRARY_SOURCE_LABELS if labels is None else labels
    disabled = {
        "active": False,
        "synthetic_anchor_objects": 0,
        "contrast_anchor_objects": 0,
        "inferred_object_count": 0,
        "high_confidence_synthetic_objects": 0,
    }
    if not getattr(representation_assist, "active", False):
        return {}, disabled
    embeddings = getattr(representation_assist, "embeddings", {})
    positive = []
    contrast = []
    for obj in objects:
        vector = embeddings.get(str(obj.get("object_id", "")))
        if vector is None:
            continue
        origin = labels.get(str(obj.get("recording", "")), "unknown")
        if origin == "synthetic":
            positive.append(np.asarray(vector, dtype=np.float64))
        elif origin in {"natural", "hybrid", "instrument_hybrid"}:
            contrast.append(np.asarray(vector, dtype=np.float64))
    if len(positive) < 2 or len(contrast) < 2:
        return {}, disabled

    positive_matrix = np.vstack(positive)
    contrast_matrix = np.vstack(contrast)
    affinities = {}
    high_confidence = 0
    for obj in objects:
        if labels.get(str(obj.get("recording", "")), "unknown") != "unknown":
            continue
        stable_id = str(obj.get("object_id", ""))
        vector = embeddings.get(stable_id)
        if vector is None:
            continue
        vector = np.asarray(vector, dtype=np.float64)
        positive_similarity = float(np.max(positive_matrix @ vector))
        contrast_similarity = float(np.max(contrast_matrix @ vector))
        if positive_similarity < 0.58:
            affinity = 0.50
        else:
            evidence = float(np.clip((positive_similarity - 0.55) / 0.35, 0.0, 1.0))
            margin = positive_similarity - contrast_similarity
            affinity = float(np.clip(0.50 + 1.65 * margin * evidence, 0.08, 0.92))
        affinities[stable_id] = affinity
        if affinity >= 0.67:
            high_confidence += 1
    snapshot = {
        "active": True,
        "synthetic_anchor_objects": len(positive),
        "contrast_anchor_objects": len(contrast),
        "inferred_object_count": len(affinities),
        "high_confidence_synthetic_objects": high_confidence,
    }
    return affinities, snapshot


def source_origin(obj):
    return LIBRARY_SOURCE_LABELS.get(str(obj.get("recording", "")), "unknown")


def effective_synthetic_score(obj):
    """Respect human source-origin labels before the acoustic synth estimate."""
    estimated = max(
        0.0,
        min(1.0, float(obj.get("features", {}).get("synthetic_score", 0.5))),
    )
    origin = source_origin(obj)
    if origin == "natural":
        return min(0.18, estimated * 0.22)
    if origin == "synthetic":
        return max(0.82, estimated)
    if origin in {"hybrid", "instrument_hybrid"}:
        return 0.50 + (estimated - 0.50) * 0.55
    inferred = LEARNED_SYNTH_AFFINITY.get(str(obj.get("object_id", "")))
    if inferred is not None:
        # Acoustic evidence still contributes, while the learned neighbourhood
        # is strong enough to generalise the listener's confirmed examples.
        return float(np.clip(0.45 * estimated + 0.55 * inferred, 0.05, 0.95))
    return estimated


def fragile_air_object(obj):
    """Detect near-silent ultrasonic/noise slices unsuitable as long beds.

    This is not a high-frequency ban. Bright material remains eligible; only
    the conjunction of extremely low source energy, very high centroid, dense
    zero crossings and low musicality marks a slice as fragile air.
    """
    features = obj.get("features", {})
    return bool(
        float(features.get("energy", 1.0) or 0.0) < 0.00020
        and float(features.get("brightness", 0.0) or 0.0) > 8500.0
        and float(features.get("zero_crossing_rate", 0.0) or 0.0) > 0.22
        and float(features.get("musicality", 1.0) or 0.0) < 0.30
    )


def bound_fragile_air_punctuation(frag, obj, dream_level):
    """Keep fragile air as a short event instead of an amplified constant bed."""
    source = np.asarray(frag, dtype=np.float32)
    if not fragile_air_object(obj):
        return source
    maximum = int({1: 3.2, 3: 4.4, 5: 5.8}[int(dream_level)] * TARGET_SR)
    if len(source) > maximum:
        start = max(0, (len(source) - maximum) // 2)
        source = source[start : start + maximum].copy()
    return fade(source, sec=min(0.55, len(source) / TARGET_SR * 0.18))

D5_REFERENCE_TARGETS = {
    "pulse_bpm_range": [122.0, 129.0],
    "target_integrated_lufs": -17.4,
    "max_abrupt_drop_rate_per_minute": 0.5,
    "design": "continuous transformations, developed synthetic material, no hard phrase cuts",
}

DEFAULT_LEARNING_WEIGHTS = {
    "musicality_weight": 1.0,
    "coherence_weight": 1.0,
    "richness_weight": 1.0,
    "transition_smoothness_weight": 1.0,
    "bloom_weight": 1.0,
    "ambient_weight": 1.0,
    "gesture_weight": 1.0,
    "noise_penalty": 1.0,
    "impact_penalty": 1.0,
    "exploration_weight": 1.0,
    "repetition_control": 1.0,
    "synthetic_material_weight": 1.0,
    "instrument_material_weight": 1.0,
    "foreground_presence_weight": 1.0,
    "arpeggio_weight": 1.0,
    "long_layer_diversity_weight": 1.0,
    "activity_weight": 1.0,
    "material_development_weight": 1.0,
    "low_frequency_control": 1.0,
    "layer_clarity_weight": 1.0,
    "structured_granulation_weight": 1.0,
}

LEARNING_WEIGHTS = dict(DEFAULT_LEARNING_WEIGHTS)


# Harmonic state supplied by Max/MSP or command-line arguments.
# Confidence below 0.55 keeps the composer in free/non-tonal mode.
HARMONY_STATE = {
    "root": 0,
    "scale": "free",
    "confidence": 0.0,
}

NOTE_NAMES = {
    "c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3,
    "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7,
    "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11,
}

SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "chromatic": tuple(range(12)),
    "free": tuple(range(12)),
}


def configure_harmony(root=0, scale="free", confidence=0.0):
    """Set a safe harmonic state. Unknown scales fall back to free mode."""
    try:
        if isinstance(root, str) and root.strip().lower() in NOTE_NAMES:
            root_pc = NOTE_NAMES[root.strip().lower()]
        else:
            root_pc = int(round(float(root))) % 12
    except (TypeError, ValueError):
        root_pc = 0

    scale_name = str(scale).strip().lower().replace(" ", "_").replace("-", "_")
    if scale_name not in SCALE_INTERVALS:
        print(f"Unknown scale '{scale}'; using free mode.")
        scale_name = "free"

    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.0

    if conf < 0.55:
        scale_name = "free"

    HARMONY_STATE.update({
        "root": root_pc,
        "scale": scale_name,
        "confidence": conf,
    })

    print("Harmony state:")
    print(f"  root pitch class: {root_pc}")
    print(f"  scale: {scale_name}")
    print(f"  confidence: {conf:.3f}")


def active_pitch_classes():
    root = HARMONY_STATE["root"]
    intervals = SCALE_INTERVALS.get(HARMONY_STATE["scale"], SCALE_INTERVALS["free"])
    return {(root + interval) % 12 for interval in intervals}


def scale_selection_factor(obj):
    """Lower is better. Prefer reliable pitched objects that belong to the active scale."""
    if HARMONY_STATE["scale"] == "free":
        return 1.0

    pitch = obj.get("features", {}).get("pitch_midi")
    if pitch is None:
        return 1.0

    try:
        pitch_value = float(pitch)
        if not np.isfinite(pitch_value):
            return 1.0
    except (TypeError, ValueError):
        return 1.0

    f = obj.get("features", {})
    harmonicity = float(f.get("harmonicity", 0.5))
    confidence = HARMONY_STATE["confidence"]

    # Do not over-constrain noisy or weakly pitched material.
    strength = max(0.0, min(1.0, confidence * (0.35 + 0.65 * harmonicity)))
    pc = int(round(pitch_value)) % 12

    if pc in active_pitch_classes():
        return 1.0 - 0.30 * strength

    allowed = active_pitch_classes()
    distance = min(min((pc - a) % 12, (a - pc) % 12) for a in allowed)
    return 1.0 + (0.10 + 0.08 * distance) * strength


def midi_to_hz(midi_note):
    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def scale_frequencies(low_midi=48, high_midi=96, count=10):
    """Generate bloom/bed frequencies from the current root and mode."""
    if HARMONY_STATE["scale"] == "free":
        return None

    pcs = active_pitch_classes()
    notes = [m for m in range(low_midi, high_midi + 1) if m % 12 in pcs]
    if not notes:
        return None

    indices = np.linspace(0, len(notes) - 1, min(count, len(notes))).round().astype(int)
    return [midi_to_hz(notes[i]) for i in indices]

OUTPUT_DURATION = 180
STEM_LOW_CROSSOVER_HZ = 250
STEM_HIGH_CROSSOVER_HZ = 4000
# Material-plan sizes: D1 / D3 / D5
CORE_COUNT = 6
EXTRA_D3_COUNT = 4
EXTRA_D5_COUNT = 6

# New renders vary by default. An explicit seed supports fair A/B listening,
# where the learned feedback must be the only intended source of change.
try:
    RENDER_SEED = int(os.environ["HYPNOIA_RENDER_SEED"])
except (KeyError, TypeError, ValueError):
    RENDER_SEED = (
        datetime.now().microsecond
        + os.getpid()
        + random.SystemRandom().randint(0, 999999)
    )
random.seed(RENDER_SEED)
np.random.seed(RENDER_SEED % (2**32 - 1))


def load_profile():
    with open(PROFILE_FILE, "r") as f:
        return json.load(f)



def load_learning_weights(dream_level=None):
    """Combine shared rating controls with only the active D-level text profile."""
    weights = dict(DEFAULT_LEARNING_WEIGHTS)
    try:
        data = load_active_learning_profile(LEARNING_FILE)
        incoming = data.get("weights", {})
        level_name = f"D{int(dream_level)}" if dream_level in (1, 3, 5) else None
        level_incoming = data.get("level_weights", {}).get(level_name, {}) if level_name else {}
        for key in weights:
            try:
                value = float(incoming.get(key, 1.0)) + float(level_incoming.get(key, 1.0)) - 1.0
                weights[key] = float(max(0.5, min(1.8, float(value))))
            except (TypeError, ValueError):
                pass
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Could not read learning profile; using neutral weights:", exc)

    print("Learning weights:")
    for key, value in weights.items():
        print(f"  {key}: {value:.4f}")
    return weights


def learned_factor(name, sensitivity=5.0, lo=0.65, hi=1.45):
    """Convert small changes around 1.0 into audible but bounded control factors."""
    value = LEARNING_WEIGHTS.get(name, 1.0)
    factor = float(np.exp((value - 1.0) * sensitivity))
    return max(lo, min(hi, factor))


def composition_feedback_audio_snapshot():
    """Expose the bounded audio consequences of the active D-level feedback."""
    low_control = learned_factor("low_frequency_control", 3.0, 0.72, 1.55)
    clarity = learned_factor("layer_clarity_weight", 3.0, 0.75, 1.50)
    synth = learned_factor("synthetic_material_weight", 4.0, 0.78, 1.60)
    instrument = learned_factor("instrument_material_weight", 3.2, 0.82, 1.55)
    foreground = learned_factor("foreground_presence_weight", 2.8, 0.88, 1.45)
    arpeggio = learned_factor("arpeggio_weight", 4.0, 0.78, 1.65)
    long_layer_diversity = learned_factor(
        "long_layer_diversity_weight", 3.0, 0.82, 1.50
    )
    development = learned_factor("material_development_weight", 2.8, 0.80, 1.55)
    activity = learned_factor("activity_weight", 1.3, 0.92, 1.18)
    release = learned_factor("transition_smoothness_weight", 3.0, 0.88, 1.50)
    granulation = learned_factor("structured_granulation_weight", 2.2, 0.82, 2.40)
    return {
        "low_frequency_gain": float(1.0 / low_control),
        "low_mid_masking_gain": float(1.0 / max(1.0, clarity)),
        "stereo_width": float(max(0.82, min(1.24, 1.0 + 0.28 * (clarity - 1.0)))),
        "reverb_clarity_gain": float(1.0 / np.sqrt(max(0.75, clarity))),
        "synthetic_layer_gain": float(max(0.0, synth - 1.0)),
        "instrument_presence_gain": float(instrument),
        "foreground_presence_gain": float(foreground),
        "arpeggio_layer_gain": float(max(0.0, arpeggio - 1.0)),
        "long_layer_rotation": float(long_layer_diversity),
        "development_drive": float(development),
        "event_activity_gain": float(activity),
        "release_smoothness": float(release),
        "release_tail_strength": float(max(0.0, release - 1.0)),
        "related_layer_overlap_strength": float(max(0.0, release - 1.0)),
        "structured_granulation_drive": float(granulation),
        "structured_granulation_wet": float(max(0.0, min(0.78, (granulation - 1.0) * 0.82))),
    }


def apply_mix_feedback_controls(output):
    """Reduce low masking and separate layers without changing render duration."""
    snapshot = composition_feedback_audio_snapshot()
    low_gain = snapshot["low_frequency_gain"]
    low_mid_gain = snapshot["low_mid_masking_gain"]
    processed = np.asarray(output, dtype=np.float32).copy()

    for channel in range(processed.shape[1]):
        low = butter_filter(processed[:, channel], "lowpass", 170)
        low_mid = butter_filter(processed[:, channel], "lowpass", 520) - low
        processed[:, channel] += low * (low_gain - 1.0)
        processed[:, channel] += low_mid * (low_mid_gain - 1.0) * 0.62

    mid = (processed[:, 0] + processed[:, 1]) * 0.5
    side = (processed[:, 0] - processed[:, 1]) * 0.5 * snapshot["stereo_width"]
    processed[:, 0] = mid + side
    processed[:, 1] = mid - side
    return processed.astype(np.float32)


def adaptive_low_balance(output, dream_level, crossover=250.0):
    """Tame only severe bass dominance while preserving intentional low material.

    The control is inactive for already balanced renders. It never creates or
    removes high-frequency content and does not impose a fixed tonal profile.
    """
    source = np.asarray(output, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 2 or len(source) < 64:
        return source.copy()
    result = source.copy()
    ratio_target = {1: 0.92, 3: 1.05, 5: 1.35}[int(dream_level)]
    for channel in range(2):
        low = butter_filter(source[:, channel], "lowpass", crossover)
        remainder = source[:, channel] - low
        low_rms = float(np.sqrt(np.mean(low * low)))
        remainder_rms = float(np.sqrt(np.mean(remainder * remainder))) + 1e-9
        ratio = low_rms / remainder_rms
        if ratio > ratio_target:
            scale = max(0.08, ratio_target / ratio)
            result[:, channel] = remainder + low * scale
    return result.astype(np.float32)


def dream_activity_multiplier(dream_level):
    """Keep D-level activity audibly ordered without turning D5 into clutter."""
    base = {1: 0.96, 3: 1.04, 5: 1.10}[dream_level]
    if dream_level == 5:
        # D5 energy now comes primarily from internal motion and phrase rhythm,
        # not from accumulating an ever larger number of unrelated layers.
        return 1.06 * learned_factor("activity_weight", 1.3, 0.92, 1.16)
    return base * learned_factor("activity_weight", 3.0, 0.88, 1.22)


def d5_energy_drive(dream_level):
    """Bounded D5 drive from explicit activity and musicality preferences."""
    if dream_level != 5:
        return 1.0
    activity = learned_factor("activity_weight", 3.0, 0.90, 1.30)
    musicality = learned_factor("musicality_weight", 3.0, 0.92, 1.20)
    return float(np.sqrt(activity * musicality))


def d5_temporal_profile(dream_level):
    """Translate level-specific activity feedback into audible forward motion.

    The historical name is retained for report compatibility. Neutral D1/D3
    remain unchanged, while an explicit request for energy makes their phrases,
    envelopes and echoes move sooner. D5 keeps a small inherent forward bias.
    """
    activity_value = float(LEARNING_WEIGHTS.get("activity_weight", 1.0))
    request = max(0.0, activity_value - 1.0)
    if dream_level != 5 and request <= 1e-9:
        return {
            "temporal_drive": 1.0,
            "stretch_scale": 1.0,
            "envelope_scale": 1.0,
            "delay_scale": 1.0,
            "ambient_scale": 1.0,
        }

    if dream_level == 5:
        activity = learned_factor("activity_weight", 1.15, 0.96, 1.42)
        temporal_drive = float(max(1.04, min(1.32, 1.03 * activity)))
    else:
        sensitivity = {1: 1.75, 3: 1.95}[dream_level]
        temporal_drive = float(max(1.0, min(1.28, np.exp(request * sensitivity))))
    return {
        "temporal_drive": temporal_drive,
        "stretch_scale": max(0.90, 1.0 - 0.28 * (temporal_drive - 1.0)),
        "envelope_scale": max(0.92, 1.0 - 0.20 * (temporal_drive - 1.0)),
        "delay_scale": max(0.80, 1.0 - 0.42 * (temporal_drive - 1.0)),
        "ambient_scale": max(0.90, 1.0 - 0.24 * (temporal_drive - 1.0)),
    }


def d5_development_drive(dream_level):
    """Bounded development strength for transformations of the chosen palette."""
    if dream_level != 5:
        return 1.0
    development = learned_factor("material_development_weight", 2.4, 0.92, 1.42)
    synthetic = learned_factor("synthetic_material_weight", 1.4, 0.94, 1.25)
    return float(np.sqrt(development * synthetic))


def d5_development_curve(length, section_name):
    """Create a learned, non-static energy trajectory inside each D5 phrase.

    The five curves describe musical direction, while listener feedback decides
    how strongly they alter the audio. At neutral feedback the curve is flat.
    This lets D5 learn development without borrowing D1/D3 source recordings.
    """
    length = max(0, int(length))
    if length == 0:
        return np.zeros(0, dtype=np.float32)
    anchors = {
        "opening": ((0.0, 0.48, 1.0), (0.72, 0.88, 1.05)),
        "activation": ((0.0, 0.32, 0.68, 1.0), (0.76, 1.10, 0.90, 1.18)),
        "complexity": ((0.0, 0.22, 0.50, 0.78, 1.0), (0.84, 1.18, 0.92, 1.24, 0.98)),
        "memory": ((0.0, 0.42, 0.72, 1.0), (1.02, 0.80, 1.12, 0.92)),
        "resolution": ((0.0, 0.38, 1.0), (0.96, 0.78, 0.52)),
    }
    positions, values = anchors.get(section_name, anchors["activation"])
    progress = np.linspace(0.0, 1.0, length, dtype=np.float32)
    shaped = np.interp(progress, positions, values).astype(np.float32)
    learned_strength = min(1.0, max(0.0, (d5_development_drive(5) - 1.0) / 0.34))
    return (1.0 + (shaped - 1.0) * learned_strength).astype(np.float32)


def d5_global_evolution_curve(length):
    """Multi-stage D5 arc used by generated synthesis, not a single slow swell."""
    length = max(0, int(length))
    if length == 0:
        return np.zeros(0, dtype=np.float32)
    positions = (0.0, 0.14, 0.31, 0.52, 0.70, 0.87, 1.0)
    values = (0.72, 0.98, 1.18, 0.86, 1.25, 1.04, 0.58)
    progress = np.linspace(0.0, 1.0, length, dtype=np.float32)
    shaped = np.interp(progress, positions, values).astype(np.float32)
    learned_strength = min(1.0, max(0.0, (d5_development_drive(5) - 1.0) / 0.34))
    return (1.0 + (shaped - 1.0) * learned_strength).astype(np.float32)


def d5_soft_grid_start(base_position, section_start, role, dream_level, pulse_bpm):
    """Pull D5 entries gently toward one shared pulse without hard quantisation."""
    if dream_level != 5:
        return float(base_position)
    beat = 60.0 / max(1.0, float(pulse_bpm))
    if role in ["gesture", "impact", "noise"]:
        subdivision, strength = beat / 2.0, 0.26
    elif role == "texture":
        subdivision, strength = beat, 0.12
    else:
        subdivision, strength = beat * 2.0, 0.08
    relative = float(base_position) - float(section_start)
    nearest = float(section_start) + round(relative / subdivision) * subdivision
    return float(base_position + (nearest - base_position) * strength)


def d5_continuity_start(start, duration, role, previous_role_end, dream_level, pulse_bpm):
    """Prevent isolated D5 lane entries and preserve overlap between related roles."""
    if dream_level != 5 or previous_role_end is None:
        return float(start)
    beat = 60.0 / max(1.0, float(pulse_bpm))
    max_gap_beats = {
        "gesture": 2.0,
        "impact": 2.5,
        "noise": 3.0,
        "texture": 1.0,
        "resonance": 1.0,
    }
    overlap = {
        "gesture": 0.18,
        "impact": 0.12,
        "noise": 0.30,
        "texture": 1.40,
        "resonance": 2.40,
    }
    latest_with_continuity = float(previous_role_end) + max_gap_beats.get(role, 2.0) * beat
    adjusted = float(start)
    if adjusted > latest_with_continuity:
        # Pull a large gap only part-way back. The old aesthetic depended on
        # elastic spacing, while the edge guard already protects phrase endings.
        adjusted -= (adjusted - latest_with_continuity) * 0.42
    if (
        role in ["texture", "resonance"]
        and 0.0 < adjusted - float(previous_role_end) < 0.75
    ):
        adjusted = float(previous_role_end) - min(
            overlap[role] * 0.45,
            max(0.1, float(duration) * 0.10),
        )
    return max(0.0, adjusted)


def form_density_multiplier(section_name, dream_level):
    """Create a render-specific energy arc without abandoning learned density."""
    variant = D5_FORM_VARIANTS.get(CURRENT_FORM_VARIANT, {})
    base = float(variant.get(section_name, 1.0))
    # D1/D3 keep the accepted structure closer to neutral. D5 is allowed the
    # largest formal contrast, but no level is forced through the same arc on
    # every render.
    drive = d5_energy_drive(dream_level) if dream_level == 5 else {1: 0.46, 3: 0.68}[dream_level]
    if base >= 1.0:
        return 1.0 + (base - 1.0) * drive
    return 1.0 - (1.0 - base) * drive


def d5_selection_character_factor(obj, dream_level):
    """Prefer energetic, musical, synthetic material only when rendering D5."""
    if dream_level != 5:
        return 1.0
    features = obj.get("features", {})
    energy = max(0.0, min(1.0, float(features.get("energy", 0.0)) / 0.16))
    musicality = max(0.0, min(1.0, float(features.get("musicality", 0.5))))
    gesture = max(0.0, min(1.0, float(features.get("gesture_strength", 0.5))))
    synthetic = effective_synthetic_score(obj)
    energetic_quality = 0.40 * musicality + 0.34 * gesture + 0.26 * energy
    drive = d5_energy_drive(dream_level)
    synthetic_drive = learned_factor("synthetic_material_weight", 4.0, 0.85, 1.50)
    energetic_factor = 1.16 - 0.24 * energetic_quality * drive
    synthetic_factor = 1.12 - 0.20 * synthetic * synthetic_drive
    focused = max(0.52, energetic_factor * synthetic_factor)
    # Keep some present-day preference for musical synthetic material, but
    # retain the broader, airier palette of the listener-preferred baseline.
    return float(1.0 + (focused - 1.0) * 0.55)


def learned_synthetic_material_factor(obj):
    """Turn learned synth feedback into a bounded object-level preference.

    A neutral learning profile leaves selection unchanged.  Once the listener
    has asked for more synthetic material, objects with stronger synthetic
    evidence are favoured and ambient, low-synthetic objects receive a small
    counterweight.  This complements the recording-level breadth quota: it
    prevents a nominally synthetic WAV from contributing only its most
    acoustic/ambient fragments.
    """
    features = obj.get("features", {})
    synthetic = effective_synthetic_score(obj)
    ambient = max(0.0, min(1.0, float(features.get("ambient_score", 0.5))))
    request = max(
        0.0,
        learned_factor("synthetic_material_weight", 4.0, 0.78, 1.60) - 1.0,
    )
    if request <= 0.0:
        return 1.0
    synth_preference = float(np.exp(-2.1 * request * (synthetic - 0.5)))
    watery_counterweight = 1.0 + 0.45 * request * ambient * (1.0 - synthetic)
    return float(max(0.55, min(1.58, synth_preference * watery_counterweight)))


def learned_instrument_material_factor(obj):
    """Prefer confirmed instrument-hybrid material after explicit feedback."""
    request = max(
        0.0,
        learned_factor("instrument_material_weight", 3.2, 0.82, 1.55) - 1.0,
    )
    if request <= 0.0:
        return 1.0
    origin = source_origin(obj)
    if origin == "instrument_hybrid":
        return float(max(0.58, np.exp(-1.55 * request)))
    if origin == "synthetic":
        return float(max(0.78, np.exp(-0.48 * request)))
    if origin == "natural":
        return float(min(1.18, 1.0 + 0.22 * request))
    return 1.0


def learned_origin_mix_factor(obj, usage_counts=None):
    """Keep requested synth/instrument presence audible across actual events."""
    if not usage_counts:
        return 1.0
    total = max(1, sum(max(0, int(value)) for value in usage_counts.values()))
    origin = source_origin(obj)
    synth_request = max(
        0.0,
        learned_factor("synthetic_material_weight", 3.0, 0.85, 1.55) - 1.0,
    )
    instrument_request = max(
        0.0,
        learned_factor("instrument_material_weight", 3.2, 0.82, 1.55) - 1.0,
    )
    synth_used = sum(
        count
        for recording, count in usage_counts.items()
        if LIBRARY_SOURCE_LABELS.get(recording) == "synthetic"
    )
    instrument_used = sum(
        count
        for recording, count in usage_counts.items()
        if LIBRARY_SOURCE_LABELS.get(recording) == "instrument_hybrid"
    )
    synth_target = min(0.30, 0.14 + 0.18 * synth_request)
    instrument_target = min(0.24, 0.11 + 0.24 * instrument_request)
    if origin == "synthetic" and synth_request > 0.04 and synth_used / total < synth_target:
        return 0.48
    if (
        origin == "instrument_hybrid"
        and instrument_request > 0.04
        and instrument_used / total < instrument_target
    ):
        return 0.52
    return 1.0


def learned_synthetic_candidate_pool(pool):
    """Probabilistically focus on the library's synth-rich objects after feedback.

    The cutoff is relative to the current candidate pool, so replacing the
    user's library does not require named files or fixed categories.  The gate
    is inactive for neutral feedback and remains probabilistic when active.
    """
    if len(pool) < 6:
        return pool
    request = max(
        0.0,
        learned_factor("synthetic_material_weight", 4.0, 0.78, 1.60) - 1.0,
    )
    if request <= 0.04:
        return pool
    scores = np.asarray(
        [effective_synthetic_score(obj) for obj in pool],
        dtype=np.float64,
    )
    cutoff = float(np.quantile(scores, 0.55))
    synth_rich = [
        obj for obj in pool
        if effective_synthetic_score(obj) >= cutoff
    ]
    focus_probability = min(0.68, 0.18 + 0.76 * request)
    if len(synth_rich) >= 3 and random.random() < focus_probability:
        return synth_rich
    return pool


def long_layer_diversity_factor(obj, dream_level, usage_counts=None):
    """Rotate sustained material between D-levels and within one render.

    Lower scores are preferred by the selector. The deterministic family bias
    gives D1, D3 and D5 different sustained-source tendencies without excluding
    any recording. Explicit listener feedback strengthens the rotation.
    """
    role = obj.get("role") or classify_object(obj)
    duration = float(obj.get("duration", 0.0))
    if role not in {"texture", "resonance"} or duration < 4.0:
        return 1.0

    target_family = {1: 0, 3: 1, 5: 2}[dream_level]
    source_family = deterministic_group(
        f"hyponoia-long-layer-v1:{obj.get('recording', '')}", groups=3
    )
    rotation = composition_feedback_audio_snapshot()["long_layer_rotation"]
    affinity = 0.88 if source_family == target_family else 1.08
    affinity = 1.0 + (affinity - 1.0) * rotation

    previous_uses = 0 if usage_counts is None else usage_counts.get(obj.get("recording"), 0)
    reuse_penalty = 1.0 + min(0.65, 0.11 * max(0, int(previous_uses))) * rotation
    # Sustained objects remain welcome, but very long sources should not
    # dominate every section after stretching, emergence and delay tails.
    duration_penalty = 1.0 + min(0.55, max(0.0, duration - 7.0) * 0.045) * rotation
    return float(max(0.58, min(2.4, affinity * reuse_penalty * duration_penalty)))


def material_plan_limits(dream_level):
    """Balanced palette that cannot shrink when development is requested.

    Development describes what happens *inside and across* phrases.  Earlier
    versions also treated it as a request for a narrower palette, which made a
    request such as "more evolution" collapse D1 from six recordings to five.
    Accepted whole-render breadth is now a floor for every D level.
    """
    recording_base = {1: 6, 3: 10, 5: 16}[dream_level]
    object_base = {1: 12, 3: 20, 5: 30}[dream_level]
    breadth = learned_factor("exploration_weight", 1.6, 0.90, 1.30)
    development = learned_factor("material_development_weight", 1.2, 0.88, 1.18)
    expansion = max(1.0, breadth) * (1.0 + 0.24 * max(0.0, development - 1.0))
    target_method = getattr(COMPOSITION_PREFERENCE, "target_unique_recordings", None)
    target_recordings = target_method(dream_level) if callable(target_method) else None
    if target_recordings is not None:
        # The positive structure defines coherent per-render breadth. Library
        # exploration may open it slightly, while coverage rotates through the
        # rest of the library across later renders.
        structure_expansion = (
            1.0
            + 0.40 * max(0.0, breadth - 1.0)
            + 0.20 * max(0.0, development - 1.0)
        )
        recordings = max(
            int(target_recordings),
            int(round(float(target_recordings) * structure_expansion)),
        )
        object_method = getattr(COMPOSITION_PREFERENCE, "target_unique_objects", None)
        target_objects = object_method(dream_level) if callable(object_method) else None
        if target_objects is not None:
            object_expansion = 1.0 + 0.28 * max(0.0, breadth - 1.0)
            objects = max(
                recordings,
                int(round(float(target_objects) * object_expansion)),
            )
        else:
            objects = max(
                recordings * 2,
                int(round(object_base * recordings / recording_base)),
            )
    else:
        recordings = max(recording_base, int(round(recording_base * expansion)))
        objects = max(object_base, recordings, int(round(object_base * expansion)))
    return recordings, objects


def composition_duration_selection_factor(obj, desired_role, dream_level):
    """Prefer material capable of reaching the accepted phrase duration.

    The learned target comes from the complete positive D1/D3/D5 render, while
    the factors below only estimate the transform already applied by each role.
    This keeps short synth gestures available without letting sub-second clips
    become every sustained layer.
    """
    if desired_role not in {"texture", "resonance"}:
        return 1.0
    target_method = getattr(
        COMPOSITION_PREFERENCE, "target_average_event_duration", None
    )
    target = target_method(dream_level) if callable(target_method) else None
    if target is None or target <= 0.0:
        return 1.0
    role_target = float(target) * (1.18 if desired_role == "resonance" else 0.90)
    transform_estimate = 5.4 if desired_role == "resonance" else 3.9
    predicted = max(0.05, float(obj.get("duration", 0.0)) * transform_estimate)
    if predicted < role_target:
        shortfall = role_target / predicted
        return float(min(2.75, 1.0 + 0.48 * (shortfall - 1.0)))
    excess = predicted / role_target
    return float(min(1.35, 1.0 + 0.08 * max(0.0, excess - 1.6)))


def recording_can_sustain(rec_objects, dream_level):
    """Whether one recording can contribute an evolving sustained phrase."""
    target_method = getattr(
        COMPOSITION_PREFERENCE, "target_average_event_duration", None
    )
    target = target_method(dream_level) if callable(target_method) else None
    minimum_source = max(2.0, (float(target) / 5.2) if target else 2.8)
    for obj in rec_objects:
        role = obj.get("role")
        if role is None and "features" in obj and "duration" in obj:
            role = classify_object(obj)
        if role in {"texture", "resonance"} and float(obj.get("duration", 0.0)) >= minimum_source:
            return True
    return False


def diversify_overused_recordings(pool, usage_counts, dream_level, palette_size):
    """Stop one WAV from swallowing a learned multi-recording palette."""
    if not usage_counts or len(pool) < 3 or palette_size <= 1:
        return pool
    total = sum(max(0, int(value)) for value in usage_counts.values())
    if total < 4:
        return pool
    fair_share = total / max(1, int(palette_size))
    dominance = {1: 2.75, 3: 2.25, 5: 1.85}[dream_level]
    dynamic_limit = max(4.0, fair_share * dominance)
    event_method = getattr(COMPOSITION_PREFERENCE, "target_event_count", None)
    target_events = (
        event_method(dream_level, OUTPUT_DURATION) if callable(event_method) else None
    )
    accepted_share_limit = {1: 0.42, 3: 0.31, 5: 0.20}[dream_level]
    learned_limit = (
        max(4.0, float(target_events) * accepted_share_limit)
        if target_events is not None
        else dynamic_limit
    )
    limit = min(dynamic_limit, learned_limit)
    overused = {
        recording
        for recording, count in usage_counts.items()
        if float(count) >= limit
    }
    diversified = [obj for obj in pool if obj.get("recording") not in overused]
    return diversified if len(diversified) >= 3 else pool


def sample_key(obj):
    """Stable identifier for one analysed object inside one source recording."""
    return stable_sample_key(str(obj["recording_id"]), str(obj["object_id"]))


def load_sample_learning_profile(memory):
    """Load persistent sample-level preferences without changing global learning weights."""
    default = {
        "version": 2,
        "generator_revision": GENERATOR_REVISION,
        "description": "Hyponoia sample-level learning profile with stable content IDs.",
        "total_render_selections": 0,
        "completed_renders": 0,
        "recording_coverage": {},
        "retired_recording_coverage": {},
        "samples": {},
    }
    if not os.path.exists(SAMPLE_LEARNING_FILE):
        return default
    try:
        with open(SAMPLE_LEARNING_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return default
        data.setdefault("version", 1)
        data.setdefault("description", default["description"])
        data.setdefault("total_render_selections", 0)
        data.setdefault("completed_renders", 0)
        data.setdefault("recording_coverage", {})
        data.setdefault("retired_recording_coverage", {})
        data.setdefault("samples", {})
        migrated, migration_report = migrate_sample_profile(data, memory)
        sync_recording_coverage(
            migrated,
            (recording.get("recording", "") for recording in memory),
        )
        if migration_report["moved_entries"] or migration_report["from_version"] < 2:
            atomic_write_json(SAMPLE_LEARNING_FILE, migrated)
            print("Sample profile migration:", migration_report)
        return migrated
    except (OSError, json.JSONDecodeError) as exc:
        print("Could not read sample learning profile; using a fresh profile:", exc)
        return default


def save_sample_learning_profile(profile):
    atomic_write_json(SAMPLE_LEARNING_FILE, profile)


def ensure_sample_entry(profile, obj):
    key = sample_key(obj)
    samples = profile.setdefault("samples", {})
    entry = samples.setdefault(key, {
        "recording": obj["recording"],
        "recording_id": obj["recording_id"],
        "object_id": obj["object_id"],
        "legacy_object_id": obj.get("legacy_id"),
        "learned_value": 0.0,
        "times_selected": 0,
        "feedback_updates": 0,
        "last_used": None,
    })
    return entry


def sample_learning_factor(obj, sample_profile):
    """Lower score is better; repeated evidence creates a bounded audible bias."""
    entry = ensure_sample_entry(sample_profile, obj)
    learned_value = float(entry.get("learned_value", 0.0))
    return float(max(0.45, min(2.20, np.exp(-1.40 * learned_value))))


def selection_probabilities(scores):
    """Return the exact normalised probabilities used by weighted selection."""
    values = np.asarray(scores, dtype=np.float64)
    weights = 1.0 / (values + 0.05)
    return weights / weights.sum()


def sample_exploration_factor(obj, sample_profile):
    """Soft exploration bonus; never bans or hard-caps familiar material."""
    entry = ensure_sample_entry(sample_profile, obj)
    plays = max(0, int(entry.get("times_selected", 0)))
    total = max(0, int(sample_profile.get("total_render_selections", 0)))
    bonus = np.sqrt(np.log1p(total + 1.0) / (plays + 1.0))
    strength = learned_factor("exploration_weight", 3.0, 0.70, 1.45)
    return float(np.exp(-0.16 * bonus * strength))


def save_render_report(
    outfile,
    dream_level,
    usage_by_sample,
    usage_details,
    usage_by_recording,
    role_counts,
    temporal_metrics=None,
    frequency_stems=None,
    event_timeline=None,
    source_development=None,
):
    os.makedirs(RENDER_REPORT_FOLDER, exist_ok=True)
    report = {
        "version": 2,
        "generator_revision": GENERATOR_REVISION,
        "timestamp": utc_timestamp(),
        "audio_file": outfile,
        "render_seed": int(RENDER_SEED),
        "dream_level": int(dream_level),
        "form_variant": CURRENT_FORM_VARIANT,
        "background_design": {
            1: "source_bed_drone_pad_and_musical_phrases",
            3: "source_bed_drone_pad_phrases_and_spectral_halo",
            5: "source_bed_drone_pad_and_pulsed_source_phrases",
        }[int(dream_level)],
        "harmony_state": dict(HARMONY_STATE),
        "total_sample_selections": int(sum(usage_by_sample.values())),
        "unique_samples": int(len(usage_by_sample)),
        "samples": dict(sorted(usage_by_sample.items())),
        "sample_usage_details": dict(sorted(usage_details.items())),
        "event_timeline": event_timeline or [],
        "source_development": source_development or {},
        "recordings": dict(sorted(usage_by_recording.items())),
        "role_counts": role_counts,
        "control_snapshot": dict(LEARNING_WEIGHTS),
        "composition_feedback_audio": composition_feedback_audio_snapshot(),
        "representation_assist": REPRESENTATION_ASSIST.snapshot(),
        "origin_learning": dict(ORIGIN_LEARNING_SNAPSHOT),
        "composition_preference": COMPOSITION_PREFERENCE.snapshot(),
        "library_coverage": dict(CURRENT_LIBRARY_COVERAGE_SNAPSHOT),
        "temporal_profile": {
            key: round(float(value), 6)
            for key, value in d5_temporal_profile(dream_level).items()
        },
        "temporal_metrics": temporal_metrics or {},
        "frequency_stems": frequency_stems or {},
        "reference_targets": dict(D5_REFERENCE_TARGETS) if dream_level == 5 else {},
        "target_sample_rate": TARGET_SR,
    }
    base = os.path.splitext(os.path.basename(outfile))[0]
    timestamped_path = os.path.join(RENDER_REPORT_FOLDER, f"{base}_render_report.json")
    for path in (RENDER_REPORT_FILE, timestamped_path):
        atomic_write_json(path, report)
    return timestamped_path


def load_memory_objects():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)

    objects = []

    for recording in memory:
        for obj in recording["objects"]:
            objects.append({
                "recording": recording["recording"],
                "recording_id": recording.get("recording_id", recording["recording"]),
                "id": obj.get("stable_id", obj["id"]),
                "object_id": obj.get("stable_id", obj["id"]),
                "legacy_id": obj.get("legacy_id", obj.get("id")),
                "start": obj["start"],
                "end": obj["end"],
                "duration": obj["duration"],
                "features": obj["features"],
                "times_used": obj.get("times_used", 0)
            })

    print("Loaded objects:", len(objects))
    return objects


def profile_distance(obj, profile):
    f = obj["features"]
    return (
        abs(f["energy"] - profile["energy"]) * 8
        + abs(f["brightness"] - profile["brightness"]) / 4500
        + abs(f["noise"] - profile["noise"]) * 12
        + abs(f["attack"] - profile["attack"]) / 6
    )


def continuity_distance(a, b):
    fa = a["features"]
    fb = b["features"]
    base = (
        abs(fa.get("energy", 0.0) - fb.get("energy", 0.0)) * 5
        + abs(fa.get("brightness", 0.0) - fb.get("brightness", 0.0)) / 6500
        + abs(fa.get("noise", 0.0) - fb.get("noise", 0.0)) * 6
        + abs(fa.get("attack", 0.0) - fb.get("attack", 0.0)) / 10
    )

    # If both objects have pitch estimates, prefer meaningful continuity.
    ma = fa.get("pitch_midi")
    mb = fb.get("pitch_midi")
    if ma is not None and mb is not None:
        interval = abs(float(ma) - float(mb)) % 12.0
        consonance = min(interval, abs(interval - 12), abs(interval - 7), abs(interval - 5), abs(interval - 3), abs(interval - 4))
        base += consonance / 12.0

    # Similar phrase function connects better.
    if fa.get("phrase_role") == fb.get("phrase_role"):
        base *= 0.92

    return base

def classify_object(obj):
    """Role classification using Memory v3 musical descriptors when available."""
    f = obj["features"]
    dur = obj["duration"]

    gesture_type = f.get("gesture_type")
    if gesture_type in ["drone", "resonance"]:
        return "resonance"
    if gesture_type in ["texture", "air"]:
        return "texture"
    if gesture_type in ["burst", "gesture"]:
        return "gesture"
    if gesture_type == "noise":
        return "noise"

    energy = f.get("energy", 0.0)
    brightness = f.get("brightness", 0.0)
    noise = f.get("noise", 0.0)
    attack = f.get("attack", 0.0)
    harmonicity = f.get("harmonicity", 0.0)
    resonance_strength = f.get("resonance_strength", 0.0)

    if attack > 4.0 and dur < 4.5:
        return "gesture"
    if energy > 0.08 and attack > 3.0:
        return "impact"
    if noise > 0.035:
        return "noise"
    if dur >= 5.0 and (noise < 0.025 or harmonicity > 0.55 or resonance_strength > 0.62):
        return "resonance"
    if dur >= 3.0:
        return "texture"
    return "gesture"

def musical_value(obj):
    """
    Memory v3 musical-value score. Lower is better for selection.
    Uses musicality/richness/harmonicity/ambient potential while still avoiding harsh short attacks.
    """
    f = obj["features"]
    dur = obj["duration"]

    energy = f.get("energy", 0.0)
    brightness = f.get("brightness", 0.0)
    noise = f.get("noise", 0.0)
    attack = f.get("attack", 0.0)

    musicality = f.get("musicality", 0.5)
    richness = f.get("richness", 0.5)
    harmonicity = f.get("harmonicity", 0.5)
    resonance_strength = f.get("resonance_strength", 0.5)
    ambient_score = f.get("ambient_score", 0.5)
    novelty = f.get("novelty", 0.5)
    critic_score = f.get("critic_score", 0.5)
    fatigue = f.get("fatigue", 0.0)
    transient_score = f.get("transient_score", 0.5)
    synthetic_score = effective_synthetic_score(obj)

    value = 1.0

    # Reward musically useful, rich, resonant objects.
    value *= (1.38 - 0.55 * musicality * learned_factor("musicality_weight", 4.5))
    value *= (1.30 - 0.38 * richness * learned_factor("richness_weight", 4.5))
    value *= (1.18 - 0.22 * harmonicity)
    value *= (1.20 - 0.30 * resonance_strength)
    value *= (1.10 - 0.12 * ambient_score * learned_factor("ambient_weight", 4.0))
    value *= (1.14 - 0.22 * novelty)
    value *= (1.12 - 0.18 * critic_score)
    value *= (1.12 - 0.20 * synthetic_score * learned_factor("synthetic_material_weight", 3.5))
    value *= (1.0 + 0.55 * fatigue)

    # Prefer usable phrase durations.
    if 1.2 <= dur <= 9.5:
        value *= 0.78
    elif 0.55 <= dur < 1.2:
        value *= 0.98
    else:
        value *= 1.10

    # Keep high-frequency material, but avoid yelpy/animal-like short attacks.
    if dur < 2.2 and attack > 5.0 and noise > 0.025 and transient_score > 0.65:
        value *= 2.35 * learned_factor("impact_penalty", 4.0)

    # Avoid uncontrolled harshness, but do not kill useful bright synth air.
    if noise > 0.05 and harmonicity < 0.35:
        value *= 1.55 * learned_factor("noise_penalty", 4.0)

    # Bright material is welcome when musical/harmonic.
    if brightness > 4500 and harmonicity > 0.45 and musicality > 0.52:
        value *= 0.82

    # Avoid overusing very loud short material as foreground.
    if energy > 0.15 and dur < 3.0 and musicality < 0.6:
        value *= 1.45

    return float(max(0.05, value))

def palette_group(recording_name):
    """Stable pseudo-family from filename. This lets D5 use different families
    across sections without hard-coding any uploaded sample.
    """
    return deterministic_group(recording_name, groups=4)


def build_role_pools(objects):
    pools = {
        "gesture": [],
        "texture": [],
        "resonance": [],
        "noise": [],
        "impact": []
    }

    for obj in objects:
        role = classify_object(obj)
        obj["role"] = role
        pools[role].append(obj)

    print("Role pools:")
    for role, pool in pools.items():
        print(f"  {role}: {len(pool)}")

    return pools


def representation_continuity_factor(previous, candidate):
    """Use learned continuity only inside validated non-drone role families."""
    if previous is None or candidate is None:
        return 1.0
    previous_role = previous.get("role") or classify_object(previous)
    candidate_role = candidate.get("role") or classify_object(candidate)
    if previous_role != candidate_role or previous_role not in REPRESENTATION_ASSIST_ROLES:
        return 1.0
    return REPRESENTATION_ASSIST.continuity_factor(
        previous.get("object_id"), candidate.get("object_id")
    )


CURRENT_MATERIAL_PLAN = None
CURRENT_FORM_VARIANT = "baseline"

D5_FORM_VARIANTS = {
    "aesthetic_bridge": {
        "opening": 0.96,
        "activation": 1.04,
        "complexity": 1.08,
        "memory": 1.00,
        "resolution": 0.92,
    },
    "central_surge": {
        "opening": 0.90,
        "activation": 1.10,
        "complexity": 1.20,
        "memory": 0.98,
        "resolution": 0.86,
    },
    "double_wave": {
        "opening": 0.88,
        "activation": 1.18,
        "complexity": 1.04,
        "memory": 1.15,
        "resolution": 0.86,
    },
    "late_bloom": {
        "opening": 0.88,
        "activation": 1.00,
        "complexity": 1.10,
        "memory": 1.22,
        "resolution": 0.84,
    },
}


FORM_TIMING_VARIANTS = {
    "aesthetic_bridge": (
        ("opening", 0.00, 0.19, 0.65),
        ("activation", 0.13, 0.37, 1.05),
        ("complexity", 0.29, 0.59, 1.45),
        ("memory", 0.50, 0.79, 1.05),
        ("resolution", 0.69, 0.99, 0.75),
    ),
    "central_surge": (
        ("opening", 0.00, 0.16, 0.62),
        ("activation", 0.10, 0.34, 1.10),
        ("complexity", 0.25, 0.66, 1.50),
        ("memory", 0.58, 0.82, 0.98),
        ("resolution", 0.73, 0.99, 0.70),
    ),
    "double_wave": (
        ("opening", 0.00, 0.17, 0.60),
        ("activation", 0.10, 0.31, 1.23),
        ("complexity", 0.25, 0.55, 1.28),
        ("memory", 0.47, 0.83, 1.20),
        ("resolution", 0.75, 0.99, 0.68),
    ),
    "late_bloom": (
        ("opening", 0.00, 0.22, 0.58),
        ("activation", 0.15, 0.40, 0.94),
        ("complexity", 0.33, 0.66, 1.30),
        ("memory", 0.56, 0.88, 1.30),
        ("resolution", 0.79, 0.99, 0.66),
    ),
}


def select_form_variant(dream_level, seed=None):
    """Select a reproducible form while keeping D1/D3/D5 structurally distinct."""
    names = tuple(FORM_TIMING_VARIANTS)
    seed = int(RENDER_SEED if seed is None else seed)
    level_offset = {1: 0, 3: 1, 5: 2}[int(dream_level)]
    return names[((seed // 997) + level_offset) % len(names)]


def composed_form(dream_level, duration=None, seed=None):
    """Scale the selected learned-compatible form to the current render duration."""
    duration = float(OUTPUT_DURATION if duration is None else duration)
    variant = select_form_variant(dream_level, seed=seed)
    return [
        (name, start * duration, end * duration, density)
        for name, start, end, density in FORM_TIMING_VARIANTS[variant]
    ]


def planned_form_items(form, dream_level):
    """Allocate density from the learned positive whole-render structure."""
    raw_items = []
    for section_name, section_start, section_end, density in form:
        adjusted = density * learned_factor("richness_weight", 3.2, 0.78, 1.25)
        adjusted *= form_density_multiplier(section_name, dream_level)
        adjusted *= ARTIST_STYLE.form_factor(
            ((section_start + section_end) * 0.5) / max(1.0, OUTPUT_DURATION),
            dream_level,
            RENDER_SEED,
        )
        if dream_level == 5 and section_name == "complexity":
            adjusted *= 1.34
        elif dream_level == 3 and section_name == "complexity":
            adjusted *= 1.16
        if dream_level == 5:
            items = int(13 * adjusted * 1.12)
        elif dream_level == 3:
            items = int(11 * adjusted * 1.18)
        else:
            items = int(11 * adjusted * 1.15)
        raw_items.append(max(1, int(items * dream_activity_multiplier(dream_level))))

    target = COMPOSITION_PREFERENCE.target_event_count(dream_level, OUTPUT_DURATION)
    if target is None or not raw_items:
        return raw_items
    target = max(len(raw_items), int(target))
    scaled = np.asarray(raw_items, dtype=np.float64)
    scaled *= target / max(1.0, float(scaled.sum()))
    allocated = np.maximum(1, np.floor(scaled).astype(int))
    fractions = scaled - np.floor(scaled)
    while int(allocated.sum()) < target:
        index = int(np.argmax(fractions))
        allocated[index] += 1
        fractions[index] = -1.0
    while int(allocated.sum()) > target:
        candidates = [i for i, value in enumerate(allocated) if value > 1]
        if not candidates:
            break
        index = min(candidates, key=lambda i: fractions[i])
        allocated[index] -= 1
    return [int(value) for value in allocated]


def planned_role_targets(total_events, dream_level):
    """Convert the accepted role distribution into exact per-render counts."""
    target_method = getattr(COMPOSITION_PREFERENCE, "target_role_distribution", None)
    distribution = target_method(dream_level) if callable(target_method) else {}
    if not distribution or total_events <= 0:
        return None
    roles = ("gesture", "texture", "resonance", "noise", "impact")
    raw = {role: max(0.0, float(distribution.get(role, 0.0))) * total_events for role in roles}
    allocated = {role: int(np.floor(value)) for role, value in raw.items()}
    while sum(allocated.values()) < total_events:
        role = max(roles, key=lambda item: raw[item] - allocated[item])
        allocated[role] += 1
    while sum(allocated.values()) > total_events:
        candidates = [role for role in roles if allocated[role] > 0]
        role = min(candidates, key=lambda item: raw[item] - allocated[item])
        allocated[role] -= 1
    return allocated


def build_material_plan(objects, profile, dream_level, sample_profile=None):
    by_recording = {}

    for obj in objects:
        rec = obj["recording"]
        by_recording.setdefault(rec, []).append(obj)

    recording_scores = []
    recording_synthetic = {}
    learned_samples = (sample_profile or {}).get("samples", {})
    exploration = learned_factor("exploration_weight", 1.4, 0.85, 1.22)

    for rec, rec_objects in by_recording.items():
        distances = [profile_distance(o, profile) for o in rec_objects]
        past_uses = sum(
            max(0, int(learned_samples.get(sample_key(obj), {}).get("times_selected", 0)))
            for obj in rec_objects
        )
        # Soft rotation between renders: familiar recordings stay available,
        # while equally suitable underused material gains a small advantage.
        history_penalty = np.log1p(past_uses) * 0.035 * exploration
        synthetic_mean = float(np.mean([
            effective_synthetic_score(obj) for obj in rec_objects
        ]))
        recording_synthetic[rec] = synthetic_mean
        synthetic_request = max(
            0.0,
            learned_factor("synthetic_material_weight", 3.0, 0.85, 1.55) - 1.0,
        )
        synthetic_bonus = max(0.70, 1.0 - 0.34 * synthetic_mean * synthetic_request)
        instrument_request = max(
            0.0,
            learned_factor("instrument_material_weight", 3.2, 0.82, 1.55) - 1.0,
        )
        instrument_bonus = (
            max(0.68, 1.0 - 0.42 * instrument_request)
            if LIBRARY_SOURCE_LABELS.get(rec) == "instrument_hybrid"
            else 1.0
        )
        preference_factor = COMPOSITION_PREFERENCE.recording_factor(rec_objects, dream_level)
        coverage_factor = recording_coverage_factor(
            rec,
            sample_profile,
            strength=exploration,
        )
        recording_scores.append(
            (
                (np.mean(distances) + history_penalty)
                * synthetic_bonus
                * instrument_bonus
                * preference_factor
                * coverage_factor,
                rec,
            )
        )

    recording_limit, _ = material_plan_limits(dream_level)
    target_family = {1: 0, 3: 1, 5: 2}[dream_level]
    rotation = learned_factor("long_layer_diversity_weight", 2.4, 0.86, 1.48)
    variety = learned_factor("exploration_weight", 2.2, 0.86, 1.45)
    # The earlier nested D1 -> D3 -> D5 plan made controlled renders reuse the
    # same audible core. Each level now receives a soft deterministic affinity
    # to a different third of the library. Nothing is excluded: suitability and
    # learned sample evidence still win when a recording is clearly preferable.
    ranked = []
    for base_score, rec in recording_scores:
        family = deterministic_group(f"hyponoia-material-plan-v2:{rec}", groups=3)
        family_factor = 0.78 if family == target_family else 1.10
        family_factor = 1.0 + (family_factor - 1.0) * rotation * variety
        ranked.append((base_score * family_factor, rec))
    ranked.sort(key=lambda item: item[0])

    candidate_limit = min(len(ranked), max(recording_limit * 4, 24))
    candidates = [rec for _, rec in ranked[:candidate_limit]]
    preferred = [
        rec for rec in candidates
        if deterministic_group(f"hyponoia-material-plan-v2:{rec}", groups=3) == target_family
    ]
    others = [rec for rec in candidates if rec not in preferred]
    preferred_count = min(len(preferred), max(1, int(round(recording_limit * 0.72))))
    allowed = random.sample(preferred, preferred_count)
    remaining = [rec for rec in others + preferred if rec not in allowed]
    needed = min(recording_limit - len(allowed), len(remaining))
    allowed.extend(random.sample(remaining, needed))

    # Guarantee breadth among strongly synthetic source recordings when the
    # listener explicitly asks for more synth. This operates at recording level,
    # so the selector cannot satisfy the request by repeating one favourite WAV.
    synth_request = max(
        0.0,
        learned_factor("synthetic_material_weight", 3.0, 0.85, 1.55) - 1.0,
    )
    if synth_request > 0.04 and allowed:
        synth_quota = min(
            recording_limit,
            max(2, int(round(recording_limit * min(0.48, 0.25 + synth_request * 0.30)))),
        )
        high_synth = [
            rec for rec in sorted(candidates, key=lambda item: recording_synthetic[item], reverse=True)
            if recording_synthetic[rec] >= 0.54
        ]
        selected_high = [rec for rec in allowed if recording_synthetic[rec] >= 0.54]
        replacements = [rec for rec in high_synth if rec not in allowed]
        replaceable = sorted(
            [rec for rec in allowed if recording_synthetic[rec] < 0.54],
            key=lambda item: recording_synthetic[item],
        )
        while len(selected_high) < synth_quota and replacements and replaceable:
            outgoing = replaceable.pop(0)
            incoming = replacements.pop(0)
            allowed[allowed.index(outgoing)] = incoming
            selected_high.append(incoming)

    # Human-confirmed origins receive an explicit palette floor. Similarity
    # inference still helps discovery, but it cannot displace all confirmed
    # examples of a category the listener explicitly requested.
    protected = set()
    confirmed_synth_request = max(
        0.0,
        learned_factor("synthetic_material_weight", 3.0, 0.85, 1.55) - 1.0,
    )
    if confirmed_synth_request > 0.04:
        quota = min(
            max(1, int(round(recording_limit * 0.34))),
            len([rec for rec in by_recording if LIBRARY_SOURCE_LABELS.get(rec) == "synthetic"]),
        )
        available = [rec for _, rec in ranked if LIBRARY_SOURCE_LABELS.get(rec) == "synthetic"]
        selected = [rec for rec in allowed if LIBRARY_SOURCE_LABELS.get(rec) == "synthetic"]
        replacements = [rec for rec in available if rec not in allowed]
        outgoing = [rec for rec in reversed(allowed) if LIBRARY_SOURCE_LABELS.get(rec) != "synthetic"]
        while len(selected) < quota and replacements and outgoing:
            old = outgoing.pop(0)
            new = replacements.pop(0)
            allowed[allowed.index(old)] = new
            selected.append(new)
        protected.update(selected)

    instrument_request = max(
        0.0,
        learned_factor("instrument_material_weight", 3.2, 0.82, 1.55) - 1.0,
    )
    if instrument_request > 0.04:
        quota = min(
            max(1, int(round(recording_limit * 0.25))),
            len([rec for rec in by_recording if LIBRARY_SOURCE_LABELS.get(rec) == "instrument_hybrid"]),
        )
        available = [rec for _, rec in ranked if LIBRARY_SOURCE_LABELS.get(rec) == "instrument_hybrid"]
        selected = [rec for rec in allowed if LIBRARY_SOURCE_LABELS.get(rec) == "instrument_hybrid"]
        replacements = [rec for rec in available if rec not in allowed]
        outgoing = [
            rec
            for rec in reversed(allowed)
            if rec not in protected and LIBRARY_SOURCE_LABELS.get(rec) != "instrument_hybrid"
        ]
        while len(selected) < quota and replacements and outgoing:
            old = outgoing.pop(0)
            new = replacements.pop(0)
            allowed[allowed.index(old)] = new
            selected.append(new)

    # A whole render cannot bloom if every chosen source is a short gesture.
    # Keep synth and instrument floors above, then reserve enough recordings
    # that contain genuine texture/resonance material.  The required duration
    # comes from each accepted D-level structure, not from one global preset.
    sustain_quota = min(recording_limit, max(2, int(round(recording_limit * 0.34))))
    sustained = [rec for rec in allowed if recording_can_sustain(by_recording[rec], dream_level)]
    sustained_candidates = [
        rec
        for _, rec in ranked
        if rec not in allowed and recording_can_sustain(by_recording[rec], dream_level)
    ]
    replaceable = [
        rec
        for rec in reversed(allowed)
        if rec not in protected and not recording_can_sustain(by_recording[rec], dream_level)
    ]
    while len(sustained) < sustain_quota and sustained_candidates and replaceable:
        outgoing = replaceable.pop(0)
        incoming = sustained_candidates.pop(0)
        allowed[allowed.index(outgoing)] = incoming
        sustained.append(incoming)

    print("Material plan:", allowed)
    return set(allowed)


def choose_weighted(objects, profile, dream_level, previous=None, desired_role=None, preferred_groups=None, usage_counts=None, sample_profile=None):
    global CURRENT_MATERIAL_PLAN

    if CURRENT_MATERIAL_PLAN is None:
        CURRENT_MATERIAL_PLAN = build_material_plan(objects, profile, dream_level, sample_profile)

    pool = [
        obj for obj in objects
        if obj["recording"] in CURRENT_MATERIAL_PLAN
    ]

    # Section palette: D5 should not get stuck in the same recordings,
    # but it should still sound like related families of material.
    if preferred_groups is not None:
        grouped = [obj for obj in pool if palette_group(obj["recording"]) in preferred_groups]
        if len(grouped) >= 12:
            pool = grouped

    # Role-based selection: the composer asks for a musical function.
    if desired_role is not None:
        role_pool = [obj for obj in pool if obj.get("role") == desired_role]
        if len(role_pool) >= 3:
            pool = role_pool
        else:
            # Graceful fallbacks if a role has too few objects.
            fallback_roles = {
                "impact": ["gesture", "noise"],
                "gesture": ["impact", "texture"],
                "texture": ["resonance", "gesture"],
                "resonance": ["texture"],
                "noise": ["gesture", "texture"],
            }
            fallback = []
            for role in fallback_roles.get(desired_role, []):
                fallback.extend([obj for obj in pool if obj.get("role") == role])
            if len(fallback) >= 3:
                pool = fallback

    pool = learned_synthetic_candidate_pool(pool)
    pool = diversify_overused_recordings(
        pool,
        usage_counts,
        dream_level,
        len(CURRENT_MATERIAL_PLAN),
    )

    scored = []

    for obj in pool:
        score = profile_distance(obj, profile)

        # General musical-value bias: favour clean, resonant, usable objects;
        # penalise harsh short noisy attacks.
        score *= musical_value(obj)
        score *= COMPOSITION_PREFERENCE.object_factor(
            obj.get("object_id"), obj.get("recording"), dream_level
        )
        score *= ARTIST_STYLE.object_factor(
            REPRESENTATION_ASSIST.embeddings.get(str(obj.get("object_id", ""))),
            dream_level,
        )
        score *= composition_duration_selection_factor(obj, desired_role, dream_level)

        # The D5 preference controls must make an audible selection difference,
        # while D1/D3 retain their established behaviour.
        score *= d5_selection_character_factor(obj, dream_level)
        score *= learned_synthetic_material_factor(obj)
        score *= learned_instrument_material_factor(obj)
        score *= learned_origin_mix_factor(obj, usage_counts)

        # Harmonic listening: pitched objects inside the detected scale are preferred.
        # Unpitched/noisy objects remain available and are not forcibly quantised.
        score *= scale_selection_factor(obj)

        if sample_profile is not None:
            score *= sample_learning_factor(obj, sample_profile)
            score *= sample_exploration_factor(obj, sample_profile)

        if previous is not None:
            score += continuity_distance(previous, obj) * 0.24 * learned_factor("transition_smoothness_weight", 5.0) * learned_factor("coherence_weight", 4.0)
            score *= representation_continuity_factor(previous, obj)

        # Anti-stuck behaviour: recordings can return as motifs,
        # but the composer should not cling to the same few recordings forever.
        if usage_counts is not None:
            repetition_guard = learned_factor("repetition_control", 2.0, 0.85, 1.55)
            score *= (1.0 + 0.11 * repetition_guard * usage_counts.get(obj["recording"], 0))

        dur = obj["duration"]
        role = obj.get("role", classify_object(obj))
        f = obj["features"]
        score *= long_layer_diversity_factor(
            obj, dream_level, usage_counts=usage_counts
        )

        # Memory v3 role fit: lower score for objects that compositionally fit the requested role.
        if desired_role == "resonance":
            score *= (1.18 - 0.40 * f.get("resonance_strength", 0.5) - 0.18 * f.get("harmonicity", 0.5))
        elif desired_role == "texture":
            score *= (1.14 - 0.32 * f.get("texture_score", 0.5) - 0.18 * f.get("ambient_score", 0.5))
        elif desired_role == "gesture":
            score *= (1.15 - 0.35 * f.get("gesture_strength", 0.5) - 0.12 * f.get("foreground_probability", 0.5))
        elif desired_role == "impact":
            score *= (1.12 - 0.26 * f.get("transient_score", 0.5))
        elif desired_role == "noise":
            score *= (1.10 - 0.20 * f.get("contrast_score", 0.5))

        # Role-specific preferences.
        if desired_role == "gesture":
            if 0.6 <= dur <= 4.5:
                score *= 0.60
            else:
                score *= 1.35

        elif desired_role == "impact":
            if role in ["impact", "gesture"]:
                score *= 0.55
            else:
                score *= 1.45

        elif desired_role == "texture":
            if dur >= 2.5:
                score *= 0.65
            else:
                score *= 1.30

        elif desired_role == "resonance":
            if dur >= 4.0:
                score *= 0.60
            else:
                score *= 1.35

        elif desired_role == "noise":
            if role == "noise":
                score *= 0.55
            else:
                score *= 1.25

        # Controlled curiosity: more adventurous at D5, still recognisably Hyponoia.
        if dream_level == 5:
            # Exploration remains audible, but it must not overpower continuity
            # and select an unrelated object merely because of a large random roll.
            score *= random.uniform(0.68, 1.32)
        elif dream_level == 3:
            score *= random.uniform(0.78, 1.24)
        else:
            score *= random.uniform(0.82, 1.16)

        scored.append((score, obj))

    if not scored:
        raise RuntimeError("No available objects in the current material plan.")

    scored.sort(key=lambda x: x[0])

    top_n = {1: 45, 3: 90, 5: 160}[dream_level]
    top = scored[:min(top_n, len(scored))]

    weights = selection_probabilities([score for score, _ in top])

    idx = np.random.choice(len(top), p=weights)
    return top[idx][1]


@lru_cache(maxsize=16)
def _load_recording(recording_name):
    """Load each source recording once per process instead of once per event."""
    path = os.path.join(MEMORY_FOLDER, recording_name)
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def load_fragment(obj):
    audio = _load_recording(obj["recording"])

    start = int(obj["start"] * TARGET_SR)
    end = int(obj["end"] * TARGET_SR)

    frag = audio[start:end].astype(np.float32)

    if len(frag) < 100:
        return None

    frag = frag / (np.max(np.abs(frag)) + 1e-9)
    return frag


def stretch_audio(x, factor):
    old = np.arange(len(x))
    new = np.linspace(0, len(x) - 1, max(2, int(len(x) * factor)))
    return np.interp(new, old, x).astype(np.float32)


def fade(x, sec=3.0):
    n = min(int(sec * TARGET_SR), len(x) // 2)
    if n <= 0:
        return x
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def reverse_blend(x, amount=0.35):
    return ((1 - amount) * x + amount * x[::-1]).astype(np.float32)


def butter_filter(x, mode, cutoff):
    b, a = butter(2, cutoff / (TARGET_SR / 2), btype=mode)
    return lfilter(b, a, x).astype(np.float32)


def split_frequency_stems(
    output,
    low_crossover=STEM_LOW_CROSSOVER_HZ,
    high_crossover=STEM_HIGH_CROSSOVER_HZ,
):
    """Create complementary LOW/MID/HIGH stems without changing the master."""
    audio = np.asarray(output, dtype=np.float32)
    if audio.ndim not in (1, 2):
        raise ValueError("Frequency stems require mono or stereo audio.")
    nyquist = TARGET_SR / 2.0
    low_crossover = float(low_crossover)
    high_crossover = float(high_crossover)
    if not 0.0 < low_crossover < high_crossover < nyquist:
        raise ValueError("Stem crossovers must be ordered and below Nyquist.")

    channels = audio[:, None] if audio.ndim == 1 else audio
    low = np.empty_like(channels)
    below_high = np.empty_like(channels)
    for channel in range(channels.shape[1]):
        low[:, channel] = butter_filter(
            channels[:, channel], "lowpass", low_crossover
        )
        below_high[:, channel] = butter_filter(
            channels[:, channel], "lowpass", high_crossover
        )
    mid = below_high - low
    high = channels - below_high
    if audio.ndim == 1:
        return low[:, 0], mid[:, 0], high[:, 0]
    return low, mid, high


def save_frequency_stems(output, outfile, current_file=None):
    """Save remix-ready complementary stems; the master files remain untouched."""
    low, mid, high = split_frequency_stems(output)
    timestamp_base = os.path.splitext(outfile)[0]
    timestamped = {}
    current = {}
    for label, audio in (("LOW", low), ("MID", mid), ("HIGH", high)):
        timestamped_path = f"{timestamp_base}_{label}.wav"
        # Float WAV preserves headroom and lets the three stems reconstruct the master.
        sf.write(timestamped_path, audio, TARGET_SR, subtype="FLOAT")
        timestamped[label.lower()] = timestamped_path
        if current_file:
            current_path = f"{os.path.splitext(current_file)[0]}_{label}.wav"
            sf.write(current_path, audio, TARGET_SR, subtype="FLOAT")
            current[label.lower()] = current_path
    return {
        "low_crossover_hz": STEM_LOW_CROSSOVER_HZ,
        "high_crossover_hz": STEM_HIGH_CROSSOVER_HZ,
        "format": "32-bit float WAV",
        "timestamped": timestamped,
        "current": current,
    }


def clean_band(x, low=35, high=12000):
    x = butter_filter(x, "highpass", low)
    x = butter_filter(x, "lowpass", high)
    return x


def harmonic_bloom(x, dream_level):
    out = np.zeros_like(x)

    layers = [
        (x, 1.0),
        (stretch_audio(x, 1.5)[:len(x)], 0.18),
        (stretch_audio(x, 2.0)[:len(x)], 0.12),
    ]

    if dream_level >= 3:
        layers.append((stretch_audio(x, 2.6)[:len(x)], 0.08))

    if dream_level >= 5:
        layers.append((stretch_audio(x, 3.4)[:len(x)], 0.06))

    for layer, amp in layers:
        if len(layer) < len(out):
            layer = np.pad(layer, (0, len(out) - len(layer)))
        out += layer[:len(out)] * amp

    return out / (np.max(np.abs(out)) + 1e-9)


def pan_stereo(mono, pan):
    left = mono * np.sqrt((1 - pan) / 2)
    right = mono * np.sqrt((1 + pan) / 2)
    return np.stack([left, right], axis=1)


def add_to_output(output, mono, start_sec, amp, pan):
    start = int(start_sec * TARGET_SR)
    if start >= len(output):
        return

    mono = mono[:len(output) - start]
    stereo = pan_stereo(mono * amp, pan)
    output[start:start + len(stereo)] += stereo


def arpeggio_phrase_growth(dream_level, progress):
    """Let source-derived musical phrases emerge gradually."""
    progress = max(0.0, min(1.0, float(progress)))
    floor = {1: 0.72, 3: 0.60, 5: 0.46}[dream_level]
    return float(floor + (1.0 - floor) * progress ** 0.72)


def phrase_window_envelope(length, dream_level, voice_index=0, seed=None):
    """Create long, smooth phrase windows with audible space between them."""
    length = max(0, int(length))
    if length == 0:
        return np.zeros(0, dtype=np.float32)
    rng = random.Random(
        int(RENDER_SEED if seed is None else seed)
        + int(dream_level) * 49979687
        + int(voice_index) * 67867967
    )
    phrase_count = {1: 5, 3: 6, 5: 7}[dream_level]
    duration = length / TARGET_SR
    envelope = np.zeros(length, dtype=np.float32)
    anchors = np.linspace(0.16, 0.84, phrase_count)
    for phrase_index, anchor in enumerate(anchors):
        centre = float(anchor) + rng.uniform(-0.045, 0.045)
        width = {
            1: rng.uniform(0.23, 0.34),
            3: rng.uniform(0.19, 0.29),
            5: rng.uniform(0.15, 0.25),
        }[dream_level]
        # Adjacent voices breathe at different points instead of forming one
        # permanent oscillator stack.
        centre += ((voice_index + phrase_index) % 3 - 1) * 0.018
        start = max(0, int((centre - width / 2.0) * duration * TARGET_SR))
        end = min(length, int((centre + width / 2.0) * duration * TARGET_SR))
        if end - start < 16:
            continue
        window = np.clip(
            np.sin(np.linspace(0.0, np.pi, end - start, dtype=np.float32)),
            0.0,
            1.0,
        ) ** 1.55
        envelope[start:end] = np.maximum(envelope[start:end], window)
    return envelope


def learned_bed_phrase_envelope(length, dream_level, seed=None):
    """Create finite background phrases from the learned whole-work curves.

    A continuous granular bed can sound like an editor placed a loop behind the
    composition even when its grains change. This envelope finds a small number
    of salient regions in the learned energy/density trajectories and gives each
    one an asymmetric arrival, body and release. The material therefore returns
    with memory, develops, and completes before the next phrase takes over.
    """
    length = max(0, int(length))
    if length == 0:
        return np.zeros(0, dtype=np.float32)
    grid_size = 257
    positions = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    local_seed = int(RENDER_SEED if seed is None else seed) + int(dream_level) * 433494437
    energy = ARTIST_STYLE.trajectory_curve("energy", positions, dream_level, local_seed)
    density = ARTIST_STYLE.trajectory_curve("density", positions, dream_level, local_seed)
    salience = 0.58 * np.clip(energy, 0.0, 1.0) + 0.42 * np.clip(density, 0.0, 1.0)

    phrase_count = {1: 4, 3: 5, 5: 6}[dream_level]
    candidates = [
        index
        for index in range(2, grid_size - 2)
        if salience[index] >= salience[index - 1]
        and salience[index] >= salience[index + 1]
    ]
    candidates.sort(key=lambda index: float(salience[index]), reverse=True)
    min_spacing = grid_size / (phrase_count + 1) * 0.56
    selected = []
    for index in candidates:
        if all(abs(index - other) >= min_spacing for other in selected):
            selected.append(index)
        if len(selected) >= phrase_count:
            break
    # Flat trajectories still receive a finite phrase form, but the centres are
    # spaced once across the composition rather than cycled as a periodic LFO.
    fallback = np.linspace(0.12, 0.88, phrase_count)
    for position in fallback:
        index = int(round(position * (grid_size - 1)))
        if all(abs(index - other) >= min_spacing * 0.72 for other in selected):
            selected.append(index)
        if len(selected) >= phrase_count:
            break
    selected = sorted(selected[:phrase_count])

    envelope = np.zeros(length, dtype=np.float32)
    for phrase_index, grid_index in enumerate(selected):
        centre = grid_index / (grid_size - 1)
        local_density = float(np.clip(density[grid_index], 0.0, 1.0))
        width = {1: 0.205, 3: 0.175, 5: 0.150}[dream_level]
        width *= 0.82 + 0.36 * local_density
        start = max(0, int((centre - width * 0.48) * length))
        end = min(length, int((centre + width * 0.52) * length))
        size = end - start
        if size < 32:
            continue
        attack_n = max(8, int(size * {1: 0.30, 3: 0.26, 5: 0.22}[dream_level]))
        release_n = max(8, int(size * {1: 0.43, 3: 0.46, 5: 0.49}[dream_level]))
        body_n = max(0, size - attack_n - release_n)
        attack = np.sin(np.linspace(0.0, np.pi / 2.0, attack_n, dtype=np.float32)) ** 1.35
        body = np.ones(body_n, dtype=np.float32)
        release = np.clip(
            np.cos(np.linspace(0.0, np.pi / 2.0, release_n, dtype=np.float32)),
            0.0,
            1.0,
        ) ** 1.55
        phrase = np.concatenate((attack, body, release))[:size]
        phrase_gain = 0.72 + 0.28 * float(np.clip(salience[grid_index], 0.0, 1.0))
        # Later D3/D5 phrases can answer earlier ones more confidently, without
        # forcing a simple linear crescendo on every render.
        if phrase_index and dream_level >= 3:
            phrase_gain *= 1.0 + min(0.12, 0.025 * phrase_index)
        envelope[start:end] = np.maximum(envelope[start:end], phrase * phrase_gain)
    envelope[0] = 0.0
    envelope[-1] = 0.0
    return envelope.astype(np.float32)


def source_musical_phrase_layer(source, dream_level, pulse_bpm=0.0, duration=None, seed=None):
    """Compose evolving foreground phrases by re-sampling the selected material.

    No oscillator is used. A motif is cut from an active region of the current
    source mix, then repeated with scale-related resampling, granular variation,
    overlap and long windows. This preserves the library's timbral identity while
    still producing recognisable musical phrasing and development.
    """
    audio = np.asarray(source, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Source musical phrases require stereo audio.")
    requested_length = len(audio) if duration is None else int(max(0.0, float(duration)) * TARGET_SR)
    length = min(len(audio), requested_length)
    layer = np.zeros((requested_length, 2), dtype=np.float32)
    feedback = composition_feedback_audio_snapshot()
    gain = feedback["synthetic_layer_gain"] + 0.72 * feedback["arpeggio_layer_gain"]
    gain *= np.sqrt(feedback["foreground_presence_gain"])
    if gain <= 1e-6 or length < 512:
        return layer

    working = audio[:length]
    mono = np.mean(working, axis=1)
    analysis_size = min(length, max(512, int(0.80 * TARGET_SR)))
    analysis_hop = max(128, int(0.24 * TARGET_SR))
    positions = []
    energies = []
    for start in range(0, max(1, length - analysis_size + 1), analysis_hop):
        positions.append(start)
        energies.append(float(np.sqrt(np.mean(mono[start:start + analysis_size] ** 2))))
    positive = [value for value in energies if value > 1e-8]
    if not positive:
        return layer
    floor = float(np.percentile(positive, 55.0))
    active_positions = [
        start for start, energy in zip(positions, energies) if energy >= floor
    ] or positions

    rng = random.Random(
        int(RENDER_SEED if seed is None else seed) + int(dream_level) * 104729
    )
    bpm = {
        1: 118.0,
        3: 146.0,
        5: max(158.0, float(pulse_bpm)) if pulse_bpm > 0 else 174.0,
    }[dream_level]
    step_sec = (60.0 / bpm) * {1: 0.60, 3: 0.46, 5: 0.34}[dream_level]
    # Phrase density rises clearly across the dream levels, but it remains a
    # finite formal layer rather than a continuous sequencer running everywhere.
    phrase_count = {1: 8, 3: 10, 5: 12}[dream_level]
    phrase_steps = {1: 10, 3: 12, 5: 14}[dream_level]
    interval_pool = list(SCALE_INTERVALS.get(HARMONY_STATE["scale"], (0, 3, 5, 7, 10)))
    if HARMONY_STATE["scale"] == "free" or HARMONY_STATE["confidence"] < 0.55:
        interval_pool = [0, 2, 3, 5, 7, 10]
    interval_patterns = {
        1: (0, 2, 1, 3, 0),
        3: (0, 2, 4, 1, 3, 5, 2),
        5: (0, 2, 4, 6, 3, 5, 1, 6, 4, 2),
    }
    pattern = interval_patterns[dream_level]
    anchors = np.linspace(0.13, 0.84, phrase_count)
    expressive = ARTIST_STYLE.control("expressive_drive", dream_level)
    base_amp = {1: 0.50, 3: 0.60, 5: 0.70}[dream_level] * gain * expressive

    # A small recurring motif family gives the listener something to remember.
    # Later phrases revisit and develop these fragments instead of introducing
    # one unrelated event after another and immediately dropping it.
    motif_bank = []
    motif_bank_size = {1: 4, 3: 5, 5: 6}[dream_level]
    for _ in range(motif_bank_size):
        motif_sec = rng.uniform(
            1.90 if dream_level == 1 else 1.35,
            3.40 if dream_level == 1 else (2.80 if dream_level == 3 else 2.25),
        )
        motif_size = min(length, max(512, int(motif_sec * TARGET_SR)))
        possible = [start for start in active_positions if start + motif_size <= length]
        motif_start = rng.choice(possible or [max(0, length - motif_size)])
        motif_bank.append(working[motif_start:motif_start + motif_size].copy())

    for phrase_index, anchor in enumerate(anchors):
        phrase_start = max(0.0, (float(anchor) + rng.uniform(-0.035, 0.035)) * (length / TARGET_SR))
        motif = motif_bank[phrase_index % len(motif_bank)]
        phrase_position = phrase_index / max(1, phrase_count - 1)
        learned_density = float(
            ARTIST_STYLE.trajectory_curve(
                "density", np.asarray([phrase_position]), dream_level, int(RENDER_SEED)
            )[0]
        )
        learned_energy = float(
            ARTIST_STYLE.trajectory_curve(
                "energy", np.asarray([phrase_position]), dream_level, int(RENDER_SEED)
            )[0]
        )
        local_step_sec = step_sec * (1.24 - 0.48 * np.clip(learned_density, 0.0, 1.0))

        for step in range(phrase_steps):
            degree = pattern[(step + phrase_index) % len(pattern)] % len(interval_pool)
            tonal_position = (phrase_index + step / max(1, phrase_steps)) / max(1, phrase_count)
            learned_tonal = float(
                ARTIST_STYLE.trajectory_curve(
                    "tonal", np.asarray([tonal_position]), dream_level, int(RENDER_SEED)
                )[0]
            )
            degree_shift = int(round(learned_tonal * {1: 1.0, 3: 2.0, 5: 3.0}[dream_level]))
            degree = (degree + degree_shift) % len(interval_pool)
            semitones = interval_pool[degree]
            if (step + phrase_index) % 5 == 4:
                semitones -= 12
            resample_factor = float(2.0 ** (-semitones / 12.0))
            scan = int((step / max(1, phrase_steps - 1)) * max(0, len(motif) - analysis_size))
            grain = motif[scan:].copy() if scan < len(motif) - 64 else motif.copy()
            transformed = np.stack(
                [stretch_audio(grain[:, channel], resample_factor) for channel in range(2)],
                axis=1,
            )
            for channel in range(2):
                transformed[:, channel] = structured_granulation(
                    transformed[:, channel], "gesture", dream_level
                )
            if dream_level >= 3 and (step + phrase_index) % 4 == 3:
                transformed = transformed[::-1]
            # Non-linear colour is derived from the grain itself; it does not
            # introduce an independent pitch or spectral signature.
            transformed = 0.72 * transformed + 0.28 * np.tanh(transformed * 2.1) / 2.1
            # Keep the derived phrase forward and readable when the uploaded
            # library is bass-heavy. This shelves only the added phrase layer;
            # the original source material and its full frequency range remain.
            for channel in range(2):
                phrase_low = butter_filter(transformed[:, channel], "lowpass", 170)
                low_scale = {1: 0.38, 3: 0.48, 5: 0.58}[dream_level]
                transformed[:, channel] += phrase_low * (low_scale - 1.0)
            envelope = np.clip(
                np.sin(np.linspace(0.0, np.pi, len(transformed), dtype=np.float32)),
                0.0,
                1.0,
            ) ** 1.45
            development = arpeggio_phrase_growth(
                dream_level, (phrase_index + step / phrase_steps) / max(1, phrase_count)
            )
            learned_phrase_gain = 0.72 + 0.56 * np.clip(learned_energy, 0.0, 1.0)
            transformed *= envelope[:, None] * base_amp * development * learned_phrase_gain
            max_phrase_pan = {1: 0.72, 3: 0.86, 5: 0.96}[dream_level]
            pan = max(
                -max_phrase_pan,
                min(
                    max_phrase_pan,
                    -max_phrase_pan + 2.0 * max_phrase_pan * step / max(1, phrase_steps - 1),
                ),
            )
            # Blend a small part of the pan trajectory measured from the owned
            # complete works. The generic traversal remains recognisable, but
            # no longer sweeps identically in every source-derived phrase.
            learned_pan = float(
                ARTIST_STYLE.trajectory_curve(
                    "pan", np.asarray([tonal_position]), dream_level, int(RENDER_SEED)
                )[0]
            )
            pan_blend = {1: 0.14, 3: 0.20, 5: 0.26}[dream_level]
            pan = float(np.clip(
                pan
                + np.clip(learned_pan, -1.0, 1.0) * max_phrase_pan * pan_blend,
                -max_phrase_pan,
                max_phrase_pan,
            ))
            transformed[:, 0] *= np.sqrt((1.0 - pan) / 2.0)
            transformed[:, 1] *= np.sqrt((1.0 + pan) / 2.0)
            destination = int((phrase_start + step * local_step_sec) * TARGET_SR)
            if destination >= len(layer):
                break
            end = min(len(layer), destination + len(transformed))
            layer[destination:end] += transformed[:end - destination]

    peak = float(np.max(np.abs(layer))) + 1e-9
    if peak > 0.22:
        layer *= 0.22 / peak
    return layer.astype(np.float32)


def add_source_musical_phrases(output, dream_level, pulse_bpm=0.0):
    """Add the source-derived musical layer without changing the dry events."""
    source = np.asarray(output, dtype=np.float32).copy()
    layer = source_musical_phrase_layer(
        source,
        dream_level,
        pulse_bpm=pulse_bpm,
    )
    layer *= ARTIST_STYLE.control("phrase_presence", dream_level)
    output += layer
    return layer


def movement_candidate_score(event, dream_level):
    """Score a rendered event for movement using the learned preference space."""
    obj = event.get("object", {})
    preference = COMPOSITION_PREFERENCE.object_factor(
        obj.get("object_id"), obj.get("recording"), dream_level
    )
    affinity = 1.0 / max(0.76, float(preference))
    features = obj.get("features", {})
    musicality = max(0.0, min(1.0, float(features.get("musicality", 0.5) or 0.5)))
    foreground = max(
        float(features.get("foreground_probability", 0.0) or 0.0),
        float(features.get("gesture_strength", 0.0) or 0.0),
        float(features.get("transient_score", 0.0) or 0.0),
    )
    foreground = max(0.0, min(1.0, foreground))
    role_gain = {
        "gesture": 1.28,
        "noise": 1.12,
        "impact": 1.04,
        "texture": 0.82,
        "resonance": 0.74,
    }.get(event.get("role"), 0.80)
    embedding_bonus = 1.08 if str(obj.get("object_id", "")) in COMPOSITION_PREFERENCE.embeddings else 1.0
    air_suitability = 0.18 if fragile_air_object(obj) else 1.0
    vector = REPRESENTATION_ASSIST.embeddings.get(str(obj.get("object_id", "")))
    artist_affinity = 1.0 / ARTIST_STYLE.object_factor(vector, dream_level)
    return float(
        affinity
        * artist_affinity
        * air_suitability
        * (0.58 + 0.24 * musicality + 0.28 * foreground)
        * role_gain
        * embedding_bonus
    )


def moving_pan_stereo(mono, start_pan, end_pan, motion_cycles=0.0):
    """Place one source-derived phrase on a continuous equal-power trajectory."""
    signal = np.asarray(mono, dtype=np.float32)
    if len(signal) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    progress = np.linspace(0.0, 1.0, len(signal), dtype=np.float32)
    pan = float(start_pan) + (float(end_pan) - float(start_pan)) * progress
    if motion_cycles:
        pan += 0.16 * np.sin(2.0 * np.pi * float(motion_cycles) * progress)
    pan = np.clip(pan, -0.98, 0.98)
    left = signal * np.sqrt((1.0 - pan) * 0.5)
    right = signal * np.sqrt((1.0 + pan) * 0.5)
    return np.stack((left, right), axis=1).astype(np.float32)


def artist_tonal_development(source, dream_level, position, seed):
    """Pitch-develop a return from a learned tonal trajectory, preserving length."""
    audio = np.asarray(source, dtype=np.float32)
    if len(audio) < 512 or not ARTIST_STYLE.active:
        return audio
    tonal = float(
        ARTIST_STYLE.trajectory_curve(
            "tonal", np.asarray([position]), dream_level, int(seed)
        )[0]
    )
    semitones = tonal * {1: 1.6, 3: 3.0, 5: 5.0}[dream_level]
    semitones *= ARTIST_STYLE.control("tonal_development", dream_level)
    if abs(semitones) < 0.35:
        return audio
    shifted = librosa.effects.pitch_shift(
        audio,
        sr=TARGET_SR,
        n_steps=float(np.clip(semitones, -6.0, 6.0)),
        res_type="soxr_hq",
    )
    if len(shifted) != len(audio):
        shifted = librosa.util.fix_length(shifted, size=len(audio))
    return np.asarray(shifted, dtype=np.float32)


def learned_form_context(position, dream_level, section_name=None, seed=None):
    """Measure whether a formal point calls for audible material development.

    The decision follows the energy/density trajectories learned from the owned
    works. Local motion is weighted more than absolute loudness, so a return is
    developed at an articulation, growth or culmination point—not simply because
    it is the next repetition. Section weights only keep openings and resolutions
    from being overworked; they do not prescribe a fixed composition template.
    """
    position = float(np.clip(position, 0.0, 1.0))
    radius = {1: 0.055, 3: 0.045, 5: 0.035}[dream_level]
    points = np.asarray(
        [max(0.0, position - radius), position, min(1.0, position + radius)],
        dtype=np.float64,
    )
    trajectory_seed = int(RENDER_SEED if seed is None else seed)
    energy = ARTIST_STYLE.trajectory_curve(
        "energy", points, dream_level, trajectory_seed
    )
    density = ARTIST_STYLE.trajectory_curve(
        "density", points, dream_level, trajectory_seed
    )
    centre = 0.55 * float(energy[1]) + 0.45 * float(density[1])
    motion = 0.55 * abs(float(energy[2] - energy[0]))
    motion += 0.45 * abs(float(density[2] - density[0]))
    score = 0.46 * np.clip(centre, 0.0, 1.0) + 0.54 * np.clip(motion * 3.2, 0.0, 1.0)
    section_scale = {
        "opening": 0.78,
        "activation": 1.04,
        "complexity": 1.18,
        "memory": 1.02,
        "resolution": 0.74,
    }.get(section_name, 1.0)
    return float(np.clip(score * section_scale, 0.0, 1.0))


def musical_evolution_moment(position, dream_level, event_index=0, section_name=None, seed=None):
    """Gate strong transformations to sparse, learned formal moments."""
    context = learned_form_context(position, dream_level, section_name, seed)
    threshold = {1: 0.48, 3: 0.43, 5: 0.38}[dream_level]
    cadence = {1: 4, 3: 3, 5: 2}[dream_level]
    structural_slot = int(event_index) % cadence == cadence - 1
    return bool(structural_slot and context >= threshold), context


def electroacoustic_movement_layer(source, events, dream_level, seed=None):
    """Develop learned-preference events into recurrent electroacoustic motion.

    The layer introduces no oscillator or stock gesture. Candidate material is
    selected in the learned embedding/preference space, then re-sampled into
    granular, spatially moving returns. The renderer supplies the DSP grammar;
    learned affinity, user feedback and the uploaded material decide its content
    and strength.
    """
    audio = np.asarray(source, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 512 or not events:
        return np.zeros_like(audio)
    source_rms = float(np.sqrt(np.mean(audio * audio)))
    if source_rms < 1e-7:
        return np.zeros_like(audio)
    duration_sec = len(audio) / TARGET_SR

    rng = random.Random(int(RENDER_SEED if seed is None else seed) + dream_level * 961748941)
    scored = sorted(
        ((movement_candidate_score(event, dream_level), event) for event in events),
        key=lambda item: item[0],
        reverse=True,
    )
    # Explore within the learned high-affinity region rather than taking the
    # exact same top event on every render.
    pool = scored[:min(len(scored), {1: 14, 3: 20, 5: 28}[dream_level])]
    selected = []
    desired = min(len(pool), {1: 6, 3: 9, 5: 12}[dream_level])
    while pool and len(selected) < desired:
        weights = [max(1e-5, score) for score, _ in pool]
        chosen = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        selected.append(pool.pop(chosen))

    layer = np.zeros_like(audio)
    feedback = composition_feedback_audio_snapshot()
    musicality = learned_factor("musicality_weight", 2.6, 0.86, 1.42)
    development = feedback["development_drive"]
    granular = feedback["structured_granulation_drive"]
    repeat_control = learned_factor("repetition_control", 2.2, 0.72, 1.45)
    recurrence = ARTIST_STYLE.control("recurrence", dream_level)
    phrase_persistence = ARTIST_STYLE.control("phrase_persistence", dream_level)
    style_granulation = ARTIST_STYLE.control("granulation", dream_level)
    return_count = {
        1: 2 + int(repeat_control < 1.02),
        3: 2 + int(repeat_control < 1.12),
        5: 3 + int(repeat_control < 1.18),
    }[dream_level]
    return_count += {1: 0, 3: 1, 5: 2}[dream_level]
    return_count = min({1: 3, 3: 4, 5: 5}[dream_level], max(2, int(round(return_count * recurrence))))
    spatial = np.sqrt(
        ARTIST_STYLE.control("spatial_width", dream_level)
        * ARTIST_STYLE.control("pan_motion", dream_level)
    )
    max_pan = min(0.98, {1: 0.78, 3: 0.92, 5: 0.98}[dream_level] * spatial)

    for candidate_index, (score, event) in enumerate(selected):
        start_sample = max(0, min(len(audio) - 1, int(float(event["start"]) * TARGET_SR)))
        available = max(512, min(len(audio) - start_sample, int(float(event["duration"]) * TARGET_SR)))
        source_size = min(
            available,
            int(rng.uniform(1.25, {1: 3.2, 3: 2.8, 5: 2.4}[dream_level]) * TARGET_SR),
        )
        if source_size < 512:
            continue
        fragment = np.mean(audio[start_sample:start_sample + source_size], axis=1)
        if float(np.sqrt(np.mean(fragment * fragment))) < 1e-7:
            continue

        stable_id = str(event.get("object", {}).get("object_id", ""))
        identity_seed = sum((index + 1) * ord(char) for index, char in enumerate(stable_id))
        local = random.Random(rng.randrange(1, 2**31 - 1) + identity_seed)
        first_destination = max(
            0.0,
            min(
                max(0.0, duration_sec - 0.05),
                float(event["start"]) + local.uniform(-2.8, 4.8),
            ),
        )
        direction = -1.0 if local.random() < 0.5 else 1.0

        candidate_position = candidate_index / max(1, len(selected) - 1)
        context = learned_form_context(
            candidate_position,
            dream_level,
            "complexity" if 0.30 <= candidate_position <= 0.68 else "memory",
            identity_seed,
        )
        # The learned trajectory controls compression and rarefaction. Even D5
        # has a finite number of returns, and low-context material may appear
        # only twice before leaving space for another phrase.
        local_return_count = int(round(2 + context * (return_count - 2)))
        local_return_count = max(2, min(return_count, local_return_count))
        development_points = {local_return_count - 1}
        if dream_level == 5 and local_return_count >= 5 and context >= 0.76:
            development_points.add(local_return_count // 2)

        for occurrence in range(local_return_count):
            stretch = local.uniform(
                {1: 0.62, 3: 0.54, 5: 0.46}[dream_level],
                {1: 1.52, 3: 1.38, 5: 1.26}[dream_level],
            ) * phrase_persistence
            stretch *= 1.0 + 0.10 * occurrence * (development - 0.8)
            gesture = stretch_audio(fragment.copy(), stretch)
            trajectory_position = (
                candidate_index + occurrence / max(1, local_return_count)
            ) / max(1, len(selected))
            evolution_moment = occurrence in development_points and context >= {
                1: 0.48, 3: 0.43, 5: 0.38
            }[dream_level]
            if evolution_moment:
                gesture = artist_tonal_development(
                    gesture,
                    dream_level,
                    trajectory_position,
                    identity_seed + occurrence,
                )
            if evolution_moment and occurrence and local.random() < 0.34 + 0.08 * dream_level:
                gesture = reverse_blend(gesture, amount=local.uniform(0.14, 0.38))
            # First and intermediate returns retain the motif clearly. The
            # developed return receives the stronger granular/spectral change.
            if evolution_moment:
                gesture = structured_granulation(gesture, "gesture", dream_level)
                gesture = learned_material_evolution(gesture, "gesture", dream_level)
            gesture_low = butter_filter(gesture, "lowpass", 190)
            gesture += gesture_low * ({1: 0.42, 3: 0.50, 5: 0.60}[dream_level] - 1.0)

            # Several overlapping source scans create a compact polyphonic
            # movement. Their number follows learned granulation, not a preset.
            effective_granular = granular * style_granulation
            voice_count = min(
                4,
                1
                + int(effective_granular > 1.10)
                + int(effective_granular > 1.42)
                + int(dream_level == 5),
            )
            movement = np.zeros(len(gesture) + int(0.55 * TARGET_SR), dtype=np.float32)
            for voice in range(voice_count):
                factor = 1.0 + local.uniform(-0.075, 0.095) * (voice + 1)
                voice_audio = stretch_audio(gesture, factor)
                offset = int(voice * local.uniform(0.07, 0.19) * TARGET_SR)
                end = min(len(movement), offset + len(voice_audio))
                movement[offset:end] += voice_audio[:end - offset] / np.sqrt(voice_count)
            movement = continuity_edge_guard(movement, "gesture", dream_level)
            peak = float(np.max(np.abs(movement))) + 1e-9
            movement /= peak

            pan_start = direction * max_pan * local.uniform(0.56, 1.0)
            pan_end = -pan_start * local.uniform(0.72, 1.0)
            pan_positions = np.asarray(
                [trajectory_position, min(1.0, trajectory_position + 0.08)],
                dtype=np.float64,
            )
            learned_pan = ARTIST_STYLE.trajectory_curve(
                "pan", pan_positions, dream_level, identity_seed + occurrence
            )
            pan_blend = {1: 0.14, 3: 0.20, 5: 0.26}[dream_level]
            pan_start = float(np.clip(
                pan_start + float(learned_pan[0]) * max_pan * pan_blend,
                -max_pan,
                max_pan,
            ))
            pan_end = float(np.clip(
                pan_end + float(learned_pan[1]) * max_pan * pan_blend,
                -max_pan,
                max_pan,
            ))
            stereo = moving_pan_stereo(
                movement,
                pan_start,
                pan_end,
                motion_cycles=local.uniform(
                    {1: 0.55, 3: 0.90, 5: 1.35}[dream_level],
                    {1: 1.05, 3: 1.60, 5: 2.30}[dream_level],
                ) * ARTIST_STYLE.control("pan_motion", dream_level),
            )
            affinity_gain = max(0.78, min(1.24, float(score)))
            gain = {1: 0.060, 3: 0.085, 5: 0.115}[dream_level]
            gain *= ARTIST_STYLE.control("expressive_drive", dream_level)
            gain *= affinity_gain * np.sqrt(musicality) * (1.0 + 0.10 * occurrence)
            destination_sec = first_destination + occurrence * local.uniform(
                {1: 10.0, 3: 7.0, 5: 4.5}[dream_level],
                {1: 19.0, 3: 14.0, 5: 10.0}[dream_level],
            )
            destination = int(destination_sec * TARGET_SR)
            if destination >= len(layer):
                break
            end = min(len(layer), destination + len(stereo))
            layer[destination:end] += stereo[:end - destination] * gain
            direction *= -1.0

    layer_rms = float(np.sqrt(np.mean(layer * layer))) + 1e-9
    target_rms = source_rms * {1: 0.22, 3: 0.32, 5: 0.44}[dream_level]
    target_rms *= np.sqrt(musicality * max(0.85, min(1.42, development)))
    target_rms *= ARTIST_STYLE.control("phrase_presence", dream_level)
    if layer_rms > 1e-8:
        layer *= min(1.85, target_rms / layer_rms)
    peak = float(np.max(np.abs(layer))) + 1e-9
    if peak > 0.28:
        layer *= 0.28 / peak
    return layer.astype(np.float32)



def smooth_tail(x, tail_sec=0.45, decay=0.45):
    """Add a very short decaying self-tail so phrases do not stop abruptly."""
    delay = max(1, int(tail_sec * TARGET_SR))
    out = np.zeros(len(x) + delay, dtype=np.float32)
    out[:len(x)] += x
    out[delay:delay + len(x)] += x * decay
    return out / (np.max(np.abs(out)) + 1e-9)


def simple_stereo_reverb(output, wet=0.09):
    """Small algorithmic glue reverb: subtle space, not a washy effect."""
    if wet <= 0:
        return output

    delays_l = [0.031, 0.047, 0.073, 0.109]
    delays_r = [0.037, 0.053, 0.079, 0.127]
    gains = [0.28, 0.20, 0.14, 0.10]

    rev = np.zeros_like(output)
    for delay, gain in zip(delays_l, gains):
        n = int(delay * TARGET_SR)
        rev[n:, 0] += output[:-n, 0] * gain
    for delay, gain in zip(delays_r, gains):
        n = int(delay * TARGET_SR)
        rev[n:, 1] += output[:-n, 1] * gain

    # soft diffusion / damping through feedback-like repeated delay copies
    for delay, gain in [(0.163, 0.055), (0.211, 0.040), (0.293, 0.026)]:
        n = int(delay * TARGET_SR)
        rev[n:, 0] += rev[:-n, 1] * gain
        rev[n:, 1] += rev[:-n, 0] * gain

    # Keep the shared space out of the bass/low-mid region. This preserves body
    # in the dry signal while preventing long layers from turning into a wash.
    for channel in range(2):
        reverberant_bass = butter_filter(rev[:, channel], "lowpass", 180)
        reverberant_low_mid = (
            butter_filter(rev[:, channel], "lowpass", 650) - reverberant_bass
        )
        rev[:, channel] -= reverberant_bass * 0.72
        rev[:, channel] -= reverberant_low_mid * 0.22
        rev[:, channel] = butter_filter(rev[:, channel], "lowpass", 9500)

    return (output * (1.0 - wet) + rev * wet).astype(np.float32)


def base_reverb_wet(dream_level):
    """Clearer level-specific room amount before listener clarity feedback."""
    return {1: 0.075, 3: 0.100, 5: 0.125}[dream_level]


def parallel_density_glue(output, dream_level):
    """Make higher dream levels fuller without flattening their transients."""
    source = np.asarray(output, dtype=np.float32)
    richness = learned_factor("richness_weight", 2.5, 0.90, 1.45)
    activity = learned_factor("activity_weight", 1.8, 0.92, 1.42)
    base_mix = {1: 0.24, 3: 0.34, 5: 0.50}[dream_level]
    requested = max(0.0, richness - 1.0) + 0.65 * max(0.0, activity - 1.0)
    wet = min(0.58, base_mix + 0.16 * requested)
    drive = {1: 2.10, 3: 2.50, 5: 3.30}[dream_level]
    compressed = np.tanh(source * drive) / np.tanh(drive)
    return (source * (1.0 - wet) + compressed * wet).astype(np.float32)


def repair_isolated_discontinuities(audio, threshold=0.075, ratio=9.0):
    """Repair isolated digital seams while preserving real bright transients.

    A click is a single derivative far larger than its immediate neighbourhood.
    Sustained high-frequency synth material has many neighbouring large slopes
    and is therefore left untouched.
    """
    source = np.asarray(audio, dtype=np.float32)
    mono_input = source.ndim == 1
    work = source[:, None].copy() if mono_input else source.copy()
    if len(work) < 256:
        return source.copy()

    radius = 24
    repair_span = max(24, int(0.0015 * TARGET_SR))
    for channel in range(work.shape[1]):
        signal = work[:, channel]
        derivative = np.diff(signal)
        magnitude = np.abs(derivative)
        padded = np.pad(magnitude, (radius, radius), mode="edge")
        cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
        window = 2 * radius + 1
        local_sum = cumulative[window:] - cumulative[:-window]
        neighbourhood = np.maximum(
            1e-5,
            (local_sum - magnitude) / max(1, window - 1),
        )
        candidates = np.flatnonzero(
            (magnitude > float(threshold))
            & (magnitude > float(ratio) * neighbourhood)
        )
        last_right = -1
        for index in candidates:
            left = max(1, int(index) - repair_span)
            right = min(len(signal) - 2, int(index) + 1 + repair_span)
            if left <= last_right or right - left < 8:
                continue
            count = right - left + 1
            t = np.linspace(0.0, 1.0, count, dtype=np.float32)
            y0, y1 = float(signal[left]), float(signal[right])
            slope0 = float(signal[left] - signal[left - 1]) * count
            slope1 = float(signal[right + 1] - signal[right]) * count
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            repaired = h00 * y0 + h10 * slope0 + h01 * y1 + h11 * slope1
            # Cubic Hermite slopes can overshoot far beyond the actual local
            # waveform. That artificial peak would make the delivery ceiling
            # turn an energetic composition down. Bound only the replacement
            # samples to the real neighbourhood; legitimate source transients
            # outside the isolated seam remain completely untouched.
            context_left = max(0, left - radius)
            context_right = min(len(signal), right + radius + 1)
            context = signal[context_left:context_right]
            local_min = float(np.min(context))
            local_max = float(np.max(context))
            signal[left:right + 1] = np.clip(repaired, local_min, local_max)
            last_right = right
        work[:, channel] = signal
    return work[:, 0] if mono_input else work


def enforce_delivery_ceiling(output, ceiling=0.88):
    """Keep the final repaired waveform below the delivery peak ceiling."""
    work = np.asarray(output, dtype=np.float32).copy()
    peak = float(np.max(np.abs(work))) if work.size else 0.0
    if peak > float(ceiling):
        work *= float(ceiling) / peak
    return work


def audition_loudness_floor(output, dream_level, ceiling=0.88):
    """Make D-level energy comparable without changing source selection.

    The composition and its internal dynamics are created upstream. This final
    bounded soft lift only prevents one isolated peak from turning an otherwise
    active D5 render quieter than D1/D3 after peak normalisation.
    """
    source = np.asarray(output, dtype=np.float32)
    rms = float(np.sqrt(np.mean(source * source))) if source.size else 0.0
    target = {1: 0.072, 3: 0.100, 5: 0.130}[int(dream_level)]
    if rms < 1e-8 or rms >= target:
        return enforce_delivery_ceiling(source, ceiling)

    low, high = 1.0, 16.0
    best = source.copy()
    for _ in range(16):
        gain = 0.5 * (low + high)
        candidate = np.tanh(source * gain / ceiling) * ceiling
        candidate_rms = float(np.sqrt(np.mean(candidate * candidate)))
        best = candidate.astype(np.float32)
        if candidate_rms < target:
            low = gain
        else:
            high = gain
    return enforce_delivery_ceiling(best, ceiling)


def final_mix(output, dream_level):
    # Global polish. No fixed high or low oscillator is injected: the spectral
    # identity comes from the selected recordings and phrase-shaped synth events.
    temporal = d5_temporal_profile(dream_level)
    ambient_scale = temporal["ambient_scale"]
    if dream_level == 5:
        ambient_scale *= max(0.76, 1.0 - 0.46 * (d5_energy_drive(5) - 1.0))

    output -= np.mean(output, axis=0)
    output = apply_mix_feedback_controls(output)
    output = adaptive_low_balance(output, dream_level)

    # Gentle glue reverb. A little more in D5, but still controlled.
    wet = base_reverb_wet(dream_level) * ambient_scale
    wet *= composition_feedback_audio_snapshot()["reverb_clarity_gain"]
    output = simple_stereo_reverb(output, wet=wet)
    output = parallel_density_glue(output, dream_level)
    # Saturation and parallel glue can rebuild low dominance after the first
    # correction, so verify the balance once more before delivery gain.
    output = adaptive_low_balance(output, dream_level)

    # Soft saturation for body; level differentiation comes from the source-
    # derived composition layers rather than a different distortion preset.
    output = np.tanh(output * 1.14)
    output = repair_isolated_discontinuities(output)

    peak = np.max(np.abs(output)) + 1e-9
    output = output / peak * 0.88

    if dream_level == 5:
        drive = d5_energy_drive(5)
        fade_in = int(max(3.2, 5.0 / drive) * TARGET_SR)
        fade_out = int(max(17.0, 24.0 / np.sqrt(drive)) * TARGET_SR)
    else:
        fade_in = int(6 * TARGET_SR)
        fade_out = int(38 * TARGET_SR)

    output[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    output[-fade_out:] *= np.linspace(1, 0, fade_out)[:, None]

    # A seam that was quiet before normalisation can become audible after the
    # final gain and form envelope, so verify delivery-level samples once more.
    output = repair_isolated_discontinuities(output, threshold=0.06, ratio=8.0)
    output = audition_loudness_floor(output, dream_level)
    # The loudness lift can make a previously inaudible one-sample seam cross
    # the delivery threshold, so perform one final isolated-seam check.
    output = repair_isolated_discontinuities(output, threshold=0.06, ratio=8.0)
    # Seam repair may lower a dense render; restore only the bounded audition
    # floor after all discontinuity work is complete.
    output = audition_loudness_floor(output, dream_level)
    # The second bounded lift can expose a seam that was below the threshold by
    # only a few hundredths. Repair only isolated one-sample discontinuities;
    # sustained bright/transient material remains explicitly protected by the
    # neighbourhood-ratio test in repair_isolated_discontinuities.
    output = repair_isolated_discontinuities(output, threshold=0.06, ratio=8.0)
    # Interpolation can overshoot the earlier normalisation by a few samples.
    # Re-apply the ceiling after repair so exported PCM stays below full scale.
    output = enforce_delivery_ceiling(output)

    return output.astype(np.float32)


def role_sequence_for_section(section_name, dream_level, remaining_role_counts=None):
    """
    Role probability tables with a stronger sense of family.
    This version favours ambience, recurrence, and musical continuity;
    noise/impact are punctuation, not constant new material.
    """
    if section_name == "opening":
        table = [("texture", 0.48), ("resonance", 0.42), ("gesture", 0.10)]
    elif section_name == "activation":
        table = [("texture", 0.36), ("gesture", 0.30), ("resonance", 0.22), ("noise", 0.08), ("impact", 0.04)]
    elif section_name == "complexity":
        table = [("gesture", 0.32), ("texture", 0.30), ("resonance", 0.22), ("noise", 0.11), ("impact", 0.05)]
    elif section_name == "memory":
        table = [("resonance", 0.42), ("texture", 0.34), ("gesture", 0.18), ("noise", 0.06)]
    else:  # resolution
        table = [("resonance", 0.52), ("texture", 0.36), ("gesture", 0.09), ("noise", 0.03)]

    if dream_level == 1:
        # D1 should remain simple but not empty.
        table = [(role, weight * (1.08 if role in ["texture", "resonance"] else 0.92)) for role, weight in table]
    elif dream_level == 5:
        # D5: richer, but not cluttered. More phrase activity and resonance;
        # noise/impact stay as punctuation.
        drive = d5_energy_drive(dream_level)
        factors = {
            "gesture": 1.40 * drive,
            "texture": 1.16 / (drive ** 0.18),
            "resonance": 1.12 / (drive ** 0.12),
            "noise": 0.78,
            "impact": 0.80 * (0.92 + 0.08 * drive),
        }
        table = [(role, weight * factors.get(role, 1.0)) for role, weight in table]
        table = [(role, weight * random.uniform(0.92, 1.10)) for role, weight in table]
    else:
        table = [(role, weight * random.uniform(0.90, 1.14)) for role, weight in table]

    role_learning = {
        "gesture": learned_factor("gesture_weight", 5.0),
        "texture": learned_factor("richness_weight", 3.5) * learned_factor("ambient_weight", 2.5),
        "resonance": learned_factor("ambient_weight", 4.0) * learned_factor("coherence_weight", 2.0),
        "noise": 1.0 / learned_factor("noise_penalty", 5.0),
        "impact": 1.0 / learned_factor("impact_penalty", 5.0),
    }
    table = [
        (
            role,
            weight
            * role_learning.get(role, 1.0)
            * COMPOSITION_PREFERENCE.role_factor(role, dream_level),
        )
        for role, weight in table
    ]

    if remaining_role_counts is not None:
        present = {role for role, _ in table}
        table.extend(
            (role, 0.02)
            for role in ("gesture", "texture", "resonance", "noise", "impact")
            if role not in present and remaining_role_counts.get(role, 0) > 0
        )

    roles = [r for r, _ in table]
    weights = np.array([w for _, w in table], dtype=np.float64)
    if remaining_role_counts is not None:
        remaining = np.asarray(
            [max(0, int(remaining_role_counts.get(role, 0))) for role in roles],
            dtype=np.float64,
        )
        if float(remaining.sum()) > 0.0:
            weights *= remaining
    weights /= weights.sum()

    idx = np.random.choice(len(roles), p=weights)
    selected = roles[idx]
    if remaining_role_counts is not None and remaining_role_counts.get(selected, 0) > 0:
        remaining_role_counts[selected] -= 1
    return selected


def maybe_variation_transform(frag, role, dream_level):
    """Small variation for repeated motifs so recurrence feels musical, not copy-paste."""
    temporal = d5_temporal_profile(dream_level)
    development = d5_development_drive(dream_level)
    if role in ["gesture", "impact"]:
        if dream_level == 5:
            factor = random.uniform(0.88, 1.08)
            factor *= temporal["stretch_scale"] ** 0.45
        else:
            factor = random.uniform(0.92, 1.12)
        frag = stretch_audio(frag, factor)
        reverse_chance = 0.18 * development if dream_level == 5 else 0.18
        if random.random() < reverse_chance:
            reverse_max = 0.18 if dream_level == 5 else 0.16
            frag = reverse_blend(frag, amount=random.uniform(0.06, reverse_max))
        fade_time = random.uniform(0.22, 0.75) * temporal["envelope_scale"] if dream_level == 5 else random.uniform(0.25, 0.90)
        frag = fade(frag, sec=fade_time)
    elif role in ["texture", "resonance"]:
        if dream_level == 5:
            factor = random.uniform(0.96, 1.28)
            factor *= temporal["stretch_scale"] ** 0.35
        else:
            factor = random.uniform(1.02, 1.35)
        frag = stretch_audio(frag, factor)
        fade_time = random.uniform(2.0, 5.0) * temporal["envelope_scale"] if dream_level == 5 else random.uniform(2.5, 6.0)
        frag = fade(frag, sec=fade_time)
    else:
        if dream_level == 5:
            factor = random.uniform(0.90, 1.14)
            factor *= temporal["stretch_scale"] ** 0.40
        else:
            factor = random.uniform(0.95, 1.20)
        frag = stretch_audio(frag, factor)
        fade_time = random.uniform(0.7, 1.8) * temporal["envelope_scale"] if dream_level == 5 else random.uniform(0.8, 2.2)
        frag = fade(frag, sec=fade_time)
    return frag.astype(np.float32)


def d5_internal_motion(frag, role, dream_level, section_name=None):
    """Develop D5 material with smooth, role-aware internal rhythmic motion."""
    if dream_level != 5 or len(frag) < 64:
        return frag.astype(np.float32)

    temporal_drive = d5_temporal_profile(dream_level)["temporal_drive"]
    development = d5_development_drive(dream_level)
    section_rate = {
        "opening": 0.72,
        "activation": 1.00,
        "complexity": 1.24,
        "memory": 0.88,
        "resolution": 0.68,
    }.get(section_name, 1.0)
    if role in ["gesture", "impact"]:
        rate = random.uniform(1.8, 3.6) * temporal_drive
        depth = min(0.34, 0.18 + 0.12 * (development - 1.0))
    elif role == "noise":
        rate = random.uniform(1.4, 2.8) * temporal_drive
        depth = min(0.30, 0.16 + 0.10 * (development - 1.0))
    elif role == "texture":
        rate = random.uniform(0.30, 0.78) * temporal_drive
        depth = min(0.25, 0.13 + 0.11 * (development - 1.0))
    else:  # resonance
        rate = random.uniform(0.12, 0.36) * temporal_drive
        depth = min(0.20, 0.10 + 0.08 * (development - 1.0))

    rate *= section_rate
    phase = random.uniform(0.0, 2.0 * np.pi)
    time = np.arange(len(frag), dtype=np.float32) / TARGET_SR
    pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * rate * time + phase)
    pulse = np.power(pulse, 1.35)
    motion = (1.0 - depth) + depth * pulse
    phrase_curve = d5_development_curve(len(frag), section_name)

    # Let the spectral detail open and close with the learned phrase trajectory.
    # This is audible material development, not merely a louder amplitude curve.
    low = butter_filter(frag, "lowpass", 2400 if role in {"gesture", "noise", "impact"} else 1700)
    detail = frag - low
    detail_gain = np.clip(0.48 + 0.62 * phrase_curve, 0.42, 1.30)
    spectral = low + detail * detail_gain
    spectral_mix = min(0.68, max(0.0, (development - 1.0) * 1.55))
    developed = frag * (1.0 - spectral_mix) + spectral * spectral_mix
    result = developed * motion.astype(np.float32) * phrase_curve
    source_peak = float(np.max(np.abs(frag))) + 1e-9
    result_peak = float(np.max(np.abs(result))) + 1e-9
    if result_peak > source_peak * 1.18:
        result *= (source_peak * 1.18) / result_peak
    return result.astype(np.float32)


def continuity_edge_guard(frag, role, dream_level):
    """Guarantee musical attacks/releases after every transformation stage."""
    if len(frag) < 32:
        return frag.astype(np.float32)

    attack_sec = {
        "gesture": 0.10,
        "impact": 0.07,
        "noise": 0.18,
        "texture": 0.48,
        "resonance": 0.75,
    }.get(role, 0.20)
    release_sec = {
        "gesture": 0.95,
        "impact": 0.62,
        "noise": 1.35,
        "texture": 3.20,
        "resonance": 4.80,
    }.get(role, 1.20)
    level_scale = {1: 0.58, 3: 0.78, 5: 1.0}[dream_level]
    smoothness = learned_factor("transition_smoothness_weight", 1.8, 0.90, 1.32)
    smoothness *= level_scale
    attack_n = min(int(attack_sec * smoothness * TARGET_SR), len(frag) // 3)
    release_n = min(int(release_sec * smoothness * TARGET_SR), len(frag) // 2)
    if attack_n > 8:
        attack = np.sin(np.linspace(0.0, np.pi / 2.0, attack_n)) ** 1.25
        frag[:attack_n] *= attack.astype(np.float32)
    if release_n > 8:
        release = np.cos(np.linspace(0.0, np.pi / 2.0, release_n)) ** 1.45
        frag[-release_n:] *= release.astype(np.float32)
    frag[0] = 0.0
    frag[-1] = 0.0
    return frag.astype(np.float32)


def feedback_release_guard(frag, role):
    """Add an audible, role-aware release only after explicit smoothness feedback."""
    if len(frag) < 32:
        return frag.astype(np.float32)
    smoothness = composition_feedback_audio_snapshot()["release_smoothness"]
    request = max(0.0, smoothness - 1.0)
    if request <= 1e-6:
        return frag.astype(np.float32)

    base_release = {
        "gesture": 0.85,
        "impact": 0.60,
        "noise": 1.45,
        "texture": 3.20,
        "resonance": 4.80,
    }.get(role, 1.20)
    release_sec = base_release * (0.38 + 1.95 * request)
    release_n = min(int(release_sec * TARGET_SR), len(frag) // 2)
    if release_n > 8:
        curve = np.cos(np.linspace(0.0, np.pi / 2.0, release_n)) ** (1.18 + 0.45 * request)
        frag[-release_n:] *= curve.astype(np.float32)
    frag[-1] = 0.0
    return frag.astype(np.float32)


def feedback_release_tail(frag, role):
    """Carry a departing layer into damped echoes so it leaves through continuity."""
    if len(frag) < 32:
        return frag.astype(np.float32)
    smoothness = composition_feedback_audio_snapshot()["release_smoothness"]
    request = max(0.0, smoothness - 1.0)
    if request <= 1e-6:
        return frag.astype(np.float32)

    base_tail_sec = {
        "gesture": 1.00,
        "impact": 0.75,
        "noise": 2.00,
        "texture": 3.50,
        "resonance": 5.00,
    }.get(role, 1.50)
    tail_sec = base_tail_sec * (0.55 + 1.10 * request)
    delay_seconds = (tail_sec * 0.22, tail_sec * 0.55, tail_sec)
    max_delay = max(1, int(delay_seconds[-1] * TARGET_SR))
    output = np.zeros(len(frag) + max_delay, dtype=np.float32)
    output[:len(frag)] += frag

    cutoff = 7200 if role in ("gesture", "impact") else 4800
    damped = butter_filter(frag, "lowpass", cutoff)
    gain_scale = 0.65 + 0.70 * request
    for delay_sec, gain in zip(delay_seconds, (0.16, 0.10, 0.055)):
        delay = max(1, int(delay_sec * TARGET_SR))
        output[delay:delay + len(damped)] += damped * gain * gain_scale
    return output.astype(np.float32)


def feedback_continuity_start(start, duration, role, previous_layer_end):
    """Overlap the next related layer after explicit complaints about abrupt exits."""
    if previous_layer_end is None:
        return float(start)
    smoothness = composition_feedback_audio_snapshot()["release_smoothness"]
    request = max(0.0, smoothness - 1.0)
    if request <= 1e-6:
        return float(start)

    gap = float(start) - float(previous_layer_end)
    desired_overlap = {
        "gesture": 0.34,
        "impact": 0.22,
        "noise": 0.58,
        "texture": 2.10,
        "resonance": 3.10,
    }.get(role, 0.45) * request
    if gap >= 0.0:
        # Nearby phrases crossfade. Larger gaps retain breathing room but become
        # less empty, so continuity never turns into a constant wall of sound.
        if gap <= 7.0:
            adjusted = float(start) - min(gap + desired_overlap, max(0.25, duration * 0.22))
        else:
            adjusted = float(start) - min(gap * 0.30 * request, 3.2)
        return max(0.0, adjusted)
    return float(start)


def bound_sustained_fragment(frag, role, dream_level):
    """Limit dominance and use a different sustained window at each D-level."""
    if role not in {"texture", "resonance"}:
        return frag
    rotation = composition_feedback_audio_snapshot()["long_layer_rotation"]
    request = max(0.0, rotation - 1.0)
    if request <= 1e-6:
        return frag

    base_cap = {
        "texture": {1: 15.0, 3: 14.0, 5: 13.0},
        "resonance": {1: 22.0, 3: 20.0, 5: 18.0},
    }[role][dream_level]
    cap_scale = 1.0 - 0.10 * min(1.0, request / 0.5)
    max_samples = max(32, int(base_cap * cap_scale * TARGET_SR))
    if len(frag) <= max_samples:
        return frag

    spare = len(frag) - max_samples
    window_position = {1: 0.12, 3: 0.48, 5: 0.78}[dream_level]
    start = int(spare * window_position)
    return frag[start:start + max_samples].copy()


def structured_granulation(frag, role, dream_level):
    """Add bounded, phrase-following grains after explicit listener feedback.

    Grain positions move forward through the source on a regular hop, with a
    small deterministic drift. This produces audible granular articulation
    without turning the material into an unrelated random spray. The dry signal
    always remains present and output length/peak are preserved.
    """
    snapshot = composition_feedback_audio_snapshot()
    wet = snapshot["structured_granulation_wet"]
    eligible_roles = {"gesture", "texture", "noise", "resonance"}
    if wet <= 1e-6 or role not in eligible_roles or len(frag) < 512:
        return np.asarray(frag, dtype=np.float32)

    if role == "resonance":
        # Resonant beds evolve too, but remain recognisable and never become a
        # full-strength granular wash.
        role_scale = {1: 0.38, 3: 0.48, 5: 0.62}[dream_level]
    else:
        role_scale = {"gesture": 0.88, "texture": 1.0, "noise": 0.76}[role]
    style_granulation = ARTIST_STYLE.control("granulation", dream_level)
    wet *= role_scale * {1: 0.96, 3: 1.10, 5: 1.24}[dream_level] * style_granulation
    drive = snapshot["structured_granulation_drive"] * style_granulation
    base_grain_ms = {1: 88.0, 3: 64.0, 5: 48.0}[dream_level]
    if role == "resonance":
        base_grain_ms *= 1.45
    grain_ms = base_grain_ms / max(1.0, 1.0 + 0.42 * (drive - 1.0))
    grain_n = max(128, min(len(frag), int(grain_ms * TARGET_SR / 1000.0)))
    hop_scale = {1: 0.48, 3: 0.40, 5: 0.32}[dream_level]
    hop_scale /= max(1.0, 1.0 + 0.22 * (drive - 1.0))
    hop_n = max(64, int(grain_n * hop_scale))
    window = np.hanning(grain_n).astype(np.float32)
    max_source = max(0, len(frag) - grain_n)
    drift = max(1, int(grain_n * (0.22 + 0.12 * max(0.0, drive - 1.0))))
    voice_count = (
        1
        + int(wet > 0.08)
        + int(dream_level >= 3 and wet > 0.16)
        + int(dream_level == 5 and wet > 0.38)
    )
    scan_rates = (1.0, 0.82, 1.17, 0.67)
    granular = np.zeros(len(frag), dtype=np.float32)
    for voice in range(voice_count):
        voice_audio = np.zeros(len(frag), dtype=np.float32)
        weights = np.zeros(len(frag), dtype=np.float32)
        phase = dream_level + voice * 2.17
        output_offset = int(voice * hop_n / max(1, voice_count))
        for grain_index, out_start in enumerate(
            range(output_offset, max(output_offset + 1, len(frag) - grain_n + 1), hop_n)
        ):
            phrase_offset = int(np.sin(grain_index * 1.618 + phase) * drift)
            if dream_level == 5 and voice > 0 and max_source > 0:
                source_start = int(out_start * scan_rates[voice % len(scan_rates)])
                source_start = (source_start + phrase_offset) % (max_source + 1)
            else:
                source_start = max(0, min(max_source, out_start + phrase_offset))
            out_end = min(len(frag), out_start + grain_n)
            size = out_end - out_start
            voice_audio[out_start:out_end] += frag[source_start:source_start + size] * window[:size]
            weights[out_start:out_end] += window[:size]
        active = weights > 1e-6
        voice_audio[active] /= weights[active]
        voice_audio[~active] = frag[~active]
        granular += voice_audio / voice_count
    # The granular identity flowers through the phrase instead of remaining a
    # static effect. This strengthens development without adding or removing
    # composition events, so density and palette breadth stay unchanged.
    progress = np.linspace(0.0, 1.0, len(frag), dtype=np.float32)
    learned_density_curve = ARTIST_STYLE.trajectory_curve(
        "density",
        progress,
        dream_level,
        int(RENDER_SEED) + sum(ord(char) for char in role),
    ).astype(np.float32)
    evolution = 0.48 + 0.34 * np.sin(np.pi * progress / 2.0) ** 1.35
    evolution += 0.36 * np.clip(learned_density_curve, 0.0, 1.0)
    local_wet = np.clip(wet * evolution, 0.0, 0.76)
    dry = np.asarray(frag, dtype=np.float32)
    mixed = dry * (1.0 - local_wet) + granular * local_wet
    dry_rms = float(np.sqrt(np.mean(dry * dry))) + 1e-9
    mixed_rms = float(np.sqrt(np.mean(mixed * mixed))) + 1e-9
    mixed *= min(1.14, dry_rms / mixed_rms)
    source_peak = float(np.max(np.abs(frag))) + 1e-9
    mixed_peak = float(np.max(np.abs(mixed))) + 1e-9
    if mixed_peak > source_peak:
        mixed *= source_peak / mixed_peak
    return mixed.astype(np.float32)


def crossfaded_circular_shift(source, shift, crossfade_sec=0.025):
    """Rotate a phrase without the hard wrap seam produced by ``np.roll``."""
    audio = np.asarray(source, dtype=np.float32)
    if len(audio) < 8:
        return audio.copy()
    shift = int(shift) % len(audio)
    if shift == 0:
        return audio.copy()
    left = audio[-shift:]
    right = audio[:-shift]
    overlap = min(
        max(8, int(float(crossfade_sec) * TARGET_SR)),
        len(left) // 2,
        len(right) // 2,
    )
    if overlap < 8:
        return audio.copy()
    phase = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    joined = left[-overlap:] * np.cos(phase) ** 2 + right[:overlap] * np.sin(phase) ** 2
    crossfaded = np.concatenate((left[:-overlap], joined, right[overlap:])).astype(np.float32)
    # Restore the exact original duration; the removed overlap is only a few
    # milliseconds and this avoids changing the learned phrase-length target.
    old = np.arange(len(crossfaded), dtype=np.float64)
    new = np.linspace(0.0, len(crossfaded) - 1.0, len(audio), dtype=np.float64)
    return np.interp(new, old, crossfaded).astype(np.float32)


def learned_material_evolution(frag, role, dream_level):
    """Apply bounded development inside an already selected formal moment."""
    source = np.asarray(frag, dtype=np.float32)
    if len(source) < 512:
        return source
    drive = composition_feedback_audio_snapshot()["development_drive"]
    request = max(0.0, drive - 1.0)
    if request <= 1e-6:
        return source

    progress = np.linspace(0.0, 1.0, len(source), dtype=np.float32)
    cycles = {1: 2.10, 3: 2.90, 5: 3.80}[dream_level]
    phase = {"resonance": 0.15, "texture": 0.55, "gesture": 1.10,
             "noise": 1.65, "impact": 2.10}.get(role, 0.0)
    motion = 0.5 + 0.5 * np.sin(2.0 * np.pi * cycles * progress + phase)
    depth = min(0.22, 0.055 + request * 0.34)
    trajectory_seed = int(RENDER_SEED) + sum((index + 1) * ord(char) for index, char in enumerate(role))
    learned_energy = ARTIST_STYLE.trajectory_curve(
        "energy", progress, dream_level, trajectory_seed
    ).astype(np.float32)
    learned_density = ARTIST_STYLE.trajectory_curve(
        "density", progress, dream_level, trajectory_seed
    ).astype(np.float32)
    learned_brightness = ARTIST_STYLE.trajectory_curve(
        "brightness", progress, dream_level, trajectory_seed
    ).astype(np.float32)
    learned_shape = 0.54 + 0.56 * (
        0.58 * np.clip(learned_energy, 0.0, 1.0)
        + 0.42 * np.clip(learned_density, 0.0, 1.0)
    )
    trajectory_blend = {1: 0.48, 3: 0.70, 5: 0.90}[dream_level]
    trajectory_blend *= min(1.0, ARTIST_STYLE.control("expressive_drive", dream_level) / 1.18)
    generic_curve = (1.0 - depth) + depth * motion
    amplitude_curve = generic_curve * (1.0 - trajectory_blend) + learned_shape * trajectory_blend

    cutoff = 1900 if role in {"texture", "resonance"} else 2700
    low = butter_filter(source, "lowpass", cutoff)
    detail = source - low
    opening_shape = np.clip(np.sin(np.pi * progress), 0.0, 1.0)
    generic_opening = 0.62 + 0.58 * opening_shape ** 1.45
    learned_opening = 0.48 + 0.96 * np.clip(learned_brightness, 0.0, 1.0)
    opening = generic_opening * (1.0 - trajectory_blend) + learned_opening * trajectory_blend
    spectral_mix = min(0.44, 0.12 + request * 0.72)
    evolved = source * (1.0 - spectral_mix) + (low + detail * opening) * spectral_mix
    evolved *= amplitude_curve
    source_rms = float(np.sqrt(np.mean(source * source))) + 1e-9
    evolved_rms = float(np.sqrt(np.mean(evolved * evolved))) + 1e-9
    evolved *= min(1.12, source_rms / evolved_rms)
    source_peak = float(np.max(np.abs(source))) + 1e-9
    evolved_peak = float(np.max(np.abs(evolved))) + 1e-9
    if evolved_peak > source_peak * 1.08:
        evolved *= (source_peak * 1.08) / evolved_peak
    return evolved.astype(np.float32)


def composition_sustain_bloom(frag, role, dream_level):
    """Carry sustained material toward the accepted whole-render duration.

    The composition preference already learns a different mean phrase duration
    for D1, D3 and D5.  This turns that learned value into a bounded overlap-add
    continuation for texture/resonance layers.  Short foreground gestures keep
    their identity and are never stretched into drones.
    """
    source = np.asarray(frag, dtype=np.float32)
    if role not in {"texture", "resonance"} or len(source) < 512:
        return source
    target_method = getattr(
        COMPOSITION_PREFERENCE, "target_average_event_duration", None
    )
    target = target_method(dream_level) if callable(target_method) else None
    if target is None or target <= 0.0:
        return source

    # D3/D5 add musical delay and every level adds learned release tails after
    # this stage. Compensate for that post-processing so the final, reported
    # duration approaches the learned target instead of overshooting it.
    post_processing_compensation = {1: 0.90, 3: 0.80, 5: 0.86}[dream_level]
    role_scale = (1.18 if role == "resonance" else 0.90) * post_processing_compensation
    target_samples = int(float(target) * role_scale * TARGET_SR)
    if len(source) >= int(target_samples * 0.82):
        return source

    # Avoid an abrupt pasted loop: neighbouring copies overlap under a
    # sine-shaped window and scan a slightly shifted part of the developed
    # phrase.  The upper bound prevents one tiny object from becoming the
    # entire composition by itself.
    target_samples = min(target_samples, max(len(source) * 4, len(source) + TARGET_SR))
    overlap = min(len(source) // 3, max(64, int(0.85 * TARGET_SR)))
    hop = max(64, len(source) - overlap)
    output = np.zeros(target_samples, dtype=np.float32)
    weights = np.zeros(target_samples, dtype=np.float32)
    window = np.ones(len(source), dtype=np.float32)
    if overlap > 1:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)) ** 2
        window[:overlap] = ramp
        window[-overlap:] = ramp[::-1]

    copy_index = 0
    for start in range(0, target_samples, hop):
        phrase = source
        if copy_index:
            shift = int(len(source) * 0.037 * copy_index) % len(source)
            phrase = crossfaded_circular_shift(source, shift)
            if copy_index % 3 == 2:
                phrase = reverse_blend(phrase, amount=0.08)
        end = min(target_samples, start + len(source))
        size = end - start
        output[start:end] += phrase[:size] * window[:size]
        weights[start:end] += window[:size]
        copy_index += 1
        if end >= target_samples:
            break
    active = weights > 1e-6
    output[active] /= weights[active]
    output[~active] = 0.0
    source_peak = float(np.max(np.abs(source))) + 1e-9
    output_peak = float(np.max(np.abs(output))) + 1e-9
    if output_peak > source_peak:
        output *= source_peak / output_peak
    return output.astype(np.float32)


def transform_fragment_for_role(
    frag,
    role,
    dream_level,
    *,
    formal_position=None,
    event_index=0,
    section_name=None,
):
    """
    Different musical functions receive different treatments.
    This prevents all layers from dissolving into the same texture.
    """
    temporal = d5_temporal_profile(dream_level)
    stretch_scale = temporal["stretch_scale"]
    envelope_scale = temporal["envelope_scale"]
    d5_phrase_scale = (
        1.0 / (d5_energy_drive(5) ** 0.58) if dream_level == 5 else 1.0
    )

    if role == "gesture":
        # D5 gestures should be clear phrases, not tiny decorative sparks.
        if dream_level == 5:
            stretch = random.uniform(0.75, 1.35) * (d5_phrase_scale ** 0.35)
            stretch *= stretch_scale
            fade_time = random.uniform(0.20, 0.75) * envelope_scale
            amp = random.uniform(0.135, 0.220)
        else:
            stretch = random.uniform(0.65, 1.75)
            stretch *= stretch_scale
            fade_time = random.uniform(0.15, 0.9)
            fade_time *= envelope_scale
            amp = random.uniform(0.125, 0.215)

        frag = stretch_audio(frag, stretch)
        if dream_level >= 3 and random.random() < 0.28:
            frag = reverse_blend(frag, amount=random.uniform(0.10, 0.25))
        frag = clean_band(frag, low=60, high=12500)
        frag = fade(frag, sec=fade_time)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.18, 0.42), decay=random.uniform(0.18, 0.35))

    elif role == "impact":
        stretch = random.uniform(0.65, 1.20) * stretch_scale * d5_phrase_scale if dream_level == 5 else random.uniform(0.55, 1.25) * stretch_scale
        frag = stretch_audio(frag, stretch)
        if random.random() < 0.22:
            frag = reverse_blend(frag, amount=random.uniform(0.10, 0.22))
        frag = clean_band(frag, low=45, high=13000)
        frag = fade(frag, sec=random.uniform(0.08, 0.55) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.10, 0.30), decay=random.uniform(0.12, 0.25))
        amp = random.uniform(0.120, 0.205) if dream_level == 5 else random.uniform(0.13, 0.24)

    elif role == "noise":
        stretch = random.uniform(1.4, 3.2) * stretch_scale * d5_phrase_scale if dream_level == 5 else random.uniform(1.2, 3.8 + dream_level * 0.25) * stretch_scale
        frag = stretch_audio(frag, stretch)
        if dream_level >= 3:
            frag = reverse_blend(frag, amount=random.uniform(0.20, 0.45))
        frag = clean_band(frag, low=80, high=14000)
        frag = fade(frag, sec=random.uniform(0.9, 2.8) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.25, 0.75), decay=random.uniform(0.18, 0.36))
        amp = random.uniform(0.065, 0.120) if dream_level == 5 else random.uniform(0.060, 0.125)

    elif role == "resonance":
        # Long layer, but quieter and more distinct in D5.
        stretch = random.uniform(3.4, 7.6) * stretch_scale * d5_phrase_scale if dream_level == 5 else random.uniform(3.0, 7.5 + dream_level * 0.45) * stretch_scale
        frag = stretch_audio(frag, stretch)
        frag = bound_sustained_fragment(frag, role, dream_level)
        if dream_level >= 3 and random.random() < (0.55 if dream_level == 5 else 1.0):
            frag = harmonic_bloom(frag, dream_level)
        frag = clean_band(frag, low=28, high=11200)
        frag = fade(frag, sec=random.uniform(4.5, 9.5) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.8, 1.8), decay=random.uniform(0.22, 0.45))
        amp = random.uniform(0.060, 0.125) if dream_level == 5 else random.uniform(0.055, 0.12)

    else:  # texture
        stretch = random.uniform(2.4, 5.8) * stretch_scale * d5_phrase_scale if dream_level == 5 else random.uniform(2.0, 5.5 + dream_level * 0.35) * stretch_scale
        frag = stretch_audio(frag, stretch)
        frag = bound_sustained_fragment(frag, role, dream_level)
        if dream_level >= 3 and random.random() < (0.45 if dream_level == 5 else 0.65):
            frag = harmonic_bloom(frag, dream_level)
        frag = clean_band(frag, low=34, high=12000)
        frag = fade(frag, sec=random.uniform(2.8, 6.8) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.45, 1.2), decay=random.uniform(0.18, 0.38))
        amp = random.uniform(0.075, 0.150) if dream_level == 5 else random.uniform(0.070, 0.145)

    frag = structured_granulation(frag, role, dream_level)
    frag = composition_sustain_bloom(frag, role, dream_level)
    if formal_position is not None:
        should_evolve, _context = musical_evolution_moment(
            formal_position,
            dream_level,
            event_index=event_index,
            section_name=section_name,
        )
        if should_evolve:
            frag = learned_material_evolution(frag, role, dream_level)
    amp *= (1 + dream_level * 0.055)
    if role == "gesture":
        amp *= learned_factor("gesture_weight", 4.0)
        amp *= d5_energy_drive(dream_level)
    elif role == "noise":
        amp /= learned_factor("noise_penalty", 4.0)
    elif role == "impact":
        amp /= learned_factor("impact_penalty", 4.0)
    elif role in ["texture", "resonance"]:
        amp *= learned_factor("ambient_weight", 2.5)
    amp *= composition_feedback_audio_snapshot()["event_activity_gain"]
    return frag, amp



def musical_delay_tail(x, role, dream_level):
    """Musical, per-phrase delay. It creates evolution and continuity, not a global wash."""
    if dream_level < 3:
        return x

    # Delay profiles by musical function.
    if role == "gesture":
        taps = [(0.19, 0.34), (0.37, 0.18)]
    elif role == "texture":
        taps = [(0.31, 0.24), (0.62, 0.13)]
    elif role == "resonance":
        taps = [(0.43, 0.22), (0.86, 0.12)]
    elif role == "noise":
        taps = [(0.13, 0.16), (0.29, 0.10)]
    else:
        taps = [(0.16, 0.18)]

    if d5_temporal_profile(dream_level)["delay_scale"] < 1.0:
        delay_scale = d5_temporal_profile(dream_level)["delay_scale"]
        taps = [(delay * delay_scale, gain) for delay, gain in taps]

    # D5 gets a little more phrase continuation, with quicker tap spacing.
    gain_scale = 1.15 if dream_level == 5 else 0.85
    extra = int(max(t[0] for t in taps) * TARGET_SR) + int(0.8 * TARGET_SR)
    out = np.zeros(len(x) + extra, dtype=np.float32)
    out[:len(x)] += x

    for delay_sec, gain in taps:
        n = int(delay_sec * TARGET_SR)
        copy = x * gain * gain_scale
        # Slightly soften delay copies so they support the phrase rather than clutter it.
        try:
            copy = butter_filter(copy, "lowpass", 9000 if role in ["gesture", "noise"] else 7500)
        except Exception:
            pass
        out[n:n + len(copy)] += copy

    return out / (np.max(np.abs(out)) + 1e-9)



def emergence_envelope_scales(features=None):
    """Return subtle, explainable envelope scaling for bright salient events.

    Bright foreground/transient material receives a little more time to enter
    and leave.  The learned transition control now affects the actual envelope,
    as well as continuity-aware object selection.
    """
    features = features or {}
    brightness = float(features.get("brightness", 0.0) or 0.0)
    brightness_amount = max(0.0, min(1.0, (brightness - 3200.0) / 6200.0))
    salience = max(
        float(features.get("foreground_probability", 0.0) or 0.0),
        float(features.get("transient_score", 0.0) or 0.0),
        float(features.get("contrast_score", 0.0) or 0.0),
    )
    salience = max(0.0, min(1.0, salience))
    bright_salience = brightness_amount * (0.45 + 0.55 * salience)
    learned_smoothness = learned_factor("transition_smoothness_weight", 3.0, 0.88, 1.35)
    return {
        "pre": learned_smoothness * (1.0 + 0.22 * bright_salience),
        "fade_in": learned_smoothness * (1.0 + 0.52 * bright_salience),
        "fade_out": learned_smoothness * (1.0 + 0.42 * bright_salience),
        "bright_salience": bright_salience,
    }


def organic_emergence(x, role, dream_level, features=None):
    """
    Makes phrase entries feel as if they grow out of the existing texture.
    It adds a soft pre-emergence shadow and longer musical envelopes,
    without blurring foreground gestures completely.
    """
    if len(x) < 32:
        return x.astype(np.float32)

    # Role-dependent emergence: textures/resonances grow slowly; gestures remain audible
    # but no longer appear as hard cuts.
    if role == "resonance":
        pre_sec = random.uniform(0.75, 1.80) if dream_level >= 3 else random.uniform(0.35, 0.90)
        in_sec = random.uniform(1.80, 4.20) if dream_level == 5 else random.uniform(1.20, 3.00)
        out_sec = random.uniform(3.00, 7.50)
        shadow_gain = 0.30
    elif role == "texture":
        pre_sec = random.uniform(0.55, 1.35) if dream_level >= 3 else random.uniform(0.25, 0.75)
        in_sec = random.uniform(1.20, 3.20) if dream_level == 5 else random.uniform(0.90, 2.40)
        out_sec = random.uniform(2.20, 5.80)
        shadow_gain = 0.24
    elif role == "gesture":
        pre_sec = random.uniform(0.12, 0.45) if dream_level >= 3 else random.uniform(0.05, 0.20)
        in_sec = random.uniform(0.22, 0.85) if dream_level == 5 else random.uniform(0.12, 0.55)
        out_sec = random.uniform(0.80, 2.20)
        shadow_gain = 0.16
    elif role == "noise":
        pre_sec = random.uniform(0.22, 0.70)
        in_sec = random.uniform(0.45, 1.40)
        out_sec = random.uniform(1.20, 3.00)
        shadow_gain = 0.13
    else:  # impact
        pre_sec = random.uniform(0.04, 0.18)
        in_sec = random.uniform(0.08, 0.32)
        out_sec = random.uniform(0.35, 1.20)
        shadow_gain = 0.10

    scales = emergence_envelope_scales(features)
    pre_sec *= scales["pre"]
    in_sec *= scales["fade_in"]
    out_sec *= scales["fade_out"]
    if dream_level == 5:
        temporal_scale = d5_temporal_profile(dream_level)["envelope_scale"]
        pre_sec *= temporal_scale
        in_sec *= temporal_scale
        out_sec *= temporal_scale

    # Pre-emergence shadow: a filtered reversed beginning that foreshadows the phrase.
    pre_n = min(int(pre_sec * TARGET_SR), max(0, len(x) // 3))
    if pre_n > 64:
        shadow = x[:pre_n][::-1].copy() * shadow_gain
        try:
            base_cutoff = 6200 if role in ["gesture", "impact"] else 4800
            shadow_cutoff = max(3800, base_cutoff - 1400 * scales["bright_salience"])
            shadow = butter_filter(shadow, "lowpass", shadow_cutoff)
        except Exception:
            pass
        shadow *= np.linspace(0.0, 1.0, len(shadow)).astype(np.float32)
        x = np.concatenate([shadow.astype(np.float32), x.astype(np.float32)])

    # Musical fade curves: cosine/sine-shaped, less mechanical than linear fades.
    n_in = min(int(in_sec * TARGET_SR), len(x) // 2)
    n_out = min(int(out_sec * TARGET_SR), len(x) // 2)

    if n_in > 8:
        curve = np.sin(np.linspace(0, np.pi / 2, n_in)) ** 1.35
        x[:n_in] *= curve.astype(np.float32)
    if n_out > 8:
        curve = np.cos(np.linspace(0, np.pi / 2, n_out)) ** 1.55
        x[-n_out:] *= curve.astype(np.float32)

    return x.astype(np.float32)


def spectral_bloom_timing(duration, dream_level, seed=None):
    """Return a render-specific bloom window instead of one fixed 75% gesture."""
    duration = max(0.0, float(duration))
    if duration <= 0.0:
        return 0.0, 0.0, 0.0
    rng = random.Random(
        int(RENDER_SEED if seed is None else seed) + int(dream_level) * 982451653
    )
    if dream_level == 1:
        peak_ratio = rng.uniform(0.42, 0.60)
    elif dream_level == 3:
        peak_ratio = rng.uniform(0.38, 0.62)
    else:
        peak_ratio = rng.uniform(0.34, 0.68)
    rise_ratio = rng.uniform(0.14, 0.25)
    fall_ratio = rng.uniform(0.13, 0.27)
    start_ratio = max(0.08, peak_ratio - rise_ratio)
    end_ratio = min(0.94, peak_ratio + fall_ratio)
    return duration * start_ratio, duration * peak_ratio, duration * end_ratio


def _delayed_signal(source, samples):
    delayed = np.zeros_like(source)
    samples = max(0, min(int(samples), max(0, len(source) - 1)))
    if samples == 0:
        delayed[:] = source
    else:
        delayed[samples:] = source[:-samples]
    return delayed


def balance_development_layer_low_end(layer, dream_level):
    """Keep source-derived beds/pads/drone blooms from stacking in the bass."""
    work = np.asarray(layer, dtype=np.float32).copy()
    if work.ndim != 2 or work.shape[1] != 2 or len(work) < 64:
        return work
    low_scale = {1: 0.34, 3: 0.44, 5: 0.54}[int(dream_level)]
    for channel in range(2):
        low = butter_filter(work[:, channel], "lowpass", 170)
        work[:, channel] += low * (low_scale - 1.0)
    return work.astype(np.float32)


def source_derived_bed(output, dream_level, seed=None):
    """Build a quiet granular bed from the render's selected source events.

    The bed re-samples overlapping windows from the actual palette mix. It adds
    no oscillator and does not alter or filter the dry composition. Long Hann
    windows and overlap-add create sustained continuity without hard loop seams.
    """
    source = np.asarray(output, dtype=np.float32)
    if source.ndim != 2 or source.shape[0] < 64 or source.shape[1] != 2:
        return source.copy()
    source_rms = float(np.sqrt(np.mean(source * source)))
    if source_rms < 1e-7:
        return source.copy()

    length = len(source)
    rng = random.Random(
        int(RENDER_SEED if seed is None else seed) + int(dream_level) * 86028121
    )
    analysis_hop = max(64, int(0.50 * TARGET_SR))
    analysis_size = max(128, int(1.25 * TARGET_SR))
    mono = np.mean(source, axis=1)
    candidates = []
    energies = []
    for start in range(0, max(1, length - analysis_size), analysis_hop):
        energy = float(np.sqrt(np.mean(mono[start:start + analysis_size] ** 2)))
        candidates.append(start)
        energies.append(energy)
    if not candidates or max(energies, default=0.0) < 1e-7:
        return source.copy()
    positive = [energy for energy in energies if energy > 1e-8]
    floor = float(np.percentile(positive, 38.0)) if positive else 0.0
    active_starts = [
        start for start, energy in zip(candidates, energies) if energy >= floor
    ] or candidates

    bed = np.zeros_like(source)
    weights = np.zeros(length, dtype=np.float32)
    destination = 0
    while destination < length:
        grain_sec = rng.uniform(
            2.2 if dream_level == 1 else 1.6,
            4.8 if dream_level == 1 else (4.0 if dream_level == 3 else 3.4),
        )
        grain_size = min(length, max(256, int(grain_sec * TARGET_SR)))
        possible = [start for start in active_starts if start + grain_size <= length]
        source_start = rng.choice(possible or [max(0, length - grain_size)])
        grain = source[source_start:source_start + grain_size].copy()
        if rng.random() < {1: 0.08, 3: 0.13, 5: 0.18}[dream_level]:
            grain = grain[::-1]
        window = np.hanning(grain_size).astype(np.float32)
        end = min(length, destination + grain_size)
        size = end - destination
        bed[destination:end] += grain[:size] * window[:size, None]
        weights[destination:end] += window[:size]
        hop_ratio = rng.uniform(0.30, 0.48)
        destination += max(64, int(grain_size * hop_ratio))

    active = weights > 1e-5
    bed[active] /= weights[active, None]
    bed[~active] = 0.0
    # Shape only the added under-layer toward the body; the original source
    # remains untouched and retains all of its high-frequency information.
    for channel in range(2):
        bed[:, channel] = butter_filter(bed[:, channel], "lowpass", 5200)
    bed = balance_development_layer_low_end(bed, dream_level)
    # The bed is audible only as a finite sequence of learned phrases. Its
    # granular source can no longer sit behind the whole work like a pasted loop.
    bed *= learned_bed_phrase_envelope(length, dream_level, seed=seed)[:, None]

    bed_rms = float(np.sqrt(np.mean(bed * bed))) + 1e-9
    target = source_rms * {1: 0.30, 3: 0.39, 5: 0.50}[dream_level]
    target *= learned_factor("ambient_weight", 3.0, 0.76, 1.28)
    bed *= min(1.0, target / bed_rms)
    result = source + bed
    source_peak = float(np.max(np.abs(source))) + 1e-9
    result_peak = float(np.max(np.abs(result))) + 1e-9
    if result_peak > source_peak * 1.12:
        bed *= (source_peak * 1.12) / result_peak
        result = source + bed
    return result.astype(np.float32)


def source_drone_bloom_layer(source, dream_level, seed=None):
    """Turn sustained source events into long, breathing drone blooms."""
    audio = np.asarray(source, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 512:
        return np.zeros_like(audio)
    source_rms = float(np.sqrt(np.mean(audio * audio)))
    if source_rms < 1e-7:
        return np.zeros_like(audio)

    local_seed = int(RENDER_SEED if seed is None else seed) + 15485863
    cloud = source_derived_bed(audio, dream_level, seed=local_seed) - audio
    stretch = {1: 1.35, 3: 1.68, 5: 2.05}[dream_level]
    slowed = np.stack(
        [stretch_audio(cloud[:, channel], stretch)[:len(audio)] for channel in range(2)],
        axis=1,
    )
    progress = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    # Multiple large blooms create arrival, expansion and release rather than
    # one permanent drone. D5 reaches a later, stronger second flowering.
    first = np.clip(np.sin(np.pi * progress), 0.0, 1.0) ** 1.25
    second = np.clip(
        np.sin(np.pi * np.clip((progress - 0.44) / 0.56, 0.0, 1.0)),
        0.0,
        1.0,
    ) ** 1.45
    envelope = np.clip(0.78 * first + {1: 0.24, 3: 0.48, 5: 0.72}[dream_level] * second, 0.0, 1.0)
    drone = (cloud * 0.50 + slowed * 0.50) * envelope[:, None]
    for channel in range(2):
        body = butter_filter(drone[:, channel], "lowpass", 3100)
        detail = drone[:, channel] - body
        opening = 0.42 + 0.78 * np.clip(first + 0.45 * second, 0.0, 1.0)
        drone[:, channel] = body + detail * opening
    drone = balance_development_layer_low_end(drone, dream_level)

    drone_rms = float(np.sqrt(np.mean(drone * drone))) + 1e-9
    target = source_rms * {1: 0.30, 3: 0.42, 5: 0.56}[dream_level]
    target *= learned_factor("material_development_weight", 2.6, 0.82, 1.36)
    drone *= min(1.35, target / drone_rms)
    return drone.astype(np.float32)


def source_pad_bloom_layer(source, dream_level, seed=None):
    """Create changing pads only from source events judged synth-like."""
    audio = np.asarray(source, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 512:
        return np.zeros_like(audio)
    source_rms = float(np.sqrt(np.mean(audio * audio)))
    if source_rms < 1e-7:
        return np.zeros_like(audio)

    local_seed = int(RENDER_SEED if seed is None else seed) + 32452843
    cloud = source_derived_bed(audio, dream_level, seed=local_seed) - audio
    rng = random.Random(local_seed + dream_level * 17)
    delay_left = int(rng.uniform(0.18, 0.42) * TARGET_SR)
    delay_right = int(rng.uniform(0.31, 0.67) * TARGET_SR)
    pad = np.empty_like(cloud)
    pad[:, 0] = cloud[:, 0] * 0.66 + _delayed_signal(cloud[:, 1], delay_left) * 0.34
    pad[:, 1] = cloud[:, 1] * 0.61 + _delayed_signal(cloud[:, 0], delay_right) * 0.39
    # Gentle source-dependent colour and evolving detail create a pad quality
    # without a separate oscillator or a repeated preset waveform.
    progress = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
    learned_energy = ARTIST_STYLE.trajectory_curve(
        "energy", progress, dream_level, local_seed
    ).astype(np.float32)
    learned_brightness = ARTIST_STYLE.trajectory_curve(
        "brightness", progress, dream_level, local_seed
    ).astype(np.float32)
    phrase_form = learned_bed_phrase_envelope(
        len(audio), dream_level, seed=local_seed + 97
    )
    evolution = 0.38 + 0.36 * learned_energy + 0.26 * learned_brightness
    evolution *= 0.16 + 0.84 * phrase_form
    for channel in range(2):
        body = butter_filter(pad[:, channel], "lowpass", 4200)
        detail = pad[:, channel] - body
        pad[:, channel] = body + detail * (0.38 + 0.82 * evolution)
    pad = 0.76 * pad + 0.24 * np.tanh(pad * 2.4) / 2.4
    pad *= evolution[:, None]
    pad = balance_development_layer_low_end(pad, dream_level)

    pad_rms = float(np.sqrt(np.mean(pad * pad))) + 1e-9
    target = source_rms * {1: 0.50, 3: 0.68, 5: 0.88}[dream_level]
    target *= learned_factor("synthetic_material_weight", 2.8, 0.80, 1.42)
    # Phrase shaping lowers the pad's whole-render RMS by design. Compensate
    # inside the finite phrases instead of filling their completed gaps with a
    # continuous layer; D5 can therefore remain the fullest level without
    # reverting to a permanent background loop.
    phrase_gain_cap = {1: 2.20, 3: 1.90, 5: 2.75}[dream_level]
    pad *= min(phrase_gain_cap, target / pad_rms)
    return pad.astype(np.float32)


def quantize_source_frequency(frequency):
    """Keep a detected source pitch, or gently place it in the chosen scale."""
    try:
        frequency = float(frequency)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(frequency) or frequency <= 0.0:
        return 0.0
    if HARMONY_STATE["scale"] == "free" or HARMONY_STATE["confidence"] < 0.55:
        return frequency

    midi = 69.0 + 12.0 * np.log2(frequency / 440.0)
    centre = int(round(midi))
    allowed = active_pitch_classes()
    candidates = [
        note for note in range(centre - 6, centre + 7) if note % 12 in allowed
    ]
    if not candidates:
        return frequency
    nearest = min(candidates, key=lambda note: abs(note - midi))
    return float(midi_to_hz(nearest))


def source_tonal_candidates(source, count=4):
    """Find several stable pitches in the current synth-like source material.

    There is deliberately no default oscillator pitch. If the selected source
    bus has no audible tonal evidence, the tonal bloom remains silent instead
    of stamping the same sine signature onto every composition.
    """
    audio = np.asarray(source, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1 or len(audio) < 512:
        return []
    source_rms = float(np.sqrt(np.mean(audio * audio)))
    if source_rms < 1e-7:
        return []

    frame_size = min(len(audio), 65_536)
    hop = max(256, frame_size // 2)
    starts = list(range(0, max(1, len(audio) - frame_size + 1), hop)) or [0]
    ranked = sorted(
        starts,
        key=lambda start: float(
            np.sqrt(np.mean(audio[start:start + frame_size] ** 2))
        ),
        reverse=True,
    )[:4]
    window = np.hanning(frame_size).astype(np.float32)
    spectra = []
    for start in ranked:
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        spectra.append(np.abs(np.fft.rfft(frame * window)))
    spectrum = np.mean(spectra, axis=0)
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / TARGET_SR)
    band = (frequencies >= 55.0) & (frequencies <= 1800.0)
    local_peaks = np.flatnonzero(
        band
        & (spectrum >= np.roll(spectrum, 1))
        & (spectrum > np.roll(spectrum, -1))
    )
    if not len(local_peaks):
        return []
    local_peaks = sorted(
        local_peaks, key=lambda index: float(spectrum[index]), reverse=True
    )
    floor = float(np.max(spectrum[local_peaks])) * 0.035
    candidates = []
    for index in local_peaks:
        if spectrum[index] < floor:
            break
        frequency = quantize_source_frequency(frequencies[index])
        if frequency <= 0.0:
            continue
        # Drones already carry the low register. Preserve the detected pitch
        # class but octave-lift tonal blooms into a clearer musical foreground.
        while frequency < 165.0:
            frequency *= 2.0
        # Closely spaced FFT bins describe one pitch, not extra musical voices.
        if all(max(frequency, other) / min(frequency, other) >= 1.055 for other in candidates):
            candidates.append(float(frequency))
        if len(candidates) >= max(1, int(count)):
            break
    return candidates


def learned_tonal_bloom_layer(source, dream_level, seed=None):
    """Add optional, source-anchored synth phrases at learned formal moments.

    Sine is allowed here as a synthesis material, but never as a compulsory
    whole-render layer. Its pitches are inferred from the selected recordings,
    its motion follows their amplitude, and every appearance has a finite
    arrival, body and release. This preserves synth/drone exploration without
    recreating the recurring tinnitus-like signature rejected in listening.
    """
    audio = np.asarray(source, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 512:
        return np.zeros_like(audio)
    source_rms = float(np.sqrt(np.mean(audio * audio)))
    synth_request = composition_feedback_audio_snapshot()["synthetic_layer_gain"]
    if source_rms < 1e-7 or synth_request <= 1e-6:
        return np.zeros_like(audio)

    frequencies = source_tonal_candidates(
        audio, count={1: 3, 3: 5, 5: 7}[dream_level]
    )
    if not frequencies:
        return np.zeros_like(audio)

    local_seed = int(RENDER_SEED if seed is None else seed) + 49979687
    rng = random.Random(local_seed + dream_level * 97)
    phrase_form = learned_bed_phrase_envelope(
        len(audio), dream_level, seed=local_seed
    )
    active = phrase_form > 0.045
    edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    regions = [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= 256]
    if not regions:
        return np.zeros_like(audio)

    desired = {1: 1, 3: 2, 5: 3}[dream_level]
    ranked_regions = []
    for start, end in regions:
        position = ((start + end) * 0.5) / max(1, len(audio) - 1)
        source_activity = float(np.sqrt(np.mean(audio[start:end] ** 2))) / (source_rms + 1e-9)
        form_score = learned_form_context(position, dream_level, seed=local_seed)
        ranked_regions.append((0.72 * form_score + 0.28 * min(1.4, source_activity), start, end))
    selected = sorted(sorted(ranked_regions, reverse=True)[:desired], key=lambda item: item[1])

    layer = np.zeros_like(audio)
    mono_source = np.mean(audio, axis=1)
    for phrase_index, (_, start, end) in enumerate(selected):
        size = end - start
        base_frequency = frequencies[(phrase_index * 2 + dream_level) % len(frequencies)]
        partner_frequency = frequencies[(phrase_index * 2 + dream_level + 1) % len(frequencies)]
        source_motion = np.abs(mono_source[start:end]).astype(np.float32)
        source_motion = butter_filter(source_motion, "lowpass", 7.0)
        source_motion -= float(np.min(source_motion))
        source_motion /= float(np.max(source_motion)) + 1e-9
        deviation_depth = {1: 0.0025, 3: 0.0045, 5: 0.0065}[dream_level]
        instantaneous = base_frequency * (
            1.0 + (source_motion - 0.5) * deviation_depth
        )
        phase = rng.uniform(0.0, 2.0 * np.pi) + np.cumsum(
            2.0 * np.pi * instantaneous / TARGET_SR
        )
        partner_phase = rng.uniform(0.0, 2.0 * np.pi) + np.cumsum(
            2.0 * np.pi * partner_frequency * (1.0 - 0.35 * (source_motion - 0.5) * deviation_depth)
            / TARGET_SR
        )
        partial_gain = {1: 0.10, 3: 0.17, 5: 0.24}[dream_level]
        tone = np.sin(phase) + partial_gain * np.sin(partner_phase)
        envelope = np.power(np.clip(phrase_form[start:end], 0.0, 1.0), 1.45)
        envelope *= 0.58 + 0.42 * source_motion
        edge_size = min(size // 2, max(16, int(0.055 * TARGET_SR)))
        edge = np.sin(
            np.linspace(0.0, np.pi / 2.0, edge_size, dtype=np.float32)
        ) ** 1.35
        envelope[:edge_size] *= edge
        envelope[-edge_size:] *= edge[::-1]
        mono = (tone * envelope).astype(np.float32)
        pan_extent = {1: 0.48, 3: 0.70, 5: 0.90}[dream_level]
        direction = -1.0 if (phrase_index + dream_level) % 2 else 1.0
        stereo = moving_pan_stereo(
            mono,
            -direction * pan_extent,
            direction * pan_extent,
            motion_cycles={1: 0.20, 3: 0.45, 5: 0.75}[dream_level],
        )
        layer[start:end] += stereo

    layer_rms = float(np.sqrt(np.mean(layer * layer))) + 1e-9
    target = source_rms * {1: 0.075, 3: 0.115, 5: 0.160}[dream_level]
    target *= 0.70 + 0.55 * synth_request
    layer *= min(1.0, target / layer_rms)
    peak = float(np.max(np.abs(layer))) + 1e-9
    peak_limit = {1: 0.075, 3: 0.105, 5: 0.140}[dream_level]
    if peak > peak_limit:
        layer *= peak_limit / peak
    return layer.astype(np.float32)


def central_spectral_bloom(output, dream_level):
    """Create a source-derived spectral flowering with no added pure-tone signature."""
    length = len(output)
    if length < 32:
        return output
    t = np.arange(length, dtype=np.float32) / TARGET_SR
    duration = length / TARGET_SR
    start_t, peak_t, end_t = spectral_bloom_timing(duration, dream_level)

    env = np.zeros_like(t)
    rise = (t >= start_t) & (t < peak_t)
    fall = (t >= peak_t) & (t <= end_t)
    env[rise] = 0.5 - 0.5 * np.cos(np.pi * (t[rise] - start_t) / max(1e-6, peak_t - start_t))
    env[fall] = 0.5 + 0.5 * np.cos(np.pi * (t[fall] - peak_t) / max(1e-6, end_t - peak_t))
    env = np.power(env, 1.18 if dream_level == 1 else 1.35)
    if dream_level == 1:
        # D1 needs an audible return, not a single early swell followed by
        # disappearance. A later, quieter bloom reuses the same source colour.
        late_start = min(0.82 * duration, peak_t + 0.14 * duration)
        late_peak = min(0.88 * duration, late_start + 0.08 * duration)
        late_end = min(0.96 * duration, late_peak + 0.10 * duration)
        late = np.zeros_like(t)
        late_rise = (t >= late_start) & (t < late_peak)
        late_fall = (t >= late_peak) & (t <= late_end)
        late[late_rise] = 0.5 - 0.5 * np.cos(
            np.pi * (t[late_rise] - late_start) / max(1e-6, late_peak - late_start)
        )
        late[late_fall] = 0.5 + 0.5 * np.cos(
            np.pi * (t[late_fall] - late_peak) / max(1e-6, late_end - late_peak)
        )
        env = np.clip(env + 0.64 * late, 0.0, 1.0)

    # The halo is extracted from the selected palette already present in the
    # composition. No fixed oscillator frequencies, pitch sweep or added noise
    # are introduced, so each library and render retains its own spectral identity.
    mono = np.mean(np.asarray(output, dtype=np.float32), axis=1)
    lower = butter_filter(mono, "lowpass", 420)
    upper = butter_filter(mono, "lowpass", 3600)
    body = np.tanh((upper - lower) * 1.35).astype(np.float32)
    # Restore source-derived air and articulation without a global high shelf.
    # If the uploaded library has no high detail, this adds none; if it does,
    # the detail opens only during the learned bloom rather than becoming a
    # permanent whistle or tinnitus-like layer.
    detail_upper = butter_filter(mono, "lowpass", 9000)
    detail_lower = butter_filter(mono, "lowpass", 2400)
    detail = np.tanh((detail_upper - detail_lower) * 1.18).astype(np.float32)
    rng = random.Random(RENDER_SEED + dream_level * 32452843)
    delay_a = int(rng.uniform(0.070, 0.190) * TARGET_SR)
    delay_b = int(rng.uniform(0.160, 0.340) * TARGET_SR)
    detail_gain = {1: 0.36, 3: 0.46, 5: 0.58}[dream_level]
    left = body * 0.64 + _delayed_signal(body, delay_a) * 0.36
    right = body * 0.58 + _delayed_signal(body, delay_b) * 0.42
    left += detail_gain * _delayed_signal(detail, max(1, delay_b // 2))
    right += detail_gain * _delayed_signal(detail, max(1, delay_a // 2))
    wet = {1: 0.150, 3: 0.205, 5: 0.270}[dream_level]
    wet *= ARTIST_STYLE.control("expressive_drive", dream_level)
    wet *= learned_factor("bloom_weight", 5.0)
    if dream_level == 5:
        wet /= np.sqrt(d5_energy_drive(5))
    output[:, 0] += left * env * wet
    output[:, 1] += right * env * wet
    return output


def generate_soundscape(dream_level):
    global CURRENT_MATERIAL_PLAN, CURRENT_FORM_VARIANT, LEARNING_WEIGHTS, REPRESENTATION_ASSIST, COMPOSITION_PREFERENCE, ARTIST_STYLE
    global LEARNED_SYNTH_AFFINITY, ORIGIN_LEARNING_SNAPSHOT
    global CURRENT_LIBRARY_COVERAGE_SNAPSHOT
    CURRENT_MATERIAL_PLAN = None
    CURRENT_FORM_VARIANT = select_form_variant(dream_level)

    print("Generator revision:", GENERATOR_REVISION)
    print("Render seed:", RENDER_SEED)
    print("Form variant:", CURRENT_FORM_VARIANT)
    LEARNING_WEIGHTS = load_learning_weights(dream_level)
    representation_config = os.environ.get(
        "HYPNOIA_REPRESENTATION_CONFIG", REPRESENTATION_CONFIG_FILE
    )
    REPRESENTATION_ASSIST = RepresentationAssist.from_config(representation_config)
    print("Representation assist:", REPRESENTATION_ASSIST.snapshot())
    composition_preference_path = os.environ.get(
        "HYPNOIA_COMPOSITION_PREFERENCE", COMPOSITION_PREFERENCE_FILE
    )
    COMPOSITION_PREFERENCE = CompositionPreferenceAssist.from_file(
        composition_preference_path
    )
    print("Composition preference:", COMPOSITION_PREFERENCE.snapshot())
    artist_style_path = os.environ.get(
        "HYPNOIA_ARTIST_STYLE_PROFILE", ARTIST_STYLE_FILE or ""
    )
    ARTIST_STYLE = ArtistStyleAssist.from_file(artist_style_path)
    print("Artist style:", ARTIST_STYLE.snapshot())
    pulse_bpm = random.uniform(*D5_REFERENCE_TARGETS["pulse_bpm_range"]) if dream_level == 5 else 0.0
    if dream_level == 5:
        print("Reference pulse BPM:", round(pulse_bpm, 3))

    profile = load_profile()
    objects = load_memory_objects()
    LEARNED_SYNTH_AFFINITY, ORIGIN_LEARNING_SNAPSHOT = build_embedding_origin_model(
        objects, REPRESENTATION_ASSIST
    )
    print("Origin learning:", ORIGIN_LEARNING_SNAPSHOT)
    with open(MEMORY_FILE, "r", encoding="utf-8") as handle:
        memory = json.load(handle)
    sample_profile = load_sample_learning_profile(memory)
    sync_recording_coverage(
        sample_profile,
        (recording.get("recording", "") for recording in memory),
    )
    build_role_pools(objects)

    output = np.zeros((OUTPUT_DURATION * TARGET_SR, 2), dtype=np.float32)
    sustained_source_bus = np.zeros_like(output)
    synthetic_source_bus = np.zeros_like(output)

    # The learned event count and role distribution stay active, while timing
    # moves among compatible arcs so every render is not the same five-part copy.
    form = composed_form(dream_level)

    previous = None
    used = set()
    motif_bank = []
    usage_counts = {}
    usage_by_sample = {}
    usage_details = {}
    event_durations = []
    foreground_durations = []
    event_starts = []
    event_timeline = []
    movement_events = []
    total_added = 0
    role_counts = {"gesture": 0, "texture": 0, "resonance": 0, "noise": 0, "impact": 0}
    role_last_end = {role: None for role in role_counts}

    section_item_counts = planned_form_items(form, dream_level)
    remaining_role_counts = planned_role_targets(sum(section_item_counts), dream_level)

    for (section_name, section_start, section_end, _density), items in zip(
        form, section_item_counts
    ):
        section_length = section_end - section_start

        for i in range(items):
            role = role_sequence_for_section(
                section_name,
                dream_level,
                remaining_role_counts=remaining_role_counts,
            )

            # Keep one palette across the form. Sections differ through role,
            # transformation and density rather than unrelated new families.
            preferred_groups = None

            # Coherence: sometimes return to a previous object from the same role.
            # This creates motif-like recurrence instead of unrelated new material.
            same_role_motifs = [m for m in motif_bank if m.get("role") == role]
            repeat_chance = (
                {1: 0.34, 3: 0.33, 5: 0.30}[dream_level]
                * learned_factor("coherence_weight", 4.0, 0.70, 1.35)
                * learned_factor("material_development_weight", 1.8, 0.85, 1.30)
                / learned_factor("repetition_control", 3.0, 0.70, 1.45)
            )
            _, unique_object_limit = material_plan_limits(dream_level)
            palette_is_full = len(used) >= unique_object_limit
            palette_hard_max = max(
                unique_object_limit,
                int(round(unique_object_limit * 1.12)),
            )
            # Reaching the motif target used to force every later event back
            # into the small early motif bank. That made one recording occupy
            # half of D1/D5 even though the material plan contained many valid
            # sources. Keep recurrence audible, but continue discovering
            # related embedded objects throughout the complete form.
            motif_probability = repeat_chance
            if palette_is_full:
                motif_probability = max(
                    motif_probability,
                    {1: 0.62, 3: 0.52, 5: 0.42}[dream_level],
                )
            use_motif = bool(same_role_motifs) and random.random() < motif_probability
            if len(used) >= palette_hard_max and same_role_motifs:
                use_motif = True

            if use_motif:
                motif_candidates = diversify_overused_recordings(
                    same_role_motifs,
                    usage_counts,
                    dream_level,
                    len(CURRENT_MATERIAL_PLAN or ()),
                )
                obj = random.choice(motif_candidates)
                key = (obj["recording_id"], obj["object_id"])
            else:
                obj = choose_weighted(objects, profile, dream_level, previous, desired_role=role, preferred_groups=preferred_groups, usage_counts=usage_counts, sample_profile=sample_profile)
                key = (obj["recording_id"], obj["object_id"])
                tries = 0
                while key in used and tries < 18:
                    obj = choose_weighted(objects, profile, dream_level, previous, desired_role=role, preferred_groups=preferred_groups, usage_counts=usage_counts, sample_profile=sample_profile)
                    key = (obj["recording_id"], obj["object_id"])
                    tries += 1
                used.add(key)
                motif_limit = palette_hard_max
                if len(motif_bank) < motif_limit and role in ["gesture", "texture", "resonance"]:
                    motif_obj = dict(obj)
                    motif_obj["role"] = role
                    motif_bank.append(motif_obj)

            previous = obj

            frag = load_fragment(obj)
            if frag is None:
                continue

            formal_position = (
                section_start + (i / max(1, items)) * section_length
            ) / max(1.0, OUTPUT_DURATION)
            frag, amp = transform_fragment_for_role(
                frag,
                role,
                dream_level,
                formal_position=formal_position,
                event_index=i,
                section_name=section_name,
            )
            frag = bound_fragile_air_punctuation(frag, obj, dream_level)
            feedback_audio = composition_feedback_audio_snapshot()
            foreground_gain = feedback_audio["foreground_presence_gain"]
            if role in {"gesture", "impact"}:
                amp *= foreground_gain
            elif role in {"texture", "resonance"}:
                amp /= np.sqrt(foreground_gain)
            if source_origin(obj) == "instrument_hybrid":
                amp *= np.sqrt(feedback_audio["instrument_presence_gain"])
            frag = d5_internal_motion(frag, role, dream_level, section_name)
            if dream_level == 5:
                # Shape an audible formal energy arc without merely normalising
                # the entire render louder.
                section_gain = {
                    "opening": 0.90,
                    "activation": 1.08,
                    "complexity": 1.22,
                    "memory": 1.04,
                    "resolution": 0.88,
                }[section_name]
                amp *= 1.0 + (section_gain - 1.0) * d5_energy_drive(dream_level)
            # A recurring motif keeps its small articulatory variation so it
            # remains alive. This is distinct from the strong learned spectral/
            # tonal evolution above, which is restricted to formal moments.
            if use_motif:
                frag = maybe_variation_transform(frag, role, dream_level)
                amp *= random.uniform(0.82, 1.05)

            # Final smoothing pass: delays first, then an organic emergence envelope,
            # so phrases feel as if they grow from previous material rather than being inserted.
            frag = musical_delay_tail(frag, role, dream_level)
            frag = feedback_release_tail(frag, role)
            frag = organic_emergence(frag, role, dream_level, obj.get("features", {}))
            frag = feedback_release_guard(frag, role)
            frag = continuity_edge_guard(frag, role, dream_level)

            # Positioning: D5 uses phrase lanes so layers stay perceptually distinct.
            base_position = section_start + (i / max(1, items)) * section_length
            base_position = d5_soft_grid_start(
                base_position,
                section_start,
                role,
                dream_level,
                pulse_bpm,
            )

            if dream_level == 5:
                role_offsets = {
                    "resonance": -2.10,
                    "texture": -0.40,
                    "gesture": 0.70,
                    "noise": 1.10,
                    "impact": 0.25,
                }
                role_jitter = {
                    "resonance": 1.80,
                    "texture": 1.30,
                    "gesture": 0.55,
                    "noise": 0.85,
                    "impact": 0.35,
                }
                jitter = role_offsets.get(role, 0.0) + random.uniform(-role_jitter.get(role, 2.0), role_jitter.get(role, 2.0))
            else:
                if role in ["gesture", "impact"]:
                    jitter = random.uniform(-1.5, 1.5)
                elif role == "noise":
                    jitter = random.uniform(-3.0, 3.0)
                else:
                    jitter = random.uniform(-6.0, 6.0)

            start = max(0, base_position + jitter)
            start = d5_continuity_start(
                start,
                len(frag) / TARGET_SR,
                role,
                role_last_end[role],
                dream_level,
                pulse_bpm,
            )
            if role in ["texture", "resonance"]:
                sustained_ends = [
                    role_last_end[name]
                    for name in ("texture", "resonance")
                    if role_last_end[name] is not None
                ]
                continuity_reference = max(sustained_ends) if sustained_ends else None
            else:
                continuity_reference = role_last_end[role]
            start = feedback_continuity_start(
                start,
                len(frag) / TARGET_SR,
                role,
                continuity_reference,
            )

            # Spatial identity: D5 separates roles into recognisable regions/layers.
            if dream_level == 5:
                if role == "resonance":
                    pan = random.uniform(-0.35, 0.35)
                elif role == "texture":
                    pan = random.choice([random.uniform(-0.75, -0.25), random.uniform(0.25, 0.75)])
                elif role == "gesture":
                    pan = random.uniform(-0.95, 0.95)
                elif role == "noise":
                    pan = random.choice([random.uniform(-0.95, -0.55), random.uniform(0.55, 0.95)])
                else:
                    pan = random.uniform(-0.55, 0.55)
            else:
                if role in ["gesture", "impact", "noise"]:
                    pan = random.uniform(-0.95, 0.95)
                else:
                    pan = random.uniform(-0.60, 0.60)

            pan *= composition_feedback_audio_snapshot()["stereo_width"]
            pan = max(-0.98, min(0.98, pan))

            add_to_output(output, frag, start, amp, pan)
            if role in {"texture", "resonance"}:
                add_to_output(sustained_source_bus, frag, start, amp, pan)
            if effective_synthetic_score(obj) >= 0.56:
                add_to_output(synthetic_source_bus, frag, start, amp, pan)
            event_duration = len(frag) / TARGET_SR
            event_timeline.append({
                "recording": obj["recording"],
                "recording_id": obj["recording_id"],
                "object_id": obj["object_id"],
                "role": role,
                "section": section_name,
                "render_start_sec": round(float(start), 6),
                "render_end_sec": round(float(start + event_duration), 6),
                "source_start_sec": round(float(obj.get("start", 0.0)), 6),
                "source_end_sec": round(float(obj.get("end", 0.0)), 6),
            })
            movement_events.append({
                "object": obj,
                "role": role,
                "start": float(start),
                "duration": float(event_duration),
            })
            role_last_end[role] = max(
                role_last_end[role] or 0.0,
                start + event_duration,
            )
            event_durations.append(event_duration)
            event_starts.append(start)
            if role in ["gesture", "impact", "noise"]:
                foreground_durations.append(event_duration)
            usage_counts[obj["recording"]] = usage_counts.get(obj["recording"], 0) + 1
            key = sample_key(obj)
            usage_by_sample[key] = usage_by_sample.get(key, 0) + 1
            detail = usage_details.setdefault(key, {
                "recording": obj["recording"],
                "recording_id": obj["recording_id"],
                "object_id": obj["object_id"],
                "legacy_object_id": obj.get("legacy_id"),
                "selection_count": 0,
                "exposure_sec": 0.0,
                "gain_sum": 0.0,
                "role_counts": {},
                "section_counts": {},
                "first_start_sec": None,
                "last_start_sec": None,
            })
            detail["selection_count"] += 1
            detail["exposure_sec"] = round(detail["exposure_sec"] + len(frag) / TARGET_SR, 6)
            detail["gain_sum"] = round(detail["gain_sum"] + float(amp), 6)
            detail["role_counts"][role] = detail["role_counts"].get(role, 0) + 1
            detail["section_counts"][section_name] = detail["section_counts"].get(section_name, 0) + 1
            detail["first_start_sec"] = round(start, 6) if detail["first_start_sec"] is None else detail["first_start_sec"]
            detail["last_start_sec"] = round(start, 6)
            sample_entry = ensure_sample_entry(sample_profile, obj)
            sample_entry["times_selected"] = int(sample_entry.get("times_selected", 0)) + 1
            sample_entry["last_used"] = datetime.now().isoformat(timespec="seconds")
            sample_profile["total_render_selections"] = int(sample_profile.get("total_render_selections", 0)) + 1

            total_added += 1
            role_counts[role] += 1

    phrase_layer = add_source_musical_phrases(
        output,
        dream_level,
        pulse_bpm=pulse_bpm,
    )
    movement_layer = electroacoustic_movement_layer(
        output,
        movement_events,
        dream_level,
    )
    drone_layer = source_drone_bloom_layer(sustained_source_bus, dream_level)
    drone_layer *= ARTIST_STYLE.control("drone_presence", dream_level)
    pad_layer = source_pad_bloom_layer(synthetic_source_bus, dream_level)
    tonal_bloom_layer = learned_tonal_bloom_layer(synthetic_source_bus, dream_level)
    output += movement_layer + drone_layer + pad_layer
    output = source_derived_bed(output, dream_level)
    output = central_spectral_bloom(output, dream_level)
    # Tonal synthesis sits above the source-derived bed and spectral bloom so
    # it articulates the form without changing their material selection.
    output += tonal_bloom_layer
    output = final_mix(output, dream_level)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(
        OUTPUT_FOLDER,
        f"Hyponoia_v3_memory_bloom_smooth_D{dream_level}_{timestamp}.wav"
    )

    sf.write(outfile, output, TARGET_SR)

    frequency_stems = save_frequency_stems(output, outfile)
    source_development = {
        "design": "source-derived recurrent phrases, learned-preference electroacoustic movement, drones, synth-like pads and optional tonal blooms",
        "oscillator_layers": int(np.max(np.abs(tonal_bloom_layer)) > 1e-8),
        "oscillator_policy": "source-anchored, scale-aware, phrase-bounded and optional",
        "sustained_bus_rms": round(float(np.sqrt(np.mean(sustained_source_bus ** 2))), 8),
        "synthetic_bus_rms": round(float(np.sqrt(np.mean(synthetic_source_bus ** 2))), 8),
        "phrase_layer_rms": round(float(np.sqrt(np.mean(phrase_layer ** 2))), 8),
        "movement_layer_rms": round(float(np.sqrt(np.mean(movement_layer ** 2))), 8),
        "movement_candidate_count": int(len(movement_events)),
        "drone_layer_rms": round(float(np.sqrt(np.mean(drone_layer ** 2))), 8),
        "pad_layer_rms": round(float(np.sqrt(np.mean(pad_layer ** 2))), 8),
        "tonal_bloom_layer_rms": round(float(np.sqrt(np.mean(tonal_bloom_layer ** 2))), 8),
        "tonal_bloom_frequencies_hz": [
            round(value, 3)
            for value in source_tonal_candidates(
                synthetic_source_bus, count={1: 3, 3: 5, 5: 7}[dream_level]
            )
        ],
        "artist_style": ARTIST_STYLE.snapshot(),
    }

    print()
    print("Added layers:", total_added)
    print("Role counts:", role_counts)
    sorted_starts = sorted(event_starts)
    quick_successions = sum(
        1 for left, right in zip(sorted_starts, sorted_starts[1:])
        if right - left <= 0.75
    )
    temporal_metrics = {
        "reference_pulse_bpm": round(float(pulse_bpm), 6) if dream_level == 5 else None,
        "event_rate_per_minute": round(total_added / (OUTPUT_DURATION / 60.0), 6),
        "foreground_event_rate_per_minute": round(
            sum(role_counts[role] for role in ["gesture", "impact", "noise"])
            / (OUTPUT_DURATION / 60.0),
            6,
        ),
        "average_event_duration_sec": round(float(np.mean(event_durations)) if event_durations else 0.0, 6),
        "average_foreground_duration_sec": round(
            float(np.mean(foreground_durations)) if foreground_durations else 0.0,
            6,
        ),
        "quick_succession_count": int(quick_successions),
    }
    print("Temporal metrics:", temporal_metrics)
    CURRENT_LIBRARY_COVERAGE_SNAPSHOT = update_recording_coverage(
        sample_profile,
        (recording.get("recording", "") for recording in memory),
        usage_counts,
    )
    print("Library coverage:", CURRENT_LIBRARY_COVERAGE_SNAPSHOT)
    print("Saved:")
    print(outfile)
    render_report_path = save_render_report(
        outfile,
        dream_level,
        usage_by_sample,
        usage_details,
        usage_counts,
        role_counts,
        temporal_metrics,
        frequency_stems,
        event_timeline,
        source_development,
    )
    save_sample_learning_profile(sample_profile)

    print("Latest composition:")
    print(outfile)
    print("Frequency stems (LOW / MID / HIGH):")
    for path in frequency_stems["timestamped"].values():
        print(path)
    print("Render report:")
    print(render_report_path)
    print("Sample learning profile:")
    print(SAMPLE_LEARNING_FILE)

if __name__ == "__main__":
    import sys

    dream_level = 5
    root = 0
    scale = "free"
    confidence = 0.0

    if len(sys.argv) > 1:
        try:
            dream_level = int(sys.argv[1])
        except ValueError:
            print("Invalid dream level; using D5.")
            dream_level = 5

    if len(sys.argv) > 2:
        root = sys.argv[2]
    if len(sys.argv) > 3:
        scale = sys.argv[3]
    if len(sys.argv) > 4:
        confidence = sys.argv[4]

    if dream_level <= 1:
        dream_level = 1
    elif dream_level <= 3:
        dream_level = 3
    else:
        dream_level = 5

    print("Dream level:", dream_level)
    configure_harmony(root, scale, confidence)
    generate_soundscape(dream_level)

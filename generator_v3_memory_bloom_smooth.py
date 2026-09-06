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
from library_coverage_v1 import (
    recording_coverage_factor,
    sync_recording_coverage,
    update_recording_coverage,
)
from representation_assist_v1 import RepresentationAssist

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
GENERATOR_REVISION = "2026-09-06-library-coverage-memory-10"
REPRESENTATION_CONFIG_FILE = os.environ.get(
    "HYPNOIA_REPRESENTATION_CONFIG",
    USER_PATHS.get("representation_config", "representation_config.json"),
)
REPRESENTATION_ASSIST = RepresentationAssist.disabled()
REPRESENTATION_ASSIST_ROLES = frozenset({"gesture", "texture", "impact", "noise"})
COMPOSITION_PREFERENCE_FILE = os.environ.get(
    "HYPNOIA_COMPOSITION_PREFERENCE",
    USER_PATHS.get("composition_preference", "composition_preference_v1.json"),
)
COMPOSITION_PREFERENCE = CompositionPreferenceAssist.disabled()
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
        with open(LEARNING_FILE, "r") as f:
            data = json.load(f)
        incoming = data.get("weights", {})
        level_name = f"D{int(dream_level)}" if dream_level in (1, 3, 5) else None
        level_incoming = data.get("level_weights", {}).get(level_name, {}) if level_name else {}
        for key in weights:
            try:
                value = float(incoming.get(key, 1.0)) + float(level_incoming.get(key, 1.0)) - 1.0
                weights[key] = float(max(0.5, min(1.8, float(value))))
            except (TypeError, ValueError):
                pass
    except FileNotFoundError:
        print("Learning profile not found; using neutral weights.")
    except (OSError, json.JSONDecodeError) as exc:
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
    granulation = learned_factor("structured_granulation_weight", 2.2, 0.82, 2.00)
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
        "structured_granulation_wet": float(max(0.0, min(0.62, (granulation - 1.0) * 0.68))),
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
    """Create one of three audibly different D5 energy arcs per render."""
    if dream_level != 5:
        return 1.0
    variant = D5_FORM_VARIANTS.get(CURRENT_FORM_VARIANT, {})
    base = float(variant.get(section_name, 1.0))
    drive = d5_energy_drive(dream_level)
    if base >= 1.0:
        return 1.0 + (base - 1.0) * drive
    return base


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
    """Balanced per-render palette: focused, but never reduced to a tiny loop."""
    recording_base = {1: 6, 3: 10, 5: 16}[dream_level]
    object_base = {1: 12, 3: 20, 5: 30}[dream_level]
    focus = learned_factor("material_development_weight", 1.2, 0.88, 1.18)
    breadth = learned_factor("exploration_weight", 1.6, 0.90, 1.30)
    recordings = max(3, int(round(recording_base * breadth / focus)))
    objects = max(recordings, int(round(object_base * breadth / focus)))
    return recordings, objects


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
            1: "accepted_d1_resonant_bed",
            3: "spectral_halo",
            5: "pulsed_harmonic_current",
        }[int(dream_level)],
        "harmony_state": dict(HARMONY_STATE),
        "total_sample_selections": int(sum(usage_by_sample.values())),
        "unique_samples": int(len(usage_by_sample)),
        "samples": dict(sorted(usage_by_sample.items())),
        "sample_usage_details": dict(sorted(usage_details.items())),
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


def planned_form_items(form, dream_level):
    """Allocate density from the learned positive whole-render structure."""
    raw_items = []
    for section_name, _section_start, _section_end, density in form:
        adjusted = density * learned_factor("richness_weight", 3.2, 0.78, 1.25)
        adjusted *= form_density_multiplier(section_name, dream_level)
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
        quota = min(5, len([rec for rec in by_recording if LIBRARY_SOURCE_LABELS.get(rec) == "synthetic"]))
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
        quota = min(4, len([rec for rec in by_recording if LIBRARY_SOURCE_LABELS.get(rec) == "instrument_hybrid"]))
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

    scored = []

    for obj in pool:
        score = profile_distance(obj, profile)

        # General musical-value bias: favour clean, resonant, usable objects;
        # penalise harsh short noisy attacks.
        score *= musical_value(obj)
        score *= COMPOSITION_PREFERENCE.object_factor(
            obj.get("object_id"), obj.get("recording"), dream_level
        )

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


def arpeggio_frequencies(dream_level):
    """Return a level-specific pitch collection without imposing tonality."""
    count = {1: 4, 3: 6, 5: 8}[dream_level]
    tuned = scale_frequencies(low_midi=48, high_midi=78, count=count)
    if tuned is not None:
        return tuned
    free_midi = {
        1: (48, 55, 60, 64),
        3: (50, 55, 59, 62, 67, 72),
        5: (48, 51, 55, 58, 62, 65, 69, 72),
    }
    return [midi_to_hz(note) for note in free_midi[dream_level]]


def arpeggio_phrase_growth(dream_level, progress):
    """Let synth phrases emerge gradually instead of arriving at full level."""
    progress = max(0.0, min(1.0, float(progress)))
    floor = {1: 0.72, 3: 0.60, 5: 0.46}[dream_level]
    return float(floor + (1.0 - floor) * progress ** 0.72)


def synth_arpeggio_layer(dream_level, pulse_bpm=0.0, duration=None):
    """Build a bounded phrase-based arpeggio requested by human feedback."""
    duration = OUTPUT_DURATION if duration is None else max(0.0, float(duration))
    length = int(duration * TARGET_SR)
    layer = np.zeros((length, 2), dtype=np.float32)
    gain = composition_feedback_audio_snapshot()["arpeggio_layer_gain"]
    if gain <= 1e-6 or length < 32:
        return layer

    bpm = {1: 76.0, 3: 96.0, 5: float(pulse_bpm) if pulse_bpm > 0 else 126.0}[dream_level]
    subdivision = {1: 1.0, 3: 0.5, 5: 0.5}[dream_level]
    step_sec = (60.0 / bpm) * subdivision
    note_sec = min(1.15, max(0.18, step_sec * {1: 1.18, 3: 1.05, 5: 0.88}[dream_level]))
    phrase_steps = {1: 4, 3: 8, 5: 12}[dream_level]
    rest_steps = {1: 2, 3: 2, 5: 1}[dream_level]
    pattern_banks = {
        1: (
            (0, 2, 1, 3),
            (0, 1, 3, 2),
            (2, 0, 3, 1),
        ),
        3: (
            (0, 2, 4, 1, 3, 5, 4, 2),
            (0, 3, 1, 4, 2, 5, 3, 1),
            (4, 2, 0, 1, 5, 3, 1, 2),
        ),
        5: (
            (0, 2, 4, 6, 3, 5, 7, 4, 6, 2, 5, 1),
            (0, 3, 6, 2, 5, 1, 7, 4, 2, 6, 3, 5),
            (5, 2, 7, 3, 0, 4, 1, 6, 3, 7, 2, 4),
        ),
    }
    frequencies = arpeggio_frequencies(dream_level)
    rng = random.Random(RENDER_SEED + dream_level * 104729)
    pattern = rng.choice(pattern_banks[dream_level])
    timbre_variant = rng.randrange(3)
    start_sec = {1: 18.0, 3: 13.0, 5: 9.0}[dream_level]
    end_sec = max(start_sec, duration - {1: 18.0, 3: 12.0, 5: 10.0}[dream_level])
    note_index = 0
    level_amp = {1: 0.020, 3: 0.023, 5: 0.026}[dream_level] * gain

    while start_sec < end_sec:
        pitch_index = pattern[note_index % len(pattern)] % len(frequencies)
        freq = frequencies[pitch_index]
        phase = rng.random() * np.pi * 2.0
        samples = max(16, int(note_sec * TARGET_SR))
        t = np.arange(samples, dtype=np.float32) / TARGET_SR
        envelope_phase = np.sin(
            np.linspace(0.0, np.pi, samples, dtype=np.float32)
        )
        envelope = np.clip(envelope_phase, 0.0, 1.0) ** 1.7
        carrier = 2 * np.pi * freq * t + phase
        if timbre_variant == 0:
            tone = np.sin(carrier)
            tone += 0.22 * np.sin(2 * carrier + phase * 0.31)
            tone += 0.07 * np.sin(3 * carrier + phase * 0.13)
        elif timbre_variant == 1:
            tone = np.sin(carrier + 0.24 * np.sin(carrier * 0.5 + phase * 0.37))
            tone += 0.12 * np.sin(3 * carrier + phase * 0.19)
        else:
            tone = 0.82 * np.sin(carrier)
            tone += 0.16 * np.sin(carrier * 1.5 + phase * 0.43)
            tone += 0.09 * np.sin(carrier * 2.5 + phase * 0.17)
        growth = arpeggio_phrase_growth(
            dream_level, start_sec / max(end_sec, 1e-6)
        )
        tone *= envelope * level_amp * growth
        pan = max(-0.72, min(0.72, -0.55 + 1.10 * (pitch_index / max(1, len(frequencies) - 1))))
        add_to_output(layer, tone.astype(np.float32), start_sec, 1.0, pan)

        note_index += 1
        start_sec += step_sec
        if note_index % phrase_steps == 0:
            start_sec += rest_steps * step_sec

    return layer


def make_ambient_bed(output, dream_level, pulse_bpm=0.0):
    duration = OUTPUT_DURATION
    t = np.linspace(0, duration, OUTPUT_DURATION * TARGET_SR, endpoint=False)
    feedback_audio = composition_feedback_audio_snapshot()
    low_gain = np.sqrt(feedback_audio["low_frequency_gain"])

    # Preserve the accepted D1 render path exactly. D3 and D5 receive distinct
    # background instruments instead of transpositions of one shared sine bed.
    if dream_level == 1:
        freqs = scale_frequencies(low_midi=33, high_midi=57, count=5)
        if freqs is None:
            freqs = [55.00, 82.41, 110.00, 164.81, 220.00]
        amps = [0.024, 0.020, 0.016, 0.011, 0.007]
        for freq, amp in zip(freqs, amps):
            phase = random.random() * np.pi * 2
            slow = 0.5 + 0.5 * np.sin(2 * np.pi * t / random.uniform(35, 80))
            frequency_gain = low_gain if freq < 125 else 1.0
            tone = np.sin(2 * np.pi * freq * t + phase) * amp * slow * frequency_gain
            pan = random.uniform(-0.35, 0.35)
            ambient_gain = learned_factor("ambient_weight", 4.0)
            ambient_gain *= d5_temporal_profile(dream_level)["ambient_scale"]
            add_to_output(output, tone.astype(np.float32), 0, ambient_gain, pan)
    else:
        rng = random.Random(RENDER_SEED + dream_level * 65537)
        voice_count = 3 if dream_level == 3 else 4
        freqs = scale_frequencies(low_midi=34, high_midi=58, count=voice_count)
        if freqs is None:
            freqs = {
                3: [69.30, 103.83, 155.56],
                5: [49.00, 73.42, 110.00, 164.81],
            }[dream_level]
        amps = {
            3: [0.017, 0.012, 0.008],
            5: [0.014, 0.011, 0.008, 0.0055],
        }[dream_level]
        progress = np.clip(t / max(duration, 1e-6), 0.0, 1.0)
        broad_form = 0.22 + 0.78 * np.sin(np.pi * progress) ** 1.35
        for index, (freq, amp) in enumerate(zip(freqs, amps)):
            phase = rng.random() * np.pi * 2.0
            frequency_gain = low_gain if freq < 125 else 1.0
            if dream_level == 3:
                slow = 0.58 + 0.42 * np.sin(
                    2 * np.pi * t / rng.uniform(47.0, 91.0) + phase
                )
                carrier = np.sin(2 * np.pi * freq * t + phase)
                carrier += 0.10 * np.sin(2 * np.pi * freq * 2.5 * t + phase * 0.37)
                tone = carrier * amp * slow * broad_form * frequency_gain
                pan = (-0.48, 0.10, 0.52)[index]
            else:
                beat = 60.0 / max(1.0, float(pulse_bpm or 126.0))
                phrase_motion = 0.68 + 0.32 * np.sin(
                    2 * np.pi * t / (beat * (12.0 + index * 4.0)) + phase
                )
                phase_motion = 0.12 * np.sin(
                    2 * np.pi * t / rng.uniform(8.0, 17.0) + phase * 0.41
                )
                carrier = np.sin(2 * np.pi * freq * t + phase + phase_motion)
                carrier += 0.15 * np.sin(2 * np.pi * freq * 1.5 * t + phase * 0.63)
                tone = carrier * amp * phrase_motion * broad_form * frequency_gain
                pan = (-0.62, 0.38, -0.22, 0.64)[index]
            ambient_gain = learned_factor("ambient_weight", 4.0)
            ambient_gain *= d5_temporal_profile(dream_level)["ambient_scale"]
            add_to_output(output, tone.astype(np.float32), 0, ambient_gain, pan)

    # Explicit synth presence requested by the listener. Neutral feedback adds
    # nothing, so the Gate 1 sound remains unchanged without an explicit request.
    synth_gain = feedback_audio["synthetic_layer_gain"]
    synth_gain *= np.sqrt(feedback_audio["foreground_presence_gain"])
    if synth_gain > 1e-6:
        synth_freqs = scale_frequencies(low_midi=48, high_midi=79, count=4)
        if synth_freqs is None:
            synth_freqs = {
                1: [130.81, 196.00, 261.63, 392.00],
                3: [146.83, 220.00, 293.66, 440.00],
                5: [110.00, 164.81, 246.94, 369.99],
            }[dream_level]
        form_env = 0.35 + 0.65 * np.sin(np.pi * np.clip(t / OUTPUT_DURATION, 0.0, 1.0)) ** 1.6
        d5_evolution = d5_global_evolution_curve(len(t)) if dream_level == 5 else 1.0
        motion_rate = 0.035 + 0.018 * feedback_audio["development_drive"]
        for index, freq in enumerate(synth_freqs):
            phase = random.random() * np.pi * 2
            motion = 0.62 + 0.38 * np.sin(2 * np.pi * motion_rate * t + phase)
            carrier = 2 * np.pi * freq * t + phase
            if dream_level == 5:
                # The synthetic colour changes through the piece: FM motion and
                # upper partials follow the learned multi-stage development arc.
                progress = np.clip(t / max(OUTPUT_DURATION, 1e-6), 0.0, 1.0)
                fm_index = 0.08 + 0.28 * progress * feedback_audio["development_drive"]
                tone = np.sin(carrier + fm_index * np.sin(carrier * 0.5 + phase * 0.31))
                tone += (0.14 + 0.18 * progress) * np.sin(carrier * 2.0 + phase * 0.7)
                tone += (0.04 + 0.10 * (1.0 - progress)) * np.sin(carrier * 3.0 + phase * 0.19)
            else:
                tone = np.sin(carrier)
                tone += 0.24 * np.sin(carrier * 2.0 + phase * 0.7)
            if dream_level == 5:
                beat = 60.0 / max(1.0, float(pulse_bpm or 126.0))
                phrase_pulse = 0.62 + 0.38 * np.power(
                    0.5 + 0.5 * np.sin(2 * np.pi * t / (beat * 2.0) + phase),
                    1.6,
                )
                tone *= phrase_pulse * d5_evolution
            level_gain = 1.38 if dream_level == 5 else 1.0
            tone *= (0.010 + 0.003 * index) * synth_gain * form_env * motion * level_gain
            pan = (-0.62, 0.45, -0.28, 0.68)[index % 4]
            add_to_output(output, tone.astype(np.float32), 0, 1.0, pan)

    output += synth_arpeggio_layer(dream_level, pulse_bpm=pulse_bpm)



def smooth_tail(x, tail_sec=0.45, decay=0.45):
    """Add a very short decaying self-tail so phrases do not stop abruptly."""
    delay = max(1, int(tail_sec * TARGET_SR))
    out = np.zeros(len(x) + delay, dtype=np.float32)
    out[:len(x)] += x
    out[delay:delay + len(x)] += x * decay
    return out / (np.max(np.abs(out)) + 1e-9)


def air_resonance_layer(length, dream_level):
    """Very quiet high-frequency air, intermittent and slowly moving."""
    t = np.arange(length, dtype=np.float32) / TARGET_SR
    out = np.zeros(length, dtype=np.float32)

    # sparse high partials, more present in D3/D5 but still subtle
    partials = [7200, 9300, 11800]
    amps = [0.0028, 0.0022, 0.0016]
    if dream_level >= 5:
        partials += [13500]
        amps += [0.0011]

    for freq, amp in zip(partials, amps):
        phase = random.random() * np.pi * 2
        slow = 0.5 + 0.5 * np.sin(2 * np.pi * t / random.uniform(28, 70) + phase)
        gate = 0.5 + 0.5 * np.sin(2 * np.pi * t / random.uniform(18, 45) + phase * 0.37)
        gate = np.power(gate, 3.0)
        drift = np.sin(2 * np.pi * random.uniform(0.015, 0.045) * t + phase) * random.uniform(8, 35)
        out += np.sin(2 * np.pi * (freq + drift) * t + phase) * amp * slow * gate

    return out.astype(np.float32)


def low_resonance_pulse(length, dream_level):
    """Warm low-end support that appears as breaths, not constant bass."""
    t = np.arange(length, dtype=np.float32) / TARGET_SR
    out = np.zeros(length, dtype=np.float32)
    tonal_freqs = scale_frequencies(low_midi=29, high_midi=45, count=3)
    base_freqs = tonal_freqs if tonal_freqs is not None else [48, 72, 96]
    base_amp = 0.0045 if dream_level == 1 else (0.0060 if dream_level == 3 else 0.0070)
    base_amp *= composition_feedback_audio_snapshot()["low_frequency_gain"]

    for freq in base_freqs:
        phase = random.random() * np.pi * 2
        breath = 0.5 + 0.5 * np.sin(2 * np.pi * t / random.uniform(38, 85) + phase)
        breath = np.power(breath, 2.5)
        out += np.sin(2 * np.pi * freq * t + phase) * base_amp * breath

    return out.astype(np.float32)


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


def final_mix(output, dream_level):
    # Global ambience/resonance polish. Kept subtle: musical glue, not soup.
    length = len(output)

    air = air_resonance_layer(length, dream_level)
    low = low_resonance_pulse(length, dream_level)

    # Slightly different placement so the high air breathes in stereo.
    temporal = d5_temporal_profile(dream_level)
    ambient_scale = temporal["ambient_scale"]
    if dream_level == 5:
        ambient_scale *= max(0.76, 1.0 - 0.46 * (d5_energy_drive(5) - 1.0))
    output[:, 0] += (air * random.uniform(0.55, 0.85) + low * 0.80) * ambient_scale
    output[:, 1] += (np.roll(air, int(0.019 * TARGET_SR)) * random.uniform(0.55, 0.85) + low * 0.82) * ambient_scale

    output -= np.mean(output, axis=0)
    output = apply_mix_feedback_controls(output)

    # Gentle glue reverb. A little more in D5, but still controlled.
    wet = base_reverb_wet(dream_level) * ambient_scale
    wet *= composition_feedback_audio_snapshot()["reverb_clarity_gain"]
    output = simple_stereo_reverb(output, wet=wet)

    # Soft saturation for body, less aggressive than previous versions.
    output = np.tanh(output * 1.14)

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

    return output.astype(np.float32)


def role_sequence_for_section(section_name, dream_level):
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

    roles = [r for r, _ in table]
    weights = np.array([w for _, w in table], dtype=np.float64)
    weights /= weights.sum()

    idx = np.random.choice(len(roles), p=weights)
    return roles[idx]


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
    """Guarantee musical D5 attacks/releases after every transformation stage."""
    if dream_level != 5 or len(frag) < 32:
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
    smoothness = learned_factor("transition_smoothness_weight", 1.8, 0.90, 1.32)
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
        "gesture": 0.20,
        "impact": 0.16,
        "noise": 0.40,
        "texture": 1.40,
        "resonance": 2.20,
    }.get(role, 0.45) * request
    if gap >= 0.0:
        # Nearby phrases crossfade. Larger gaps retain breathing room but become
        # less empty, so continuity never turns into a constant wall of sound.
        if gap <= 5.0:
            adjusted = float(start) - min(gap + desired_overlap, max(0.25, duration * 0.22))
        else:
            adjusted = float(start) - min(gap * 0.22 * request, 2.0)
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
    eligible_roles = {"gesture", "texture", "noise"}
    if dream_level == 5:
        # At D5, sustained natural/organic material also needs to develop rather
        # than remain as an untouched bed. It receives a gentler granular mix.
        eligible_roles.add("resonance")
    if wet <= 1e-6 or role not in eligible_roles or len(frag) < 512:
        return np.asarray(frag, dtype=np.float32)

    role_scale = {"gesture": 0.88, "texture": 1.0, "noise": 0.76, "resonance": 0.62}[role]
    wet *= role_scale * {1: 0.62, 3: 0.82, 5: 1.0}[dream_level]
    drive = snapshot["structured_granulation_drive"]
    base_grain_ms = {1: 155.0, 3: 115.0, 5: 82.0}[dream_level]
    if role == "resonance":
        base_grain_ms *= 1.45
    grain_ms = base_grain_ms / max(1.0, 1.0 + 0.42 * (drive - 1.0))
    grain_n = max(128, min(len(frag), int(grain_ms * TARGET_SR / 1000.0)))
    hop_scale = {1: 0.72, 3: 0.58, 5: 0.46}[dream_level]
    hop_scale /= max(1.0, 1.0 + 0.22 * (drive - 1.0))
    hop_n = max(64, int(grain_n * hop_scale))
    window = np.hanning(grain_n).astype(np.float32)
    max_source = max(0, len(frag) - grain_n)
    drift = max(1, int(grain_n * (0.22 + 0.12 * max(0.0, drive - 1.0))))
    voice_count = (
        1
        + int(wet > 0.08)
        + int(dream_level == 5 and wet > 0.16)
        + int(dream_level == 5 and wet > 0.48)
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
    mixed = np.asarray(frag, dtype=np.float32) * (1.0 - wet) + granular * wet
    source_peak = float(np.max(np.abs(frag))) + 1e-9
    mixed_peak = float(np.max(np.abs(mixed))) + 1e-9
    if mixed_peak > source_peak:
        mixed *= source_peak / mixed_peak
    return mixed.astype(np.float32)


def transform_fragment_for_role(frag, role, dream_level):
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


def central_spectral_bloom(output, dream_level):
    """
    Compositional middle bloom: a controlled flowering of mid-frequency resonance.
    This is not random density; it creates a gradual spectral opening around the centre
    of the form, with warm middle partials and smooth transitions in/out.
    """
    if dream_level < 3:
        return output

    length = len(output)
    t = np.arange(length, dtype=np.float32) / TARGET_SR

    # Broad envelope: starts before the middle, peaks around 88s, fades naturally.
    start_t = 42.0
    peak_t = 88.0
    end_t = 132.0

    env = np.zeros_like(t)
    rise = (t >= start_t) & (t < peak_t)
    fall = (t >= peak_t) & (t <= end_t)
    env[rise] = 0.5 - 0.5 * np.cos(np.pi * (t[rise] - start_t) / max(1e-6, peak_t - start_t))
    env[fall] = 0.5 + 0.5 * np.cos(np.pi * (t[fall] - peak_t) / max(1e-6, end_t - peak_t))
    env = np.power(env, 1.35)

    # Mid-frequency harmonic field. When Max provides a reliable scale,
    # the bloom is constructed from that scale; otherwise the original field is retained.
    freqs = scale_frequencies(low_midi=52, high_midi=99, count=10)
    if freqs is None:
        base_freqs = [330, 392, 494, 587, 740, 880, 1175, 1480, 1760, 2217]
        ratio = {3: 0.982, 5: 1.018}[dream_level]
        freqs = [freq * ratio for freq in base_freqs]
    amps = [0.006, 0.006, 0.005, 0.005, 0.004, 0.004, 0.0032, 0.0028, 0.0024, 0.0020]

    if dream_level == 5:
        amp_scale = (
            1.08
            * learned_factor("bloom_weight", 5.0)
            / np.sqrt(d5_energy_drive(5))
        )
        drift_amt = 5.5
    else:
        amp_scale = 0.82 * learned_factor("bloom_weight", 5.0)
        drift_amt = 3.0

    left = np.zeros(length, dtype=np.float32)
    right = np.zeros(length, dtype=np.float32)

    for i, (freq, amp) in enumerate(zip(freqs, amps)):
        phase = random.random() * np.pi * 2
        slow = 0.55 + 0.45 * np.sin(2 * np.pi * t / random.uniform(18, 42) + phase)
        drift = np.sin(2 * np.pi * random.uniform(0.012, 0.040) * t + phase) * drift_amt
        tone = np.sin(2 * np.pi * (freq + drift) * t + phase) * amp * amp_scale * env * slow

        # alternating stereo spread, but keeping the bloom coherent in the centre
        pan = np.sin(i * 1.7) * 0.55
        stereo = pan_stereo(tone.astype(np.float32), pan)
        left += stereo[:, 0]
        right += stereo[:, 1]

    # Soft filtered noise excitation gives the bloom a living spectral body.
    noise = np.random.normal(0, 1, length).astype(np.float32) * (0.0018 if dream_level == 5 else 0.0010)
    noise = butter_filter(noise, "highpass", 420)
    noise = butter_filter(noise, "lowpass", 3400)
    noise *= env.astype(np.float32)

    output[:, 0] += left + noise * 0.55
    output[:, 1] += right + np.roll(noise, int(0.017 * TARGET_SR)) * 0.55
    return output


def generate_soundscape(dream_level):
    global CURRENT_MATERIAL_PLAN, CURRENT_FORM_VARIANT, LEARNING_WEIGHTS, REPRESENTATION_ASSIST, COMPOSITION_PREFERENCE
    global LEARNED_SYNTH_AFFINITY, ORIGIN_LEARNING_SNAPSHOT
    global CURRENT_LIBRARY_COVERAGE_SNAPSHOT
    CURRENT_MATERIAL_PLAN = None
    CURRENT_FORM_VARIANT = "aesthetic_bridge" if dream_level == 5 else "baseline"

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
    make_ambient_bed(output, dream_level, pulse_bpm=pulse_bpm)

    # A simple composed form: each section has a role tendency.
    form = [
        ("opening", 0, 32, 0.65),
        ("activation", 24, 62, 1.05),
        ("complexity", 52, 102, 1.45),
        ("memory", 90, 138, 1.05),
        ("resolution", 125, 178, 0.75),
    ]

    previous = None
    used = set()
    motif_bank = []
    usage_counts = {}
    usage_by_sample = {}
    usage_details = {}
    event_durations = []
    foreground_durations = []
    event_starts = []
    total_added = 0
    role_counts = {"gesture": 0, "texture": 0, "resonance": 0, "noise": 0, "impact": 0}
    role_last_end = {role: None for role in role_counts}

    section_item_counts = planned_form_items(form, dream_level)

    for (section_name, section_start, section_end, _density), items in zip(
        form, section_item_counts
    ):
        section_length = section_end - section_start

        for i in range(items):
            role = role_sequence_for_section(section_name, dream_level)

            # Keep one palette across the form. Sections differ through role,
            # transformation and density rather than unrelated new families.
            preferred_groups = None

            # Coherence: sometimes return to a previous object from the same role.
            # This creates motif-like recurrence instead of unrelated new material.
            same_role_motifs = [m for m in motif_bank if m.get("role") == role]
            repeat_chance = (
                {1: 0.25, 3: 0.26, 5: 0.22}[dream_level]
                * learned_factor("coherence_weight", 4.0, 0.70, 1.35)
                * learned_factor("material_development_weight", 1.8, 0.85, 1.30)
                / learned_factor("repetition_control", 3.0, 0.70, 1.45)
            )
            _, unique_object_limit = material_plan_limits(dream_level)
            palette_is_full = len(used) >= unique_object_limit
            use_motif = bool(same_role_motifs) and (
                palette_is_full or random.random() < repeat_chance
            )

            if use_motif:
                obj = random.choice(same_role_motifs)
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
                motif_limit = {1: 6, 3: 10, 5: 14}[dream_level]
                if len(motif_bank) < motif_limit and role in ["gesture", "texture", "resonance"]:
                    motif_obj = dict(obj)
                    motif_obj["role"] = role
                    motif_bank.append(motif_obj)

            previous = obj

            frag = load_fragment(obj)
            if frag is None:
                continue

            frag, amp = transform_fragment_for_role(frag, role, dream_level)
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
            event_duration = len(frag) / TARGET_SR
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

    output = central_spectral_bloom(output, dream_level)
    output = final_mix(output, dream_level)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(
        OUTPUT_FOLDER,
        f"Hyponoia_v3_memory_bloom_smooth_D{dream_level}_{timestamp}.wav"
    )

    sf.write(outfile, output, TARGET_SR)

    current_file = os.path.join(OUTPUT_FOLDER, "current.wav")
    sf.write(current_file, output, TARGET_SR)

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
    )
    save_sample_learning_profile(sample_profile)

    print("Current:")
    print(current_file)
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

# Hyponoia AI Composer v3 + Memory v3 — richer musical pass
# More active D5, phrase evolution, musical delay/reverb, and less over-soft selection.
import os
import json
import random
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

MEMORY_FILE = "memory_index_v3.json"
MEMORY_FOLDER = "alpha_memory"
OUTPUT_FOLDER = "output"
PROFILE_FILE = "alpha_profile.json"
LEARNING_FILE = "learning_profile.json"
SAMPLE_LEARNING_FILE = "sample_learning_profile.json"
RENDER_REPORT_FILE = "render_report.json"
RENDER_REPORT_FOLDER = "render_reports"
GENERATOR_REVISION = "2026-08-21-d5-aesthetic-bridge-5.1"

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
    "activity_weight": 1.0,
    "material_development_weight": 1.0,
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

# IMPORTANT: every render should be different.
# We explicitly seed from current time + process id, so each run produces a new variation.
RENDER_SEED = (datetime.now().microsecond + os.getpid() + random.SystemRandom().randint(0, 999999))
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
    """Translate D5 activity learning into audible speed, not only more layers.

    D1 and D3 deliberately return a neutral profile. For D5, the learned
    activity control shortens time-stretch ratios, emergence envelopes and
    delay spacing while slightly reducing the slow ambient wash.
    """
    if dream_level != 5:
        return {
            "temporal_drive": 1.0,
            "stretch_scale": 1.0,
            "envelope_scale": 1.0,
            "delay_scale": 1.0,
            "ambient_scale": 1.0,
        }

    # The preferred 20-August render breathed more freely than the later
    # pulse-driven revisions. Activity therefore creates gentle forward motion
    # without shortening phrases or stripping away the ambient body.
    activity = learned_factor("activity_weight", 0.65, 0.96, 1.32)
    temporal_drive = float(max(1.04, min(1.34, 1.03 * activity)))
    return {
        "temporal_drive": temporal_drive,
        "stretch_scale": max(0.94, 1.0 - 0.12 * (temporal_drive - 1.0)),
        "envelope_scale": max(0.96, 1.0 - 0.08 * (temporal_drive - 1.0)),
        "delay_scale": max(0.84, 1.0 - 0.28 * (temporal_drive - 1.0)),
        "ambient_scale": max(0.94, 1.0 - 0.16 * (temporal_drive - 1.0)),
    }


def d5_development_drive(dream_level):
    """Bounded development strength for transformations of the chosen palette."""
    if dream_level != 5:
        return 1.0
    development = learned_factor("material_development_weight", 2.4, 0.92, 1.42)
    synthetic = learned_factor("synthetic_material_weight", 1.4, 0.94, 1.25)
    return float(np.sqrt(development * synthetic))


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
    synthetic = max(0.0, min(1.0, float(features.get("synthetic_score", 0.5))))
    energetic_quality = 0.40 * musicality + 0.34 * gesture + 0.26 * energy
    drive = d5_energy_drive(dream_level)
    synthetic_drive = learned_factor("synthetic_material_weight", 4.0, 0.85, 1.50)
    energetic_factor = 1.16 - 0.24 * energetic_quality * drive
    synthetic_factor = 1.12 - 0.20 * synthetic * synthetic_drive
    focused = max(0.52, energetic_factor * synthetic_factor)
    # Keep some present-day preference for musical synthetic material, but
    # retain the broader, airier palette of the listener-preferred baseline.
    return float(1.0 + (focused - 1.0) * 0.55)


def material_plan_limits(dream_level):
    """Balanced per-render palette: focused, but never reduced to a tiny loop."""
    recording_base = {1: 6, 3: 10, 5: 16}[dream_level]
    object_base = {1: 12, 3: 20, 5: 30}[dream_level]
    focus = learned_factor("material_development_weight", 1.2, 0.88, 1.18)
    recordings = max(3, int(round(recording_base / focus)))
    objects = max(recordings, int(round(object_base / focus)))
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
        data.setdefault("samples", {})
        migrated, migration_report = migrate_sample_profile(data, memory)
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
        "harmony_state": dict(HARMONY_STATE),
        "total_sample_selections": int(sum(usage_by_sample.values())),
        "unique_samples": int(len(usage_by_sample)),
        "samples": dict(sorted(usage_by_sample.items())),
        "sample_usage_details": dict(sorted(usage_details.items())),
        "recordings": dict(sorted(usage_by_recording.items())),
        "role_counts": role_counts,
        "control_snapshot": dict(LEARNING_WEIGHTS),
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
    synthetic_score = f.get(
        "synthetic_score",
        max(0.0, min(1.0, 0.55 * harmonicity + 0.25 * f.get("pitch_confidence", 0.0) + 0.20 * resonance_strength)),
    )

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


def build_material_plan(objects, profile, dream_level, sample_profile=None):
    by_recording = {}

    for obj in objects:
        rec = obj["recording"]
        by_recording.setdefault(rec, []).append(obj)

    recording_scores = []
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
        recording_scores.append((np.mean(distances) + history_penalty, rec))

    recording_scores.sort(key=lambda x: x[0])

    good = [r for _, r in recording_scores[:25]]
    wider = [r for _, r in recording_scores[25:80]]

    core = random.sample(good, min(CORE_COUNT, len(good)))

    candidates_d3 = [r for r in good + wider if r not in core]
    extra_d3 = random.sample(candidates_d3, min(EXTRA_D3_COUNT, len(candidates_d3)))

    candidates_d5 = [r for r in good + wider if r not in core + extra_d3]
    extra_d5 = random.sample(candidates_d5, min(EXTRA_D5_COUNT, len(candidates_d5)))

    if dream_level == 1:
        allowed = core
    elif dream_level == 3:
        allowed = core + extra_d3
    else:
        allowed = core + extra_d3 + extra_d5

    recording_limit, _ = material_plan_limits(dream_level)
    if len(allowed) > recording_limit:
        allowed = random.sample(allowed, recording_limit)

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

    scored = []

    for obj in pool:
        score = profile_distance(obj, profile)

        # General musical-value bias: favour clean, resonant, usable objects;
        # penalise harsh short noisy attacks.
        score *= musical_value(obj)

        # The D5 preference controls must make an audible selection difference,
        # while D1/D3 retain their established behaviour.
        score *= d5_selection_character_factor(obj, dream_level)

        # Harmonic listening: pitched objects inside the detected scale are preferred.
        # Unpitched/noisy objects remain available and are not forcibly quantised.
        score *= scale_selection_factor(obj)

        if sample_profile is not None:
            score *= sample_learning_factor(obj, sample_profile)
            score *= sample_exploration_factor(obj, sample_profile)

        if previous is not None:
            score += continuity_distance(previous, obj) * 0.24 * learned_factor("transition_smoothness_weight", 5.0) * learned_factor("coherence_weight", 4.0)

        # Anti-stuck behaviour: recordings can return as motifs,
        # but the composer should not cling to the same few recordings forever.
        if usage_counts is not None:
            score *= (1.0 + 0.09 * usage_counts.get(obj["recording"], 0))

        dur = obj["duration"]
        role = obj.get("role", classify_object(obj))
        f = obj["features"]

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


def make_ambient_bed(output, dream_level):
    duration = OUTPUT_DURATION
    t = np.linspace(0, duration, OUTPUT_DURATION * TARGET_SR, endpoint=False)

    freqs = scale_frequencies(low_midi=33, high_midi=57, count=5)
    if freqs is None:
        freqs = [55, 82.5, 110, 165, 220]
    amps = [0.024, 0.020, 0.016, 0.011, 0.007]

    for freq, amp in zip(freqs, amps):
        phase = random.random() * np.pi * 2
        slow = 0.5 + 0.5 * np.sin(2 * np.pi * t / random.uniform(35, 80))
        tone = np.sin(2 * np.pi * freq * t + phase) * amp * slow

        if dream_level >= 5:
            tone += np.sin(2 * np.pi * (freq * 1.5) * t + phase) * amp * 0.16

        pan = random.uniform(-0.35, 0.35)
        ambient_gain = learned_factor("ambient_weight", 4.0)
        ambient_gain *= d5_temporal_profile(dream_level)["ambient_scale"]
        add_to_output(output, tone.astype(np.float32), 0, ambient_gain, pan)



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

    rev[:, 0] = butter_filter(rev[:, 0], "lowpass", 9500)
    rev[:, 1] = butter_filter(rev[:, 1], "lowpass", 9500)

    return (output * (1.0 - wet) + rev * wet).astype(np.float32)

def final_mix(output, dream_level):
    # Global ambience/resonance polish. Kept subtle: musical glue, not soup.
    length = len(output)

    air = air_resonance_layer(length, dream_level)
    low = low_resonance_pulse(length, dream_level)

    # Slightly different placement so the high air breathes in stereo.
    temporal = d5_temporal_profile(dream_level)
    ambient_scale = temporal["ambient_scale"]
    output[:, 0] += (air * random.uniform(0.55, 0.85) + low * 0.80) * ambient_scale
    output[:, 1] += (np.roll(air, int(0.019 * TARGET_SR)) * random.uniform(0.55, 0.85) + low * 0.82) * ambient_scale

    output -= np.mean(output, axis=0)

    # Gentle glue reverb. A little more in D5, but still controlled.
    wet = {1: 0.095, 3: 0.135, 5: 0.175}[dream_level] * ambient_scale
    output = simple_stereo_reverb(output, wet=wet)

    # Soft saturation for body, less aggressive than previous versions.
    output = np.tanh(output * 1.14)

    peak = np.max(np.abs(output)) + 1e-9
    output = output / peak * 0.88

    if dream_level == 5:
        fade_in = int(5.5 * TARGET_SR)
        fade_out = int(36.0 * TARGET_SR)
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
    table = [(role, weight * role_learning.get(role, 1.0)) for role, weight in table]

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
        depth = min(0.20, 0.10 * development)
    elif role == "noise":
        rate = random.uniform(1.4, 2.8) * temporal_drive
        depth = min(0.18, 0.09 * development)
    elif role == "texture":
        rate = random.uniform(0.30, 0.78) * temporal_drive
        depth = min(0.13, 0.065 * development)
    else:  # resonance
        rate = random.uniform(0.12, 0.36) * temporal_drive
        depth = min(0.10, 0.05 * development)

    rate *= section_rate
    phase = random.uniform(0.0, 2.0 * np.pi)
    time = np.arange(len(frag), dtype=np.float32) / TARGET_SR
    pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * rate * time + phase)
    pulse = np.power(pulse, 1.35)
    motion = (1.0 - depth) + depth * pulse
    return (frag * motion.astype(np.float32)).astype(np.float32)


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

def transform_fragment_for_role(frag, role, dream_level):
    """
    Different musical functions receive different treatments.
    This prevents all layers from dissolving into the same texture.
    """
    temporal = d5_temporal_profile(dream_level)
    stretch_scale = temporal["stretch_scale"]
    envelope_scale = temporal["envelope_scale"]

    if role == "gesture":
        # D5 gestures should be clear phrases, not tiny decorative sparks.
        if dream_level == 5:
            stretch = random.uniform(0.75, 1.35)
            stretch *= stretch_scale
            fade_time = random.uniform(0.20, 0.75) * envelope_scale
            amp = random.uniform(0.135, 0.220)
        else:
            stretch = random.uniform(0.65, 1.75)
            fade_time = random.uniform(0.15, 0.9)
            amp = random.uniform(0.125, 0.215)

        frag = stretch_audio(frag, stretch)
        if dream_level >= 3 and random.random() < 0.28:
            frag = reverse_blend(frag, amount=random.uniform(0.10, 0.25))
        frag = clean_band(frag, low=60, high=12500)
        frag = fade(frag, sec=fade_time)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.18, 0.42), decay=random.uniform(0.18, 0.35))

    elif role == "impact":
        stretch = random.uniform(0.65, 1.20) * stretch_scale if dream_level == 5 else random.uniform(0.55, 1.25)
        frag = stretch_audio(frag, stretch)
        if random.random() < 0.22:
            frag = reverse_blend(frag, amount=random.uniform(0.10, 0.22))
        frag = clean_band(frag, low=45, high=13000)
        frag = fade(frag, sec=random.uniform(0.08, 0.55) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.10, 0.30), decay=random.uniform(0.12, 0.25))
        amp = random.uniform(0.120, 0.205) if dream_level == 5 else random.uniform(0.13, 0.24)

    elif role == "noise":
        stretch = random.uniform(1.4, 3.2) * stretch_scale if dream_level == 5 else random.uniform(1.2, 3.8 + dream_level * 0.25)
        frag = stretch_audio(frag, stretch)
        if dream_level >= 3:
            frag = reverse_blend(frag, amount=random.uniform(0.20, 0.45))
        frag = clean_band(frag, low=80, high=14000)
        frag = fade(frag, sec=random.uniform(0.9, 2.8) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.25, 0.75), decay=random.uniform(0.18, 0.36))
        amp = random.uniform(0.065, 0.120) if dream_level == 5 else random.uniform(0.060, 0.125)

    elif role == "resonance":
        # Long layer, but quieter and more distinct in D5.
        stretch = random.uniform(3.4, 7.6) * stretch_scale if dream_level == 5 else random.uniform(3.0, 7.5 + dream_level * 0.45)
        frag = stretch_audio(frag, stretch)
        if dream_level >= 3 and random.random() < (0.55 if dream_level == 5 else 1.0):
            frag = harmonic_bloom(frag, dream_level)
        frag = clean_band(frag, low=28, high=11200)
        frag = fade(frag, sec=random.uniform(4.5, 9.5) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.8, 1.8), decay=random.uniform(0.22, 0.45))
        amp = random.uniform(0.060, 0.125) if dream_level == 5 else random.uniform(0.055, 0.12)

    else:  # texture
        stretch = random.uniform(2.4, 5.8) * stretch_scale if dream_level == 5 else random.uniform(2.0, 5.5 + dream_level * 0.35)
        frag = stretch_audio(frag, stretch)
        if dream_level >= 3 and random.random() < (0.45 if dream_level == 5 else 0.65):
            frag = harmonic_bloom(frag, dream_level)
        frag = clean_band(frag, low=34, high=12000)
        frag = fade(frag, sec=random.uniform(2.8, 6.8) * envelope_scale)
        frag = smooth_tail(frag, tail_sec=random.uniform(0.45, 1.2), decay=random.uniform(0.18, 0.38))
        amp = random.uniform(0.075, 0.150) if dream_level == 5 else random.uniform(0.070, 0.145)

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

    if dream_level == 5:
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
        freqs = [330, 392, 494, 587, 740, 880, 1175, 1480, 1760, 2217]
    amps = [0.006, 0.006, 0.005, 0.005, 0.004, 0.004, 0.0032, 0.0028, 0.0024, 0.0020]

    if dream_level == 5:
        amp_scale = 1.25 * learned_factor("bloom_weight", 5.0)
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
    global CURRENT_MATERIAL_PLAN, CURRENT_FORM_VARIANT, LEARNING_WEIGHTS
    CURRENT_MATERIAL_PLAN = None
    CURRENT_FORM_VARIANT = "aesthetic_bridge" if dream_level == 5 else "baseline"

    print("Generator revision:", GENERATOR_REVISION)
    print("Render seed:", RENDER_SEED)
    print("Form variant:", CURRENT_FORM_VARIANT)
    LEARNING_WEIGHTS = load_learning_weights(dream_level)
    pulse_bpm = random.uniform(*D5_REFERENCE_TARGETS["pulse_bpm_range"]) if dream_level == 5 else 0.0
    if dream_level == 5:
        print("Reference pulse BPM:", round(pulse_bpm, 3))

    profile = load_profile()
    objects = load_memory_objects()
    with open(MEMORY_FILE, "r", encoding="utf-8") as handle:
        memory = json.load(handle)
    sample_profile = load_sample_learning_profile(memory)
    build_role_pools(objects)

    output = np.zeros((OUTPUT_DURATION * TARGET_SR, 2), dtype=np.float32)
    make_ambient_bed(output, dream_level)

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

    for section_name, section_start, section_end, density in form:
        section_length = section_end - section_start
        density *= learned_factor("richness_weight", 3.2, 0.78, 1.25)
        density *= form_density_multiplier(section_name, dream_level)

        # Middle bloom: the central section should feel like a flower opening,
        # with more independent phrase activity but smooth transitions.
        if dream_level == 5 and section_name == "complexity":
            density *= 1.34
        elif dream_level == 3 and section_name == "complexity":
            density *= 1.16

        # D5: not more events; clearer simultaneous phrase-layers.
        # More complexity should come from distinct musical functions, not clutter.
        if dream_level == 5:
            items = int(15 * density * 1.23)
        elif dream_level == 3:
            items = int(12 * density * 1.28)
        else:
            items = int(11 * density * 1.15)
        items = max(1, int(items * dream_activity_multiplier(dream_level)))

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
            frag = d5_internal_motion(frag, role, dream_level, section_name)
            if dream_level == 5:
                # Shape an audible formal energy arc without merely normalising
                # the entire render louder.
                section_gain = {
                    "opening": 0.96,
                    "activation": 1.03,
                    "complexity": 1.07,
                    "memory": 1.01,
                    "resolution": 0.94,
                }[section_name]
                amp *= 1.0 + (section_gain - 1.0) * d5_energy_drive(dream_level)
            if use_motif:
                frag = maybe_variation_transform(frag, role, dream_level)
                amp *= random.uniform(0.82, 1.05)

            # Final smoothing pass: delays first, then an organic emergence envelope,
            # so phrases feel as if they grow from previous material rather than being inserted.
            frag = musical_delay_tail(frag, role, dream_level)
            frag = organic_emergence(frag, role, dream_level, obj.get("features", {}))
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

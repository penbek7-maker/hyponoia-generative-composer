import json
import random

import numpy as np
import soundfile as sf

import build_alpha_profile
import critic_v2
import generator_v3_memory_bloom_smooth as generator
import memory_builder_v3
from hyponoia_stability import TARGET_SR


def _sine(sample_rate, duration, frequency=220.0, amplitude=0.2):
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * time)


def test_profile_builder_and_critic_use_48khz(tmp_path):
    source = tmp_path / "source_44100.wav"
    sf.write(source, _sine(44_100, 3.0), 44_100)
    profile = build_alpha_profile.analyse_reference(str(source))
    report = critic_v2.analyse_audio(str(source))
    assert TARGET_SR == 48_000
    assert profile["sample_rate"] == 48_000
    assert report["analysis_sample_rate"] == 48_000


def test_memory_builder_produces_stable_ids_and_inventory(tmp_path, monkeypatch):
    memory_folder = tmp_path / "memory"
    memory_folder.mkdir()
    sf.write(memory_folder / "tone.wav", _sine(44_100, 2.0), 44_100)
    output = tmp_path / "memory_index.json"
    report_path = tmp_path / "memory_report.json"
    monkeypatch.setattr(memory_builder_v3, "MEMORY_FOLDER", str(memory_folder))
    monkeypatch.setattr(memory_builder_v3, "OUTPUT_FILE", str(output))
    monkeypatch.setattr(memory_builder_v3, "REPORT_FILE", str(report_path))
    memory_builder_v3.build_memory()
    memory = json.loads(output.read_text())
    report = json.loads(report_path.read_text())
    assert memory[0]["recording_id"].startswith("rec_")
    assert memory[0]["sample_rate"] == 48_000
    assert memory[0]["objects"][0]["stable_id"].startswith("obj_")
    assert report["target_sample_rate"] == 48_000
    assert report["failed_recordings"] == []


def test_generator_recording_cache_reuses_loaded_wav(tmp_path, monkeypatch):
    sf.write(tmp_path / "tone.wav", _sine(48_000, 2.0), 48_000)
    monkeypatch.setattr(generator, "MEMORY_FOLDER", str(tmp_path))
    generator._load_recording.cache_clear()
    obj = {"recording": "tone.wav", "start": 0.0, "end": 1.0}
    first = generator.load_fragment(obj)
    second = generator.load_fragment(obj)
    info = generator._load_recording.cache_info()
    assert len(first) == len(second) == 48_000
    assert info.misses == 1
    assert info.hits == 1


def test_frequency_stems_reconstruct_the_master_and_separate_bands():
    sample_rate = generator.TARGET_SR
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    master = (
        0.20 * np.sin(2 * np.pi * 80.0 * time)
        + 0.14 * np.sin(2 * np.pi * 1000.0 * time)
        + 0.08 * np.sin(2 * np.pi * 8000.0 * time)
    ).astype(np.float32)
    stereo = np.stack([master, master * 0.85], axis=1)

    low, mid, high = generator.split_frequency_stems(stereo)

    assert low.shape == mid.shape == high.shape == stereo.shape
    assert np.isfinite(low).all() and np.isfinite(mid).all() and np.isfinite(high).all()
    assert np.max(np.abs(low)) > 0
    assert np.max(np.abs(mid)) > 0
    assert np.max(np.abs(high)) > 0
    assert np.allclose(low + mid + high, stereo, atol=2e-6)


def test_frequency_stems_are_saved_for_timestamped_and_current_outputs(tmp_path):
    samples = generator.TARGET_SR // 10
    time = np.arange(samples, dtype=np.float32) / generator.TARGET_SR
    master = np.stack([
        0.2 * np.sin(2 * np.pi * 100.0 * time),
        0.2 * np.sin(2 * np.pi * 1200.0 * time),
    ], axis=1).astype(np.float32)
    outfile = tmp_path / "Hyponoia_D1_test.wav"
    current = tmp_path / "current.wav"

    metadata = generator.save_frequency_stems(master, str(outfile), str(current))

    paths = list(metadata["timestamped"].values()) + list(metadata["current"].values())
    assert all((tmp_path / path.split("/")[-1]).exists() for path in paths)
    assert metadata["low_crossover_hz"] == 250
    assert metadata["high_crossover_hz"] == 4000
    low, _ = sf.read(metadata["current"]["low"], dtype="float32")
    mid, _ = sf.read(metadata["current"]["mid"], dtype="float32")
    high, _ = sf.read(metadata["current"]["high"], dtype="float32")
    assert np.allclose(low + mid + high, master, atol=2e-6)


def test_dream_activity_and_bright_event_smoothing_are_ordered(monkeypatch):
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    assert generator.dream_activity_multiplier(1) < generator.dream_activity_multiplier(3)
    assert generator.dream_activity_multiplier(3) < generator.dream_activity_multiplier(5)

    dark = generator.emergence_envelope_scales({
        "brightness": 1800.0,
        "foreground_probability": 0.2,
        "transient_score": 0.2,
    })
    bright = generator.emergence_envelope_scales({
        "brightness": 9000.0,
        "foreground_probability": 0.9,
        "transient_score": 0.85,
    })
    assert bright["fade_in"] > dark["fade_in"]
    assert bright["fade_out"] > dark["fade_out"]


def test_composition_feedback_reduces_low_masking_and_opens_layers(monkeypatch):
    sample_rate = generator.TARGET_SR
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low = np.sin(2 * np.pi * 80.0 * time).astype(np.float32) * 0.40
    detail = np.sin(2 * np.pi * 1200.0 * time).astype(np.float32) * 0.12
    stereo = np.stack([low + detail, low - detail], axis=1)

    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    neutral_output = generator.apply_mix_feedback_controls(stereo)

    requested = dict(neutral)
    requested.update({
        "low_frequency_control": 1.12,
        "layer_clarity_weight": 1.12,
        "synthetic_material_weight": 1.08,
        "material_development_weight": 1.10,
    })
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    adjusted = generator.apply_mix_feedback_controls(stereo)
    snapshot = generator.composition_feedback_audio_snapshot()

    neutral_low = generator.butter_filter(neutral_output[:, 0], "lowpass", 170)
    adjusted_low = generator.butter_filter(adjusted[:, 0], "lowpass", 170)
    neutral_side = (neutral_output[:, 0] - neutral_output[:, 1]) * 0.5
    adjusted_side = (adjusted[:, 0] - adjusted[:, 1]) * 0.5
    assert np.sqrt(np.mean(adjusted_low ** 2)) < np.sqrt(np.mean(neutral_low ** 2))
    assert np.sqrt(np.mean(adjusted_side ** 2)) > np.sqrt(np.mean(neutral_side ** 2))
    assert snapshot["synthetic_layer_gain"] > 0
    assert snapshot["development_drive"] > 1.0


def test_arpeggio_feedback_creates_bounded_source_derived_phrases(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    rng = np.random.default_rng(7)
    source = rng.normal(0.0, 0.08, (generator.TARGET_SR * 4, 2)).astype(np.float32)
    silent = generator.source_musical_phrase_layer(source, 3, duration=4.0)
    assert silent.shape == (generator.TARGET_SR * 4, 2)
    assert np.count_nonzero(silent) == 0

    requested = dict(neutral)
    requested["arpeggio_weight"] = 1.10
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    d3 = generator.source_musical_phrase_layer(source, 3, duration=4.0)
    d5 = generator.source_musical_phrase_layer(source, 5, pulse_bpm=126.0, duration=4.0)
    assert np.isfinite(d3).all() and np.isfinite(d5).all()
    assert np.max(np.abs(d3)) <= 0.22
    assert np.max(np.abs(d5)) <= 0.22
    assert np.count_nonzero(d3) > 0
    assert not np.array_equal(d3, d5)


def test_source_phrase_motif_and_transformations_change_between_render_seeds(monkeypatch):
    requested = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    requested["arpeggio_weight"] = 1.10
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    rng = np.random.default_rng(11)
    source = rng.normal(0.0, 0.08, (generator.TARGET_SR * 4, 2)).astype(np.float32)
    monkeypatch.setattr(generator, "RENDER_SEED", 101)
    first = generator.source_musical_phrase_layer(source, 1, duration=4.0)
    monkeypatch.setattr(generator, "RENDER_SEED", 202)
    second = generator.source_musical_phrase_layer(source, 1, duration=4.0)
    assert not np.array_equal(first, second)


def test_arpeggio_phrases_emerge_gradually():
    for dream_level in (1, 3, 5):
        early = generator.arpeggio_phrase_growth(dream_level, 0.1)
        middle = generator.arpeggio_phrase_growth(dream_level, 0.5)
        late = generator.arpeggio_phrase_growth(dream_level, 0.9)
        assert 0.0 < early < middle < late <= 1.0


def test_long_layers_rotate_between_levels_and_repeated_sources_are_penalised(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights["long_layer_diversity_weight"] = 1.10
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    by_family = {}
    for index in range(100):
        recording = f"source-{index}.wav"
        family = generator.deterministic_group(
            f"hyponoia-long-layer-v1:{recording}", groups=3
        )
        by_family.setdefault(family, recording)
        if len(by_family) == 3:
            break

    for dream_level, target_family in ((1, 0), (3, 1), (5, 2)):
        preferred = {
            "recording": by_family[target_family],
            "duration": 8.0,
            "role": "resonance",
            "features": {},
        }
        other = {
            "recording": by_family[(target_family + 1) % 3],
            "duration": 8.0,
            "role": "resonance",
            "features": {},
        }
        preferred_factor = generator.long_layer_diversity_factor(
            preferred, dream_level, usage_counts={}
        )
        other_factor = generator.long_layer_diversity_factor(
            other, dream_level, usage_counts={}
        )
        repeated_factor = generator.long_layer_diversity_factor(
            preferred, dream_level, usage_counts={preferred["recording"]: 5}
        )
        very_long = dict(preferred, duration=22.0)
        very_long_factor = generator.long_layer_diversity_factor(
            very_long, dream_level, usage_counts={}
        )
        assert preferred_factor < other_factor
        assert repeated_factor > preferred_factor
        assert very_long_factor > preferred_factor


def test_explicit_long_layer_diversity_bounds_and_varies_sustained_windows(monkeypatch):
    fragment = np.linspace(-1.0, 1.0, generator.TARGET_SR * 30, dtype=np.float32)
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    assert np.array_equal(
        generator.bound_sustained_fragment(fragment, "resonance", 1), fragment
    )

    requested = dict(neutral)
    requested["long_layer_diversity_weight"] = 1.10
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    d1 = generator.bound_sustained_fragment(fragment, "resonance", 1)
    d3 = generator.bound_sustained_fragment(fragment, "resonance", 3)
    d5 = generator.bound_sustained_fragment(fragment, "resonance", 5)
    assert len(d5) < len(d3) < len(d1) < len(fragment)
    assert d1[0] < d3[0] < d5[0]


def test_clearer_base_reverb_is_bounded_and_level_ordered():
    assert generator.base_reverb_wet(1) < generator.base_reverb_wet(3)
    assert generator.base_reverb_wet(3) < generator.base_reverb_wet(5)
    assert generator.base_reverb_wet(5) <= 0.125


def test_explicit_smoothness_feedback_adds_role_aware_release(monkeypatch):
    fragment = np.ones(generator.TARGET_SR * 3, dtype=np.float32)
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    unchanged = generator.feedback_release_guard(fragment.copy(), "texture")
    assert np.array_equal(unchanged, fragment)

    smoother = dict(neutral)
    smoother["transition_smoothness_weight"] = 1.15
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", smoother)
    released = generator.feedback_release_guard(fragment.copy(), "texture")
    assert released[-1] == 0.0
    assert released[-generator.TARGET_SR // 2] < 0.9
    assert released[0] == 1.0


def test_explicit_smoothness_adds_damped_tail_and_related_overlap(monkeypatch):
    fragment = np.linspace(-0.4, 0.4, generator.TARGET_SR, dtype=np.float32)
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    assert np.array_equal(generator.feedback_release_tail(fragment, "texture"), fragment)
    assert generator.feedback_continuity_start(8.0, 4.0, "texture", 6.0) == 8.0

    smoother = dict(neutral)
    smoother["transition_smoothness_weight"] = 1.15
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", smoother)
    tailed = generator.feedback_release_tail(fragment, "texture")
    overlapped = generator.feedback_continuity_start(8.0, 4.0, "texture", 6.0)
    assert len(tailed) > len(fragment)
    assert np.isfinite(tailed).all()
    assert np.any(np.abs(tailed[len(fragment):]) > 0)
    assert 0.0 <= overlapped < 8.0


def test_d5_energy_and_character_controls_are_level_specific(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    neutral_drive = generator.d5_energy_drive(5)
    assert generator.d5_energy_drive(1) == 1.0
    assert generator.d5_energy_drive(3) == 1.0

    elevated = dict(neutral)
    elevated.update({
        "activity_weight": 1.24,
        "musicality_weight": 1.08,
        "synthetic_material_weight": 1.12,
    })
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", elevated)
    assert generator.d5_energy_drive(5) > neutral_drive

    energetic_synthetic = {
        "features": {
            "energy": 0.15,
            "musicality": 0.90,
            "gesture_strength": 0.90,
            "synthetic_score": 0.90,
        }
    }
    weak_acoustic = {
        "features": {
            "energy": 0.02,
            "musicality": 0.30,
            "gesture_strength": 0.20,
            "synthetic_score": 0.10,
        }
    }
    assert generator.d5_selection_character_factor(energetic_synthetic, 5) < generator.d5_selection_character_factor(weak_acoustic, 5)
    assert generator.d5_selection_character_factor(energetic_synthetic, 3) == 1.0


def test_learned_synth_feedback_changes_object_selection_without_affecting_neutral(monkeypatch):
    synth = {"features": {"synthetic_score": 0.92, "ambient_score": 0.18}}
    watery = {"features": {"synthetic_score": 0.12, "ambient_score": 0.90}}

    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    assert generator.learned_synthetic_material_factor(synth) == 1.0
    assert generator.learned_synthetic_material_factor(watery) == 1.0

    requested = dict(neutral)
    requested["synthetic_material_weight"] = 1.24
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    assert generator.learned_synthetic_material_factor(synth) < 1.0
    assert generator.learned_synthetic_material_factor(watery) > 1.0


def test_learned_synth_candidate_focus_is_library_relative_and_feedback_driven(monkeypatch):
    pool = [
        {"object_id": str(index), "features": {"synthetic_score": index / 9.0}}
        for index in range(10)
    ]
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    assert generator.learned_synthetic_candidate_pool(pool) is pool

    requested = dict(neutral)
    requested["synthetic_material_weight"] = 1.24
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    monkeypatch.setattr(generator.random, "random", lambda: 0.0)
    focused = generator.learned_synthetic_candidate_pool(pool)
    assert 3 <= len(focused) < len(pool)
    assert min(item["features"]["synthetic_score"] for item in focused) >= 5 / 9


def test_human_natural_source_label_overrides_acoustic_synth_estimate(monkeypatch):
    obj = {
        "recording": "water.wav",
        "features": {"synthetic_score": 0.91},
    }
    monkeypatch.setattr(generator, "LIBRARY_SOURCE_LABELS", {})
    assert generator.effective_synthetic_score(obj) == 0.91
    monkeypatch.setattr(generator, "LIBRARY_SOURCE_LABELS", {"water.wav": "natural"})
    assert generator.source_origin(obj) == "natural"
    assert generator.effective_synthetic_score(obj) <= 0.18


def test_human_hybrid_source_label_stays_between_natural_and_synthetic(monkeypatch):
    obj = {"recording": "hybrid.wav", "features": {"synthetic_score": 0.90}}
    monkeypatch.setattr(generator, "LIBRARY_SOURCE_LABELS", {"hybrid.wav": "hybrid"})
    score = generator.effective_synthetic_score(obj)
    assert generator.source_origin(obj) == "hybrid"
    assert 0.50 < score < 0.90

    monkeypatch.setattr(
        generator, "LIBRARY_SOURCE_LABELS", {"hybrid.wav": "instrument_hybrid"}
    )
    instrument_score = generator.effective_synthetic_score(obj)
    assert generator.source_origin(obj) == "instrument_hybrid"
    assert instrument_score == score


def test_embedding_origin_model_generalises_confirmed_synth_examples():
    unit = lambda values: np.asarray(values, dtype=np.float64) / np.linalg.norm(values)
    embeddings = {
        "s1": unit([1.0, 0.0]),
        "s2": unit([0.96, 0.10]),
        "n1": unit([0.0, 1.0]),
        "n2": unit([0.10, 0.96]),
        "near_synth": unit([0.99, 0.04]),
        "near_natural": unit([0.04, 0.99]),
    }
    assist = type("Assist", (), {"active": True, "embeddings": embeddings})()
    objects = [
        {"recording": "synth_a.wav", "object_id": "s1"},
        {"recording": "synth_b.wav", "object_id": "s2"},
        {"recording": "natural_a.wav", "object_id": "n1"},
        {"recording": "natural_b.wav", "object_id": "n2"},
        {"recording": "unknown_a.wav", "object_id": "near_synth"},
        {"recording": "unknown_b.wav", "object_id": "near_natural"},
    ]
    affinities, snapshot = generator.build_embedding_origin_model(
        objects,
        assist,
        {
            "synth_a.wav": "synthetic",
            "synth_b.wav": "synthetic",
            "natural_a.wav": "natural",
            "natural_b.wav": "natural",
        },
    )
    assert snapshot["active"] is True
    assert snapshot["synthetic_anchor_objects"] == 2
    assert affinities["near_synth"] > 0.67
    assert affinities["near_natural"] == 0.50


def test_instrument_and_foreground_feedback_have_audible_bounded_controls(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights.update({
        "synthetic_material_weight": 1.08,
        "instrument_material_weight": 1.08,
        "foreground_presence_weight": 1.08,
    })
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    monkeypatch.setattr(
        generator,
        "LIBRARY_SOURCE_LABELS",
        {"instrument.wav": "instrument_hybrid", "natural.wav": "natural"},
    )
    instrument = {"recording": "instrument.wav", "features": {}}
    natural = {"recording": "natural.wav", "features": {}}
    assert generator.learned_instrument_material_factor(instrument) < 1.0
    assert generator.learned_instrument_material_factor(natural) > 1.0
    snapshot = generator.composition_feedback_audio_snapshot()
    assert 1.0 < snapshot["instrument_presence_gain"] <= 1.55
    assert 1.0 < snapshot["foreground_presence_gain"] <= 1.45

    monkeypatch.setattr(
        generator,
        "LIBRARY_SOURCE_LABELS",
        {
            "instrument.wav": "instrument_hybrid",
            "synth.wav": "synthetic",
            "other.wav": "unknown",
        },
    )
    usage = {"other.wav": 12}
    assert generator.learned_origin_mix_factor(
        {"recording": "instrument.wav"}, usage
    ) < 1.0
    assert generator.learned_origin_mix_factor(
        {"recording": "synth.wav"}, usage
    ) < 1.0


def test_d3_and_d5_source_phrase_layers_are_distinct(monkeypatch):
    monkeypatch.setattr(generator, "OUTPUT_DURATION", 1)
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights["synthetic_material_weight"] = 1.30
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    rng = np.random.default_rng(13)
    source = rng.normal(0.0, 0.08, (generator.TARGET_SR, 2)).astype(np.float32)
    d3 = generator.source_musical_phrase_layer(source, 3, pulse_bpm=96.0)
    d5 = generator.source_musical_phrase_layer(source, 5, pulse_bpm=126.0)
    assert np.any(np.abs(d3) > 0)
    assert np.any(np.abs(d5) > 0)
    assert not np.allclose(d3, d5)


def test_phrase_synth_voices_enter_and_leave(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 101)
    for level in (1, 3, 5):
        envelope = generator.phrase_window_envelope(
            generator.TARGET_SR * 20, level, voice_index=2
        )
        assert np.max(envelope) > 0.9
        assert np.count_nonzero(envelope == 0.0) > generator.TARGET_SR


def test_learned_bed_form_has_finite_phrases_and_completed_edges(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 101)
    monkeypatch.setattr(generator, "ARTIST_STYLE", generator.ArtistStyleAssist.disabled())
    length = generator.TARGET_SR * 60
    for level in (1, 3, 5):
        first = generator.learned_bed_phrase_envelope(length, level, seed=22)
        second = generator.learned_bed_phrase_envelope(length, level, seed=22)
        assert np.array_equal(first, second)
        assert first[0] == 0.0
        assert first[-1] == 0.0
        assert np.max(first) > 0.70
        # A background phrase must actually leave before another one returns;
        # it cannot become one whole-render loop or periodic LFO.
        assert np.count_nonzero(first < 1e-5) > int(length * 0.03)


def test_source_derived_bed_is_silent_without_source_and_depends_on_source(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 101)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    silent = np.zeros((generator.TARGET_SR * 2, 2), dtype=np.float32)
    assert np.array_equal(generator.source_derived_bed(silent, 3), silent)

    time = np.arange(generator.TARGET_SR * 2, dtype=np.float32) / generator.TARGET_SR
    source_a = np.stack([
        0.1 * np.sin(2 * np.pi * 180.0 * time),
        0.1 * np.sin(2 * np.pi * 270.0 * time),
    ], axis=1).astype(np.float32)
    source_b = np.stack([
        0.1 * np.sin(2 * np.pi * 620.0 * time),
        0.1 * np.sin(2 * np.pi * 930.0 * time),
    ], axis=1).astype(np.float32)
    bed_a = generator.source_derived_bed(source_a, 3, seed=22) - source_a
    bed_b = generator.source_derived_bed(source_b, 3, seed=22) - source_b
    assert np.max(np.abs(bed_a)) > 0.0
    assert np.max(np.abs(bed_b)) > 0.0
    assert not np.allclose(bed_a, bed_b)


def test_source_drones_and_pads_require_and_follow_source_audio(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 303)
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights.update({
        "synthetic_material_weight": 1.30,
        "material_development_weight": 1.20,
    })
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    silent = np.zeros((generator.TARGET_SR * 2, 2), dtype=np.float32)
    assert np.array_equal(generator.source_drone_bloom_layer(silent, 5), silent)
    assert np.array_equal(generator.source_pad_bloom_layer(silent, 5), silent)

    rng = np.random.default_rng(19)
    source_a = rng.normal(0.0, 0.07, silent.shape).astype(np.float32)
    source_b = np.roll(source_a, generator.TARGET_SR // 3, axis=0)
    drone_a = generator.source_drone_bloom_layer(source_a, 5, seed=91)
    drone_b = generator.source_drone_bloom_layer(source_b, 5, seed=91)
    pad = generator.source_pad_bloom_layer(source_a, 5, seed=91)
    assert np.max(np.abs(drone_a)) > 0.0
    assert np.max(np.abs(pad)) > 0.0
    assert not np.allclose(drone_a, drone_b)


def test_source_tonal_candidates_follow_the_source_and_active_scale(monkeypatch):
    monkeypatch.setitem(generator.HARMONY_STATE, "root", 0)
    monkeypatch.setitem(generator.HARMONY_STATE, "scale", "minor")
    monkeypatch.setitem(generator.HARMONY_STATE, "confidence", 0.8)
    time = np.arange(generator.TARGET_SR * 3, dtype=np.float32) / generator.TARGET_SR
    first_mono = (
        0.14 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.06 * np.sin(2.0 * np.pi * 330.0 * time)
    ).astype(np.float32)
    second_mono = (
        0.14 * np.sin(2.0 * np.pi * 510.0 * time)
        + 0.06 * np.sin(2.0 * np.pi * 765.0 * time)
    ).astype(np.float32)
    first = generator.source_tonal_candidates(np.stack((first_mono, first_mono), axis=1), count=4)
    second = generator.source_tonal_candidates(np.stack((second_mono, second_mono), axis=1), count=4)
    assert first
    assert second
    assert first != second
    for frequency in first + second:
        midi = int(round(69.0 + 12.0 * np.log2(frequency / 440.0)))
        assert midi % 12 in generator.active_pitch_classes()


def test_tonal_bloom_is_optional_source_anchored_and_phrase_bounded(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 303)
    monkeypatch.setattr(generator, "ARTIST_STYLE", generator.ArtistStyleAssist.disabled())
    silent = np.zeros((generator.TARGET_SR * 12, 2), dtype=np.float32)
    assert np.array_equal(generator.learned_tonal_bloom_layer(silent, 5), silent)

    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    time = np.arange(len(silent), dtype=np.float32) / generator.TARGET_SR
    first_mono = (
        0.12 * np.sin(2.0 * np.pi * 173.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 317.0 * time)
    ).astype(np.float32)
    first_source = np.stack((first_mono, np.roll(first_mono, 37)), axis=1)
    assert np.array_equal(
        generator.learned_tonal_bloom_layer(first_source, 5, seed=91), silent
    )

    preferred = dict(neutral)
    preferred["synthetic_material_weight"] = 1.30
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", preferred)
    second_mono = (
        0.12 * np.sin(2.0 * np.pi * 431.0 * time)
        + 0.05 * np.sin(2.0 * np.pi * 683.0 * time)
    ).astype(np.float32)
    second_source = np.stack((second_mono, np.roll(second_mono, 53)), axis=1)
    first = generator.learned_tonal_bloom_layer(first_source, 5, seed=91)
    second = generator.learned_tonal_bloom_layer(second_source, 5, seed=91)
    assert np.max(np.abs(first)) > 0.0
    assert np.max(np.abs(first)) <= 0.140001
    assert np.allclose(first[0], 0.0)
    assert np.allclose(first[-1], 0.0)
    assert np.count_nonzero(np.max(np.abs(first), axis=1) < 1e-7) > int(len(first) * 0.03)
    assert not np.allclose(first, second)


def test_continuity_guard_closes_event_edges_for_every_dream_level(monkeypatch):
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    source = np.ones(generator.TARGET_SR * 2, dtype=np.float32)
    for level in (1, 3, 5):
        guarded = generator.continuity_edge_guard(source.copy(), "texture", level)
        assert guarded[0] == 0.0
        assert guarded[-1] == 0.0
        assert guarded[1] < guarded[generator.TARGET_SR]
        assert guarded[-2] < guarded[-generator.TARGET_SR]


def test_spectral_bloom_adds_no_fixed_tone_to_silence(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 101)
    silent = np.zeros((generator.TARGET_SR, 2), dtype=np.float32)
    assert np.array_equal(generator.central_spectral_bloom(silent.copy(), 3), silent)
    assert np.array_equal(generator.central_spectral_bloom(silent.copy(), 5), silent)


def test_spectral_bloom_is_source_derived_and_timing_varies_by_render(monkeypatch):
    sample_rate = generator.TARGET_SR
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    source_a = np.stack([
        0.1 * np.sin(2 * np.pi * 700.0 * time),
        0.1 * np.sin(2 * np.pi * 920.0 * time),
    ], axis=1).astype(np.float32)
    source_b = np.stack([
        0.1 * np.sin(2 * np.pi * 1180.0 * time),
        0.1 * np.sin(2 * np.pi * 1430.0 * time),
    ], axis=1).astype(np.float32)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    monkeypatch.setattr(generator, "RENDER_SEED", 101)
    bloomed_a = generator.central_spectral_bloom(source_a.copy(), 3)
    bloomed_b = generator.central_spectral_bloom(source_b.copy(), 3)
    assert not np.array_equal(bloomed_a - source_a, bloomed_b - source_b)

    timing_a = generator.spectral_bloom_timing(180.0, 5, seed=101)
    timing_b = generator.spectral_bloom_timing(180.0, 5, seed=202)
    assert timing_a != timing_b
    assert 0.0 < timing_a[0] < timing_a[1] < timing_a[2] < 180.0
    assert 0.0 < timing_b[0] < timing_b[1] < timing_b[2] < 180.0


def test_final_mix_does_not_inject_a_fixed_high_frequency_layer(monkeypatch):
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    silent = np.zeros((generator.TARGET_SR * 40, 2), dtype=np.float32)
    for level in (1, 3, 5):
        mixed = generator.final_mix(silent.copy(), level)
        assert np.array_equal(mixed, silent)


def test_adaptive_low_balance_only_reduces_severe_bass_dominance():
    sample_rate = generator.TARGET_SR
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    bass_heavy = np.stack([
        0.40 * np.sin(2 * np.pi * 70.0 * time) + 0.04 * np.sin(2 * np.pi * 900.0 * time),
        0.38 * np.sin(2 * np.pi * 82.0 * time) + 0.04 * np.sin(2 * np.pi * 1100.0 * time),
    ], axis=1).astype(np.float32)
    balanced = np.stack([
        0.06 * np.sin(2 * np.pi * 70.0 * time) + 0.20 * np.sin(2 * np.pi * 900.0 * time),
        0.06 * np.sin(2 * np.pi * 82.0 * time) + 0.20 * np.sin(2 * np.pi * 1100.0 * time),
    ], axis=1).astype(np.float32)

    corrected = generator.adaptive_low_balance(bass_heavy, 1)
    untouched = generator.adaptive_low_balance(balanced, 1)
    assert np.sqrt(np.mean(corrected * corrected)) < np.sqrt(np.mean(bass_heavy * bass_heavy))
    assert np.allclose(untouched, balanced, atol=2e-4)


def test_form_selection_varies_by_render_and_separates_dream_levels(monkeypatch):
    names = [generator.select_form_variant(level, seed=811_337) for level in (1, 3, 5)]
    assert len(set(names)) == 3
    assert generator.select_form_variant(1, seed=811_337) != generator.select_form_variant(1, seed=812_334)
    form = generator.composed_form(3, duration=180.0, seed=811_337)
    assert [section[0] for section in form] == [
        "opening", "activation", "complexity", "memory", "resolution"
    ]
    assert form[0][1] == 0.0
    assert 176.0 <= form[-1][2] <= 180.0


def test_d1_spectral_bloom_returns_and_remains_source_derived(monkeypatch):
    monkeypatch.setattr(generator, "RENDER_SEED", 403)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    sample_rate = generator.TARGET_SR
    time = np.arange(sample_rate * 8, dtype=np.float32) / sample_rate
    source = np.stack([
        0.08 * np.sin(2 * np.pi * 620.0 * time),
        0.08 * np.sin(2 * np.pi * 910.0 * time),
    ], axis=1).astype(np.float32)
    bloomed = generator.central_spectral_bloom(source.copy(), 1)
    assert not np.array_equal(bloomed, source)
    late = bloomed[int(len(bloomed) * 0.74):] - source[int(len(source) * 0.74):]
    assert np.max(np.abs(late)) > 0.0


def test_positive_structure_controls_total_density_without_equalising_levels(monkeypatch):
    class StructureAssist:
        def target_event_count(self, dream_level, _duration):
            return {1: 84, 3: 91, 5: 97}[dream_level]

    form = [
        ("opening", 0, 32, 0.65),
        ("activation", 24, 62, 1.05),
        ("complexity", 52, 102, 1.45),
        ("memory", 90, 138, 1.05),
        ("resolution", 125, 178, 0.75),
    ]
    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", StructureAssist())
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    assert sum(generator.planned_form_items(form, 1)) == 84
    assert sum(generator.planned_form_items(form, 3)) == 91
    assert sum(generator.planned_form_items(form, 5)) == 97


def test_positive_structure_controls_exact_role_balance(monkeypatch):
    class StructureAssist:
        def target_role_distribution(self, dream_level):
            assert dream_level == 1
            return {
                "gesture": 14 / 84,
                "texture": 27 / 84,
                "resonance": 36 / 84,
                "noise": 5 / 84,
                "impact": 2 / 84,
            }

        def role_factor(self, _role, _dream_level):
            return 1.0

    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", StructureAssist())
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    targets = generator.planned_role_targets(84, 1)
    assert targets == {
        "gesture": 14,
        "texture": 27,
        "resonance": 36,
        "noise": 5,
        "impact": 2,
    }
    remaining = dict(targets)
    generated = {role: 0 for role in targets}
    for _ in range(84):
        role = generator.role_sequence_for_section(
            "resolution", 1, remaining_role_counts=remaining
        )
        generated[role] += 1
    assert generated == targets
    assert sum(remaining.values()) == 0


def test_d5_temporal_energy_is_audible_and_level_specific(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    neutral_d5 = generator.d5_temporal_profile(5)
    assert generator.d5_temporal_profile(1) == {
        "temporal_drive": 1.0,
        "stretch_scale": 1.0,
        "envelope_scale": 1.0,
        "delay_scale": 1.0,
        "ambient_scale": 1.0,
    }
    assert generator.d5_temporal_profile(3) == generator.d5_temporal_profile(1)

    active = dict(neutral)
    active["activity_weight"] = 1.32
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", active)
    active_d5 = generator.d5_temporal_profile(5)
    assert active_d5["temporal_drive"] > neutral_d5["temporal_drive"]
    assert active_d5["stretch_scale"] < neutral_d5["stretch_scale"]
    assert active_d5["envelope_scale"] < neutral_d5["envelope_scale"]
    assert active_d5["ambient_scale"] < neutral_d5["ambient_scale"]


def test_explicit_activity_feedback_changes_d1_and_d3_timing(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    assert generator.d5_temporal_profile(1)["temporal_drive"] == 1.0
    assert generator.d5_temporal_profile(3)["temporal_drive"] == 1.0

    active = dict(neutral)
    active["activity_weight"] = 1.16
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", active)
    for level in (1, 3):
        profile = generator.d5_temporal_profile(level)
        assert profile["temporal_drive"] > 1.0
        assert profile["stretch_scale"] < 1.0
        assert profile["delay_scale"] < 1.0


def test_structured_granulation_is_bounded_and_feedback_controlled(monkeypatch):
    fragment = np.linspace(-0.8, 0.8, 48_000, dtype=np.float32)
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    unchanged = generator.structured_granulation(fragment.copy(), "texture", 5)
    assert np.array_equal(unchanged, fragment)

    requested = dict(neutral)
    requested["structured_granulation_weight"] = 1.10
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    developed = generator.structured_granulation(fragment.copy(), "texture", 5)
    assert len(developed) == len(fragment)
    assert not np.array_equal(developed, fragment)
    assert np.max(np.abs(developed)) <= np.max(np.abs(fragment)) + 1e-6

    stronger = dict(neutral)
    stronger["structured_granulation_weight"] = 1.30
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", stronger)
    pronounced = generator.structured_granulation(fragment.copy(), "texture", 5)
    assert np.mean(np.abs(pronounced - fragment)) > np.mean(np.abs(developed - fragment))

    resonance_d5 = generator.structured_granulation(fragment.copy(), "resonance", 5)
    resonance_d3 = generator.structured_granulation(fragment.copy(), "resonance", 3)
    assert not np.array_equal(resonance_d5, fragment)
    assert not np.array_equal(resonance_d3, fragment)


def test_crossfaded_rotation_removes_the_hard_loop_seam():
    source = np.linspace(-0.9, 0.9, 48_000, dtype=np.float32)
    shift = 12_000
    hard = np.roll(source, shift)
    soft = generator.crossfaded_circular_shift(source, shift)
    assert len(soft) == len(source)
    hard_jump = float(np.max(np.abs(np.diff(hard))))
    soft_jump = float(np.max(np.abs(np.diff(soft))))
    assert soft_jump < hard_jump * 0.10


def test_declick_repairs_an_isolated_step_but_preserves_bright_oscillation():
    smooth = np.sin(np.linspace(0.0, 35.0, 24_000)).astype(np.float32) * 0.2
    stepped = smooth.copy()
    stepped[12_000:] += 0.5
    repaired = generator.repair_isolated_discontinuities(stepped)
    assert len(repaired) == len(stepped)
    assert np.max(np.abs(np.diff(repaired))) < np.max(np.abs(np.diff(stepped))) * 0.35
    assert np.max(repaired) <= np.max(stepped) + 1e-7
    assert np.min(repaired) >= np.min(stepped) - 1e-7

    bright = np.sin(np.linspace(0.0, 8_000.0, 24_000)).astype(np.float32) * 0.35
    untouched = generator.repair_isolated_discontinuities(bright)
    assert np.allclose(untouched, bright)


def test_delivery_ceiling_prevents_post_repair_full_scale_samples():
    output = np.array([[0.25, -0.25], [1.04, -0.96], [0.40, -0.40]], dtype=np.float32)
    limited = generator.enforce_delivery_ceiling(output)
    assert limited.shape == output.shape
    assert np.max(np.abs(limited)) <= 0.880001
    assert np.allclose(generator.enforce_delivery_ceiling(output * 0.5), output * 0.5)


def test_audition_loudness_floor_is_ordered_and_silent_safe():
    rng = np.random.default_rng(918)
    source = rng.normal(0.0, 0.012, (generator.TARGET_SR, 2)).astype(np.float32)
    rms = {}
    for level in (1, 3, 5):
        lifted = generator.audition_loudness_floor(source, level)
        rms[level] = float(np.sqrt(np.mean(lifted * lifted)))
        assert np.max(np.abs(lifted)) <= 0.88 + 1e-6
    assert rms[1] < rms[3] < rms[5]
    assert np.count_nonzero(generator.audition_loudness_floor(np.zeros_like(source), 5)) == 0


def test_density_glue_makes_d5_fullest_without_changing_shape(monkeypatch):
    learned = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    learned["richness_weight"] = 1.12
    learned["activity_weight"] = 1.14
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", learned)
    t = np.linspace(0.0, 40.0, 48_000, dtype=np.float32)
    source = (0.11 * np.sin(t) + 0.015 * np.sin(t * 9.0))[:, None]
    source = np.repeat(source, 2, axis=1)
    outputs = {level: generator.parallel_density_glue(source, level) for level in (1, 3, 5)}
    assert all(value.shape == source.shape for value in outputs.values())
    rms = {level: float(np.sqrt(np.mean(value * value))) for level, value in outputs.items()}
    assert rms[5] > rms[3] > rms[1] > float(np.sqrt(np.mean(source * source)))


def test_learned_evolution_changes_all_levels_without_losing_length_or_peak(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    source = np.sin(np.linspace(0.0, 120.0, 48_000)).astype(np.float32) * 0.7
    assert np.array_equal(generator.learned_material_evolution(source, "texture", 1), source)

    learned = dict(neutral)
    learned["material_development_weight"] = 1.16
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", learned)
    for level in (1, 3, 5):
        evolved = generator.learned_material_evolution(source, "texture", level)
        assert len(evolved) == len(source)
        assert not np.array_equal(evolved, source)
        assert np.max(np.abs(evolved)) <= np.max(np.abs(source)) * 1.081


def test_strong_evolution_is_sparse_and_requires_learned_form_context(monkeypatch):
    class HighContextStyle:
        active = True

        @staticmethod
        def trajectory_curve(_name, positions, _dream_level, _seed):
            return np.full_like(np.asarray(positions, dtype=np.float64), 0.90)

    class LowContextStyle(HighContextStyle):
        @staticmethod
        def trajectory_curve(_name, positions, _dream_level, _seed):
            return np.full_like(np.asarray(positions, dtype=np.float64), 0.10)

    monkeypatch.setattr(generator, "ARTIST_STYLE", HighContextStyle())
    # D5 can develop every second structural event at most, never every event.
    first, _ = generator.musical_evolution_moment(0.5, 5, 0, "complexity", 17)
    second, _ = generator.musical_evolution_moment(0.5, 5, 1, "complexity", 17)
    assert first is False
    assert second is True

    # A cadence slot is not enough on its own: the learned trajectory must also
    # indicate meaningful growth, articulation or culmination at that point.
    monkeypatch.setattr(generator, "ARTIST_STYLE", LowContextStyle())
    quiet_slot, _ = generator.musical_evolution_moment(0.5, 5, 1, "complexity", 17)
    assert quiet_slot is False


def test_material_plans_are_not_nested_between_d_levels(monkeypatch):
    objects = [
        {
            "recording": f"sample{index}.wav",
            "recording_id": f"rec-{index}",
            "object_id": f"obj-{index}",
        }
        for index in range(45)
    ]
    monkeypatch.setattr(generator, "profile_distance", lambda _obj, _profile: 1.0)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    plans = {}
    for level in (1, 3, 5):
        random.seed(9042026)
        plans[level] = generator.build_material_plan(objects, {}, level, {"samples": {}})
    assert not plans[1].issubset(plans[3])
    assert not plans[3].issubset(plans[5])
    assert len(plans[1] & plans[3]) < len(plans[1])
    assert len(plans[3] & plans[5]) < len(plans[3])


def test_d5_aesthetic_bridge_preserves_breathing_room(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights.update({"activity_weight": 1.32, "musicality_weight": 1.03})
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    profile = generator.d5_temporal_profile(5)
    assert profile["temporal_drive"] <= 1.32
    assert profile["stretch_scale"] >= 0.90
    assert profile["envelope_scale"] >= 0.92
    assert profile["ambient_scale"] >= 0.90

    generator.CURRENT_FORM_VARIANT = "aesthetic_bridge"
    variant = generator.D5_FORM_VARIANTS[generator.CURRENT_FORM_VARIANT]
    assert variant["complexity"] > variant["opening"]
    assert variant["resolution"] < variant["complexity"]


def test_d5_internal_motion_preserves_length_and_does_not_touch_d3(monkeypatch):
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    fragment = np.ones(48_000, dtype=np.float32)
    unchanged = generator.d5_internal_motion(fragment.copy(), "gesture", 3)
    developed = generator.d5_internal_motion(fragment.copy(), "gesture", 5)
    assert np.array_equal(unchanged, fragment)
    assert len(developed) == len(fragment)
    assert not np.array_equal(developed, fragment)


def test_d5_development_feedback_creates_multi_stage_phrase_and_global_arcs(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    assert np.allclose(generator.d5_development_curve(1000, "complexity"), 1.0)
    assert np.allclose(generator.d5_global_evolution_curve(1000), 1.0)

    learned = dict(neutral)
    learned.update({
        "material_development_weight": 1.16,
        "synthetic_material_weight": 1.25,
    })
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", learned)
    phrase = generator.d5_development_curve(1000, "complexity")
    global_curve = generator.d5_global_evolution_curve(1000)
    assert float(np.ptp(phrase)) > 0.20
    assert float(np.ptp(global_curve)) > 0.25
    assert int(np.argmax(global_curve)) > 400


def test_d5_reference_grid_and_lane_continuity_are_soft_not_hard():
    base = 11.37
    gridded = generator.d5_soft_grid_start(base, 0.0, "gesture", 5, 126.0)
    beat_subdivision = (60.0 / 126.0) / 2.0
    nearest = round(base / beat_subdivision) * beat_subdivision
    assert abs(gridded - nearest) < abs(base - nearest)
    assert generator.d5_soft_grid_start(base, 0.0, "gesture", 3, 126.0) == base

    adjusted = generator.d5_continuity_start(8.0, 2.0, "gesture", 3.0, 5, 126.0)
    assert adjusted < 8.0
    assert generator.d5_continuity_start(8.0, 2.0, "gesture", 3.0, 3, 126.0) == 8.0


def test_edge_guard_prevents_scissor_cuts_at_every_level(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights["transition_smoothness_weight"] = 1.20
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    fragment = np.ones(96_000, dtype=np.float32)
    guarded_d3 = generator.continuity_edge_guard(fragment.copy(), "gesture", 3)
    guarded = generator.continuity_edge_guard(fragment.copy(), "gesture", 5)
    assert guarded_d3[0] == 0.0
    assert guarded_d3[-1] == 0.0
    assert 0.0 < guarded_d3[2_000] < 1.0
    assert 0.0 < guarded_d3[-2_000] < 1.0
    assert guarded[0] == 0.0
    assert guarded[-1] == 0.0
    assert 0.0 < guarded[2_000] < 1.0
    assert 0.0 < guarded[-2_000] < 1.0


def test_d5_activity_is_bounded_to_avoid_layer_confetti(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights["activity_weight"] = 1.8
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    assert generator.dream_activity_multiplier(5) <= 1.23


def test_moving_pan_stereo_preserves_signal_and_reaches_wider_d5_space():
    mono = np.ones(48_000, dtype=np.float32) * 0.2
    narrow = generator.moving_pan_stereo(mono, -0.46, 0.46)
    wide = generator.moving_pan_stereo(mono, -0.96, 0.96)

    assert narrow.shape == wide.shape == (48_000, 2)
    assert np.isfinite(narrow).all() and np.isfinite(wide).all()
    assert np.allclose(np.sum(narrow * narrow, axis=1), mono * mono, atol=2e-6)
    assert np.allclose(np.sum(wide * wide, axis=1), mono * mono, atol=2e-6)
    assert abs(float(wide[0, 0] - wide[0, 1])) > abs(float(narrow[0, 0] - narrow[0, 1]))


def test_electroacoustic_movement_is_source_derived_and_silence_safe(monkeypatch):
    class NeutralPreference:
        embeddings = {}

        @staticmethod
        def object_factor(_object_id, _recording, _dream_level):
            return 1.0

    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", NeutralPreference())
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    event = {
        "object": {
            "object_id": "source-object",
            "recording": "source.wav",
            "features": {
                "musicality": 0.8,
                "foreground_probability": 0.7,
                "gesture_strength": 0.75,
            },
        },
        "role": "gesture",
        "start": 0.1,
        "duration": 1.8,
    }
    silence = np.zeros((3 * generator.TARGET_SR, 2), dtype=np.float32)
    assert np.array_equal(
        generator.electroacoustic_movement_layer(silence, [event], 5, seed=41),
        silence,
    )

    first_mono = _sine(generator.TARGET_SR, 3.0, frequency=173.0, amplitude=0.18)
    second_mono = _sine(generator.TARGET_SR, 3.0, frequency=317.0, amplitude=0.18)
    first = np.stack([first_mono, first_mono], axis=1)
    second = np.stack([second_mono, second_mono], axis=1)
    first_layer = generator.electroacoustic_movement_layer(first, [event], 5, seed=41)
    second_layer = generator.electroacoustic_movement_layer(second, [event], 5, seed=41)

    assert first_layer.shape == second_layer.shape == first.shape
    assert np.isfinite(first_layer).all() and np.isfinite(second_layer).all()
    assert np.max(np.abs(first_layer)) <= 0.280001
    assert np.max(np.abs(first_layer)) > 0.0
    assert not np.allclose(first_layer, second_layer)


def test_movement_candidates_follow_learned_preference_affinity(monkeypatch):
    class LearnedPreference:
        embeddings = {"favoured": np.ones(4), "other": np.ones(4)}

        @staticmethod
        def object_factor(object_id, _recording, _dream_level):
            return 0.76 if object_id == "favoured" else 1.28

    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", LearnedPreference())

    def event(object_id):
        return {
            "object": {
                "object_id": object_id,
                "recording": f"{object_id}.wav",
                "features": {"musicality": 0.75, "gesture_strength": 0.70},
            },
            "role": "gesture",
            "start": 0.0,
            "duration": 2.0,
        }

    assert generator.movement_candidate_score(event("favoured"), 5) > generator.movement_candidate_score(
        event("other"), 5
    )


def test_fragile_air_remains_available_but_cannot_become_a_long_bed():
    bright_musical = {
        "features": {
            "energy": 0.03,
            "brightness": 12_000.0,
            "zero_crossing_rate": 0.31,
            "musicality": 0.72,
        }
    }
    fragile = {
        "features": {
            "energy": 0.00001,
            "brightness": 12_000.0,
            "zero_crossing_rate": 0.32,
            "musicality": 0.15,
        }
    }
    assert not generator.fragile_air_object(bright_musical)
    assert generator.fragile_air_object(fragile)

    source = np.ones(12 * generator.TARGET_SR, dtype=np.float32)
    untouched = generator.bound_fragile_air_punctuation(source, bright_musical, 1)
    bounded = generator.bound_fragile_air_punctuation(source, fragile, 1)
    assert len(untouched) == len(source)
    assert len(bounded) == int(3.2 * generator.TARGET_SR)
    assert bounded[0] == 0.0 and bounded[-1] == 0.0


def test_material_plan_limits_are_balanced(monkeypatch):
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    assert generator.material_plan_limits(1) == (6, 12)
    assert generator.material_plan_limits(3) == (10, 20)
    assert generator.material_plan_limits(5) == (16, 30)


def test_library_exploration_expands_each_level_palette(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    neutral_limits = {level: generator.material_plan_limits(level) for level in (1, 3, 5)}
    wider = dict(neutral)
    wider["exploration_weight"] = 1.18
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", wider)
    for level in (1, 3, 5):
        recording_limit, object_limit = generator.material_plan_limits(level)
        assert recording_limit > neutral_limits[level][0]
        assert object_limit > neutral_limits[level][1]


def test_development_feedback_never_shrinks_any_level_palette(monkeypatch):
    class NoStructureAssist:
        def target_unique_recordings(self, _dream_level):
            return None

    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", NoStructureAssist())
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    neutral_limits = {level: generator.material_plan_limits(level) for level in (1, 3, 5)}
    developed = dict(neutral)
    developed["material_development_weight"] = 1.20
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", developed)
    for level in (1, 3, 5):
        assert generator.material_plan_limits(level)[0] >= neutral_limits[level][0]
        assert generator.material_plan_limits(level)[1] >= neutral_limits[level][1]


def test_accepted_structure_is_a_palette_floor_for_all_levels(monkeypatch):
    class StructureAssist:
        def target_unique_recordings(self, dream_level):
            return {1: 8, 3: 12, 5: 18}[dream_level]

    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", StructureAssist())
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    assert generator.material_plan_limits(1) == (8, 16)
    assert generator.material_plan_limits(3) == (12, 24)
    assert generator.material_plan_limits(5) == (18, 36)


def test_composition_duration_favours_long_sustained_material(monkeypatch):
    class DurationAssist:
        def target_average_event_duration(self, dream_level):
            return {1: 18.4, 3: 17.9, 5: 12.1}[dream_level]

    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", DurationAssist())
    short = {"duration": 0.9}
    long = {"duration": 5.0}
    for level in (1, 3, 5):
        assert generator.composition_duration_selection_factor(short, "texture", level) > 1.0
        assert generator.composition_duration_selection_factor(short, "resonance", level) > 1.0
        assert generator.composition_duration_selection_factor(long, "resonance", level) <= 1.35
        assert generator.composition_duration_selection_factor(short, "gesture", level) == 1.0


def test_composition_sustain_bloom_uses_each_level_learned_duration(monkeypatch):
    class DurationAssist:
        def target_average_event_duration(self, dream_level):
            return {1: 6.0, 3: 5.0, 5: 4.0}[dream_level]

    monkeypatch.setattr(generator, "COMPOSITION_PREFERENCE", DurationAssist())
    source = np.sin(np.linspace(0.0, 40.0, generator.TARGET_SR)).astype(np.float32)
    d1 = generator.composition_sustain_bloom(source, "texture", 1)
    d5 = generator.composition_sustain_bloom(source, "texture", 5)
    gesture = generator.composition_sustain_bloom(source, "gesture", 1)
    assert len(d1) > len(d5) > len(source)
    assert np.array_equal(gesture, source)
    assert np.max(np.abs(d1)) <= np.max(np.abs(source)) + 1e-6


def test_recording_dominance_guard_preserves_available_alternatives():
    pool = [
        {"recording": recording, "object_id": f"{recording}-{index}"}
        for recording in ("dominant.wav", "other-a.wav", "other-b.wav")
        for index in range(3)
    ]
    guarded = generator.diversify_overused_recordings(
        pool,
        {"dominant.wav": 30, "other-a.wav": 3, "other-b.wav": 2},
        3,
        palette_size=9,
    )
    assert guarded
    assert all(obj["recording"] != "dominant.wav" for obj in guarded)


def test_recording_dominance_guard_never_removes_the_only_viable_pool():
    pool = [
        {"recording": "dominant.wav", "object_id": f"object-{index}"}
        for index in range(3)
    ]
    assert generator.diversify_overused_recordings(
        pool,
        {"dominant.wav": 30},
        1,
        palette_size=6,
    ) == pool


def test_critic_scores_have_dynamic_range_without_hard_ceiling(tmp_path):
    sample_rate = 48_000
    smooth = _sine(sample_rate, 4.0, amplitude=0.15)
    discontinuous = smooth.copy()
    discontinuous[sample_rate : 2 * sample_rate] *= 0.02
    discontinuous[2 * sample_rate : 3 * sample_rate] += np.random.default_rng(7).normal(
        0.0, 0.18, sample_rate
    )
    smooth_path = tmp_path / "smooth.wav"
    rough_path = tmp_path / "rough.wav"
    sf.write(smooth_path, smooth, sample_rate)
    sf.write(rough_path, discontinuous, sample_rate)
    smooth_scores = critic_v2.analyse_audio(str(smooth_path))["internal_scores"]
    rough_scores = critic_v2.analyse_audio(str(rough_path))["internal_scores"]
    assert all(3.0 <= value <= 97.0 for value in smooth_scores.values())
    assert all(3.0 <= value <= 97.0 for value in rough_scores.values())
    assert smooth_scores != rough_scores
    assert smooth_scores["coherence"] > rough_scores["coherence"]
    assert smooth_scores["coherence"] < 90.0


def test_critic_detects_repeated_render_structure(tmp_path):
    def report(audio, timestamp, role_counts, section_counts):
        return {
            "audio_file": audio,
            "timestamp": timestamp,
            "dream_level": 5,
            "role_counts": role_counts,
            "sample_usage_details": {
                "sample": {"section_counts": section_counts},
            },
        }

    prior = report(
        "output/prior.wav",
        "2026-08-21T10:00:00Z",
        {"gesture": 30, "texture": 35, "resonance": 30, "noise": 4, "impact": 1},
        {"opening": 12, "activation": 20, "complexity": 36, "memory": 20, "resolution": 12},
    )
    repeated = report(
        "output/repeated.wav",
        "2026-08-21T10:05:00Z",
        dict(prior["role_counts"]),
        dict(prior["sample_usage_details"]["sample"]["section_counts"]),
    )
    prior_path = tmp_path / "prior_render_report.json"
    repeated_path = tmp_path / "repeated_render_report.json"
    prior_path.write_text(json.dumps(prior))
    repeated_path.write_text(json.dumps(repeated))
    novelty = critic_v2.historical_structure_novelty(str(repeated_path))
    assert novelty["prior_renders"] == 1
    assert novelty["score"] == 0.0

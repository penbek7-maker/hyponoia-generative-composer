import json

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


def test_arpeggio_feedback_creates_bounded_level_specific_audio(monkeypatch):
    neutral = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", neutral)
    silent = generator.synth_arpeggio_layer(3, duration=4.0)
    assert silent.shape == (generator.TARGET_SR * 4, 2)
    assert np.count_nonzero(silent) == 0

    requested = dict(neutral)
    requested["arpeggio_weight"] = 1.10
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", requested)
    d3 = generator.synth_arpeggio_layer(3, duration=40.0)
    d5 = generator.synth_arpeggio_layer(5, pulse_bpm=126.0, duration=40.0)
    assert np.isfinite(d3).all() and np.isfinite(d5).all()
    assert np.max(np.abs(d3)) < 0.08
    assert np.max(np.abs(d5)) < 0.08
    assert np.count_nonzero(d3) > 0
    assert not np.array_equal(d3, d5)


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


def test_d5_aesthetic_bridge_preserves_breathing_room(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights.update({"activity_weight": 1.32, "musicality_weight": 1.03})
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    profile = generator.d5_temporal_profile(5)
    assert profile["temporal_drive"] <= 1.30
    assert profile["stretch_scale"] >= 0.94
    assert profile["envelope_scale"] >= 0.96
    assert profile["ambient_scale"] >= 0.94

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


def test_d5_edge_guard_prevents_scissor_cuts_without_touching_d3(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights["transition_smoothness_weight"] = 1.20
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    fragment = np.ones(96_000, dtype=np.float32)
    unchanged = generator.continuity_edge_guard(fragment.copy(), "gesture", 3)
    guarded = generator.continuity_edge_guard(fragment.copy(), "gesture", 5)
    assert np.array_equal(unchanged, fragment)
    assert guarded[0] == 0.0
    assert guarded[-1] == 0.0
    assert 0.0 < guarded[2_000] < 1.0
    assert 0.0 < guarded[-2_000] < 1.0


def test_d5_activity_is_bounded_to_avoid_layer_confetti(monkeypatch):
    weights = dict(generator.DEFAULT_LEARNING_WEIGHTS)
    weights["activity_weight"] = 1.8
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", weights)
    assert generator.dream_activity_multiplier(5) <= 1.23


def test_material_plan_limits_are_balanced(monkeypatch):
    monkeypatch.setattr(generator, "LEARNING_WEIGHTS", dict(generator.DEFAULT_LEARNING_WEIGHTS))
    assert generator.material_plan_limits(1) == (6, 12)
    assert generator.material_plan_limits(3) == (10, 20)
    assert generator.material_plan_limits(5) == (16, 30)


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

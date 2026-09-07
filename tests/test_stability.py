import json
import subprocess
import sys

import numpy as np

from hyponoia_stability import (
    apply_control_deltas,
    atomic_write_json,
    deterministic_group,
    migrate_sample_profile,
    parse_feedback_comment,
    stable_object_id,
    stable_recording_id,
)


def test_content_ids_are_stable_and_content_sensitive():
    audio = np.linspace(-0.5, 0.5, 4_800, dtype=np.float32)
    assert stable_recording_id(audio, 48_000) == stable_recording_id(audio.copy(), 48_000)
    assert stable_object_id(audio, 48_000) != stable_object_id(audio + 0.01, 48_000)


def test_palette_group_is_stable_across_processes():
    local = deterministic_group("field_recording_A.wav")
    command = (
        "from hyponoia_stability import deterministic_group; "
        "print(deterministic_group('field_recording_A.wav'))"
    )
    remote = int(subprocess.check_output([sys.executable, "-c", command], text=True).strip())
    assert local == remote


def test_phrase_to_action_changes_real_controls():
    interpretation = parse_feedback_comment(
        "More library objects, less repetition, more synthesizers, and more energetic"
    )
    assert interpretation["status"] == "interpreted"
    weights = {}
    updates = apply_control_deltas(weights, interpretation["combined_control_deltas"])
    assert weights["exploration_weight"] > 1.0
    assert weights["repetition_control"] > 1.0
    assert weights["synthetic_material_weight"] > 1.0
    assert weights["activity_weight"] > 1.0
    assert {item["control"] for item in updates} >= {
        "exploration_weight",
        "repetition_control",
        "synthetic_material_weight",
        "activity_weight",
    }


def test_feedback_separates_development_from_cross_render_exploration():
    interpretation = parse_feedback_comment(
        "Fewer samples, develop the selected sounds, and use different materials"
    )
    assert interpretation["status"] == "interpreted"
    weights = {}
    apply_control_deltas(weights, interpretation["combined_control_deltas"])
    assert weights["material_development_weight"] > 1.0
    assert weights["exploration_weight"] > 1.0


def test_faster_is_interpreted_as_real_activity_control_for_d5():
    interpretation = parse_feedback_comment(
        "D5: more energetic, faster, and develop the selected sounds",
        5,
    )
    assert interpretation["status"] == "interpreted"
    assert interpretation["target_levels"] == ["D5"]
    intents = {action["intent"] for action in interpretation["actions"]}
    assert "increase_activity" in intents
    assert "increase_material_development" in intents
    assert interpretation["combined_control_deltas"]["activity_weight"] == 0.08


def test_organised_granulation_is_a_distinct_bounded_intent():
    interpretation = parse_feedback_comment(
        "D5: περισσότερη ενέργεια και περισσότερο οργανωμένο granulation",
        5,
    )
    intents = {action["intent"] for action in interpretation["actions"]}
    assert "increase_activity" in intents
    assert "increase_structured_granulation" in intents
    assert interpretation["combined_control_deltas"]["structured_granulation_weight"] == 0.10


def test_natural_greek_low_energy_and_same_materials_feedback_is_understood():
    interpretation = parse_feedback_comment(
        "Είναι πολύ υποτονικό, παραείναι αργό και ακούω τα ίδια υλικά και στα τρία",
        3,
    )
    intents = {action["intent"] for action in interpretation["actions"]}
    assert "increase_activity" in intents
    assert "increase_library_exploration" in intents


def test_mixed_greek_feedback_maps_synth_granulation_arpeggios_and_layers():
    interpretation = parse_feedback_comment(
        "Σε όλα τα επίπεδα: περισσότερο οργανωμένο granulation, περισσότερα layers, "
        "περισσότερα arpeggios και περισσότερους συνθετικούς ήχους",
        1,
    )
    intents = {action["intent"] for action in interpretation["actions"]}
    assert interpretation["target_levels"] == ["D1", "D3", "D5"]
    assert {
        "increase_structured_granulation",
        "increase_richness",
        "increase_arpeggios",
        "increase_synthetic_material",
    } <= intents


def test_greek_instruments_and_forward_mix_have_distinct_controls():
    interpretation = parse_feedback_comment(
        "D5: περισσότερα όργανα και synth, με τα μουσικά στοιχεία πιο μπροστά",
        5,
    )
    intents = {action["intent"] for action in interpretation["actions"]}
    assert {
        "increase_instrument_material",
        "increase_synthetic_material",
        "bring_musical_material_forward",
    } <= intents
    deltas = interpretation["combined_control_deltas"]
    assert deltas["instrument_material_weight"] == 0.08
    assert deltas["foreground_presence_weight"] == 0.08


def test_arpeggio_can_be_removed_from_one_selected_level():
    interpretation = parse_feedback_comment("Δεν χρειάζεται το arpeggio εδώ", 3)
    assert interpretation["target_levels"] == ["D3"]
    assert {action["intent"] for action in interpretation["actions"]} == {
        "decrease_arpeggios"
    }
    assert interpretation["combined_control_deltas"]["arpeggio_weight"] == -0.20


def test_d5_musical_rhythmic_synthetic_bloom_feedback_is_fully_interpreted():
    interpretation = parse_feedback_comment(
        "D5: more musical, a little more rhythmic, more synthetic, and greater bloom",
        5,
    )
    assert interpretation["status"] == "interpreted"
    assert interpretation["target_levels"] == ["D5"]
    intents = {action["intent"] for action in interpretation["actions"]}
    assert intents == {
        "increase_musicality",
        "increase_rhythmicity",
        "increase_synthetic_material",
        "increase_bloom",
    }
    assert set(interpretation["combined_control_deltas"]) == {
        "musicality_weight",
        "activity_weight",
        "synthetic_material_weight",
        "bloom_weight",
    }


def test_greek_explicit_global_scope_routes_to_all_levels_without_bilingual_intents():
    interpretation = parse_feedback_comment("Σε όλα τα επίπεδα: smoother", 3)
    assert interpretation["scope"] == "global"
    assert interpretation["target_levels"] == ["D1", "D3", "D5"]
    assert interpretation["status"] == "interpreted"


def test_level_mentioned_as_comparison_does_not_override_selected_level():
    interpretation = parse_feedback_comment(
        "Ήταν καλό για D1, αλλά είχε πολύ χαμηλή ενέργεια",
        3,
    )
    assert interpretation["target_levels"] == ["D3"]


def test_d5_boring_and_slow_feedback_requests_development_and_activity():
    interpretation = parse_feedback_comment(
        "D5: είναι πολύ αργό και βαρετό, θέλω περισσότερη κίνηση",
        5,
    )
    assert interpretation["target_levels"] == ["D5"]
    intents = {action["intent"] for action in interpretation["actions"]}
    assert "increase_activity" in intents
    assert "increase_material_development" in intents


def test_impressive_energetic_d5_request_maps_to_bloom_and_activity():
    interpretation = parse_feedback_comment(
        "D5: το θέλω πιο εντυπωσιακό, σούπερ δυναμικό και παθιασμένο",
        5,
    )
    intents = {action["intent"] for action in interpretation["actions"]}
    assert "increase_bloom" in intents
    assert "increase_activity" in intents


def test_preserving_granulation_does_not_request_an_increase():
    interpretation = parse_feedback_comment(
        "D5: πιο εντυπωσιακό, διατηρώντας το οργανωμένο granulation",
        5,
    )
    intents = {action["intent"] for action in interpretation["actions"]}
    assert "increase_bloom" in intents
    assert "increase_structured_granulation" not in intents


def test_v1_sample_profile_migrates_without_losing_evidence():
    memory = [{
        "recording": "source.wav",
        "recording_id": "rec_content",
        "objects": [{"id": "obj_content", "stable_id": "obj_content", "legacy_id": 3}],
    }]
    old = {
        "version": 1,
        "total_render_selections": 7,
        "samples": {
            "source.wav::3": {
                "learned_value": 0.25,
                "times_selected": 7,
                "feedback_updates": 2,
            }
        },
    }
    migrated, report = migrate_sample_profile(old, memory)
    entry = migrated["samples"]["rec_content::obj_content"]
    assert migrated["version"] == 2
    assert entry["learned_value"] == 0.25
    assert entry["times_selected"] == 7
    assert report["moved_entries"] == 1


def test_atomic_json_write(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"ok": True, "value": 2})
    assert json.loads(path.read_text()) == {"ok": True, "value": 2}

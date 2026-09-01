import copy
import json

import pytest

from composition_feedback_v1 import (
    apply_composition_feedback,
    apply_feedback_file,
    build_composition_feedback,
)
from human_feedback_v1 import DEFAULT_LEARNING_PROFILE
from hyponoia_stability import parse_feedback_comment


RATINGS = {
    "musicality": 2,
    "material_coherence": 2.5,
    "transition_smoothness": 2,
    "variety_without_disconnection": 3,
    "synth_material_presence": 2,
    "development_over_repetition": 2,
    "overall_artistic_impression": 2.5,
}


def test_greek_feedback_recognises_compositional_intents():
    result = parse_feedback_comment(
        "Περισσότερη μουσικότητα και περισσότερα ηχητικά επίπεδα. "
        "Λιγότερα μπάσα και κρυμμένα πράγματα. Πιο ομαλές μεταβάσεις.",
        1,
    )
    intents = {action["intent"] for action in result["actions"]}
    assert result["status"] == "interpreted"
    assert result["target_levels"] == ["D1"]
    assert {
        "increase_musicality",
        "increase_richness",
        "reduce_low_frequency_masking",
        "increase_layer_clarity",
        "increase_smoothness",
    } <= intents


def test_more_energy_maps_to_bounded_activity_control():
    event = build_composition_feedback(
        RATINGS,
        dream_level="D1",
        keep_as_baseline=False,
        more="συνθετικό υλικό και λίγη περισσότερη ενέργεια",
        less="να φεύγουν λιγότερο απότομα και ομαλά",
    )
    assert event["requested_control_deltas"]["activity_weight"] == pytest.approx(0.07)
    assert event["requested_control_deltas"]["synthetic_material_weight"] > 0.08
    assert event["requested_control_deltas"]["transition_smoothness_weight"] > 0.08


def test_listener_ratings_build_bounded_level_specific_event():
    event = build_composition_feedback(
        RATINGS,
        dream_level="D1",
        keep_as_baseline=False,
        more="μουσικότητα και ηχητικά επίπεδα",
        less="μπάσα και κρυμμένα πράγματα",
        comment="έτσι και έτσι, οκ για πρώτη φορά",
        render_name="Hyponoia_DeepLearning_Assist_Smoke_D1",
    )
    assert event["accepted_as_aesthetic_baseline"] is False
    assert event["ratings_0_to_100"]["musicality"] == 40
    assert event["requested_control_deltas"]["musicality_weight"] > 0
    assert event["requested_control_deltas"]["synthetic_material_weight"] > 0
    assert event["requested_control_deltas"]["low_frequency_control"] > 0
    assert event["requested_control_deltas"]["layer_clarity_weight"] > 0


def test_composition_feedback_updates_only_the_render_dream_level():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    d3_before = copy.deepcopy(profile["level_weights"]["D3"])
    d5_before = copy.deepcopy(profile["level_weights"]["D5"])
    event = build_composition_feedback(RATINGS, dream_level=1, keep_as_baseline=False)
    updated, changes = apply_composition_feedback(profile, event)
    assert changes
    assert updated["level_weights"]["D1"]["musicality_weight"] > 1.0
    assert updated["level_weights"]["D3"] == d3_before
    assert updated["level_weights"]["D5"] == d5_before
    assert updated["weights"] == profile["weights"]
    assert updated["history"][-1]["event_id"] == event["event_id"]


def test_composition_feedback_rejects_missing_or_out_of_range_rating():
    with pytest.raises(ValueError, match="Missing composition ratings"):
        build_composition_feedback({}, dream_level=1, keep_as_baseline=False)
    invalid = dict(RATINGS, musicality=0)
    with pytest.raises(ValueError, match="between 1 and 5"):
        build_composition_feedback(invalid, dream_level=1, keep_as_baseline=False)


def test_feedback_file_persists_only_the_requested_level(tmp_path):
    feedback_path = tmp_path / "d1_review.json"
    profile_path = tmp_path / "learning_profile.json"
    event_path = tmp_path / "d1_event.json"
    feedback_path.write_text(json.dumps({
        "dream_level": "D1",
        "keep_as_baseline": False,
        "render_name": "D1-A",
        "ratings": RATINGS,
        "more": "μουσικότητα και ηχητικά επίπεδα",
        "less": "μπάσα και κρυμμένα πράγματα",
        "comment": "έτσι και έτσι, οκ για πρώτη φορά",
    }), encoding="utf-8")

    profile, event, changes = apply_feedback_file(
        feedback_path,
        profile_path=profile_path,
        event_output_path=event_path,
    )

    assert changes
    assert event["dream_level"] == "D1"
    assert profile["level_weights"]["D1"]["low_frequency_control"] > 1.0
    assert profile["level_weights"]["D1"]["layer_clarity_weight"] > 1.0
    assert profile["level_weights"]["D3"] == DEFAULT_LEARNING_PROFILE["level_weights"]["D3"]
    assert profile["level_weights"]["D5"] == DEFAULT_LEARNING_PROFILE["level_weights"]["D5"]
    assert profile_path.exists()
    assert event_path.exists()

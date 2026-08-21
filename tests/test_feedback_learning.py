import copy
import pytest

from human_feedback_v1 import (
    DEFAULT_LEARNING_PROFILE,
    append_feedback_history,
    apply_comment_to_profile,
    update_sample_values,
    update_weights,
)
from hyponoia_stability import parse_feedback_comment
import generator_v3_memory_bloom_smooth as generator


def _scores(value):
    return {
        "musicality": value,
        "coherence": value,
        "richness": value,
        "transitions": value,
        "bloom_quality": value,
        "overall": value,
    }


def test_low_transition_rating_requests_more_smoothing_regardless_of_critic_residual():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    internal = _scores(30.0)
    human = _scores(90.0)
    human["transitions"] = 35.0  # Still above the Critic: old residual direction was unsafe.
    result = update_weights(profile, internal, human)
    assert profile["weights"]["transition_smoothness_weight"] > 1.0
    assert result["signed_residuals_human_minus_critic"]["transitions"] == 5.0


def test_positive_feedback_rewards_and_negative_feedback_weakens_samples():
    report = {
        "samples": {"rec::long": 4, "rec::short": 1},
        "sample_usage_details": {
            "rec::long": {
                "selection_count": 4,
                "exposure_sec": 12.0,
                "gain_sum": 2.0,
                "role_counts": {"texture": 4},
            },
            "rec::short": {
                "selection_count": 1,
                "exposure_sec": 1.0,
                "gain_sum": 0.2,
                "role_counts": {"gesture": 1},
            },
        },
    }
    profile = {"version": 2, "samples": {}, "total_render_selections": 5}
    positive = update_sample_values(profile, report, _scores(90), _scores(60))
    assert positive["rec::long"]["delta"] > positive["rec::short"]["delta"] > 0

    negative_profile = {"version": 2, "samples": {}, "total_render_selections": 5}
    negative = update_sample_values(negative_profile, report, _scores(20), _scores(60))
    assert negative["rec::long"]["delta"] < negative["rec::short"]["delta"] < 0


def test_sample_credit_report_contains_exposure_role_and_uncertainty():
    report = {
        "samples": {"rec::obj": 2},
        "sample_usage_details": {
            "rec::obj": {
                "selection_count": 2,
                "exposure_sec": 4.0,
                "gain_sum": 1.0,
                "role_counts": {"resonance": 2},
            }
        },
    }
    profile = {"version": 2, "samples": {}, "total_render_selections": 2}
    update = update_sample_values(profile, report, _scores(80), _scores(70))["rec::obj"]
    assert update["exposure_sec"] == 4.0
    assert update["roles"] == {"resonance": 2}
    assert 0 < update["credit"] <= 1
    assert 0 < update["uncertainty"] <= 1


def test_feedback_changes_learned_value_and_actual_selection_probability():
    report = {
        "samples": {"rec::liked": 4},
        "sample_usage_details": {
            "rec::liked": {
                "selection_count": 4,
                "exposure_sec": 12.0,
                "gain_sum": 2.0,
                "role_counts": {"texture": 4},
            }
        },
    }
    positive = {"version": 2, "samples": {}, "total_render_selections": 4}
    negative = {"version": 2, "samples": {}, "total_render_selections": 4}
    for _ in range(8):
        update_sample_values(positive, report, _scores(90), _scores(60))
        update_sample_values(negative, report, _scores(10), _scores(60))

    liked_value = positive["samples"]["rec::liked"]["learned_value"]
    disliked_value = negative["samples"]["rec::liked"]["learned_value"]
    assert liked_value > 0 > disliked_value

    liked_factor = generator.sample_learning_factor(
        {"recording_id": "rec", "object_id": "liked", "recording": "x", "legacy_id": 1},
        positive,
    )
    disliked_factor = generator.sample_learning_factor(
        {"recording_id": "rec", "object_id": "liked", "recording": "x", "legacy_id": 1},
        negative,
    )
    liked_probability = generator.selection_probabilities([liked_factor, 1.0])[0]
    disliked_probability = generator.selection_probabilities([disliked_factor, 1.0])[0]
    assert liked_probability > 0.55
    assert disliked_probability < 0.45


def test_level_specific_comments_do_not_leak_between_dream_levels():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    shared_before = copy.deepcopy(profile["weights"])
    d1_before = copy.deepcopy(profile["level_weights"]["D1"])
    d5_before = copy.deepcopy(profile["level_weights"]["D5"])

    interpretation = parse_feedback_comment("D3: more active and smoother", 3)
    updates = apply_comment_to_profile(profile, interpretation)

    assert interpretation["target_levels"] == ["D3"]
    assert profile["weights"] == shared_before
    assert profile["level_weights"]["D1"] == d1_before
    assert profile["level_weights"]["D5"] == d5_before
    assert profile["level_weights"]["D3"]["activity_weight"] > 1.0
    assert profile["level_weights"]["D3"]["transition_smoothness_weight"] > 1.0
    assert {update["target_level"] for update in updates} == {"D3"}


def test_explicit_global_comment_updates_all_levels_but_not_shared_ratings():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    shared_before = copy.deepcopy(profile["weights"])
    interpretation = parse_feedback_comment("Globally, use less repetition", 3)
    apply_comment_to_profile(profile, interpretation)
    assert interpretation["scope"] == "global"
    assert profile["weights"] == shared_before
    for level in ("D1", "D3", "D5"):
        assert profile["level_weights"][level]["repetition_control"] > 1.0


def test_three_distinct_level_comments_remain_isolated():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    cases = (
        ("D1: smoother", 1),
        ("D3: more active", 3),
        ("D5: less repetition", 5),
    )
    for comment, level in cases:
        apply_comment_to_profile(profile, parse_feedback_comment(comment, level))

    assert profile["level_weights"]["D1"]["transition_smoothness_weight"] > 1.0
    assert profile["level_weights"]["D1"]["activity_weight"] == 1.0
    assert profile["level_weights"]["D1"]["repetition_control"] == 1.0

    assert profile["level_weights"]["D3"]["activity_weight"] > 1.0
    assert profile["level_weights"]["D3"]["transition_smoothness_weight"] == 1.0
    assert profile["level_weights"]["D3"]["repetition_control"] == 1.0

    assert profile["level_weights"]["D5"]["repetition_control"] > 1.0
    assert profile["level_weights"]["D5"]["transition_smoothness_weight"] == 1.0
    assert profile["level_weights"]["D5"]["activity_weight"] == 1.0


def test_generator_loads_shared_plus_only_the_active_level_profile(tmp_path, monkeypatch):
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    profile["weights"]["activity_weight"] = 1.05
    profile["level_weights"]["D1"]["activity_weight"] = 0.95
    profile["level_weights"]["D3"]["activity_weight"] = 1.15
    profile["level_weights"]["D5"]["activity_weight"] = 1.30
    path = tmp_path / "learning_profile.json"
    path.write_text(__import__("json").dumps(profile))
    monkeypatch.setattr(generator, "LEARNING_FILE", str(path))

    assert generator.load_learning_weights(1)["activity_weight"] == pytest.approx(1.0)
    assert generator.load_learning_weights(3)["activity_weight"] == pytest.approx(1.2)
    assert generator.load_learning_weights(5)["activity_weight"] == pytest.approx(1.35)


def test_feedback_history_is_append_only_and_rejects_duplicate_event_ids():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    first = {"event_id": "fb_one", "human_scores": {"overall": 70}}
    second = {"event_id": "fb_two", "human_scores": {"overall": 80}}
    append_feedback_history(profile, first)
    append_feedback_history(profile, second)
    assert profile["history"] == [first, second]
    with pytest.raises(ValueError, match="Duplicate feedback event_id"):
        append_feedback_history(profile, first)

import json

import pytest

from feedback_input_v1 import build_feedback_preview
from local_llm_feedback_v1 import LocalLLMUnavailable, interpret_with_local_llm


def fake_response(result):
    def request_json(_url, payload, _timeout):
        assert payload["think"] is False
        assert payload["options"]["temperature"] == 0
        assert payload["format"]["additionalProperties"] is False
        return {"message": {"content": json.dumps(result, ensure_ascii=False)}}

    return request_json


def test_local_llm_maps_free_greek_language_to_hyponoia_owned_bounded_deltas():
    result = interpret_with_local_llm(
        "Αυτό είναι βασικά λίγο άδειο",
        "D3",
        request_json=fake_response({
            "summary_el": "Ζητά πλουσιότερη υφή και περισσότερη ανάπτυξη.",
            "intents": ["increase_richness", "increase_material_development"],
            "confidence": 0.91,
            "ambiguities": [],
        }),
    )

    assert result["status"] == "interpreted"
    assert result["target_levels"] == ["D3"]
    assert result["combined_control_deltas"] == {
        "richness_weight": 0.07,
        "material_development_weight": 0.08,
    }
    assert result["interpreter"] == "local_llm"


def test_local_llm_does_not_turn_preserved_energy_into_an_increase():
    result = interpret_with_local_llm(
        "Κρατάμε την ενέργεια, αλλά οι ήχοι φεύγουν απότομα",
        "D5",
        request_json=fake_response({
            "summary_el": "Διατηρεί την ενέργεια και ζητά ομαλότερες αποχωρήσεις.",
            "intents": ["increase_smoothness"],
            "confidence": 0.96,
            "ambiguities": [],
        }),
    )

    assert [action["intent"] for action in result["actions"]] == ["increase_smoothness"]
    assert "activity_weight" not in result["combined_control_deltas"]


def test_low_confidence_model_output_cannot_be_applied():
    result = interpret_with_local_llm(
        "Κάπως άλλο αλλά όχι ακριβώς",
        "D1",
        request_json=fake_response({
            "summary_el": "Το αίτημα δεν είναι σαφές.",
            "intents": ["increase_activity"],
            "confidence": 0.40,
            "ambiguities": ["Δεν είναι σαφές ποιο στοιχείο πρέπει να αλλάξει."],
        }),
    )

    assert result["status"] == "unrecognised"
    assert result["actions"] == []
    assert result["combined_control_deltas"] == {}


def test_unknown_model_intent_is_rejected():
    with pytest.raises(LocalLLMUnavailable, match="unsupported intent"):
        interpret_with_local_llm(
            "Κάν' το διαφορετικό",
            "D1",
            request_json=fake_response({
                "summary_el": "Άγνωστη αλλαγή.",
                "intents": ["rewrite_everything"],
                "confidence": 0.9,
                "ambiguities": [],
            }),
        )


def test_auto_mode_falls_back_to_rules_only_on_technical_failure():
    def unavailable(_text, _level):
        raise LocalLLMUnavailable("offline")

    preview = build_feedback_preview(
        "Περισσότερη μουσικότητα",
        dream_level="D1",
        interpreter="auto",
        llm_interpreter=unavailable,
    )

    assert preview["can_apply"] is True
    assert preview["interpreter"] == "rules_fallback"
    assert preview["fallback_reason"] == "offline"


def test_text_and_voice_share_the_same_local_model_interpretation_path():
    calls = []

    def interpreter(text, level):
        calls.append((text, level))
        return interpret_with_local_llm(
            text,
            level,
            request_json=fake_response({
                "summary_el": "Ζητά περισσότερα arpeggios.",
                "intents": ["increase_arpeggios"],
                "confidence": 0.95,
                "ambiguities": [],
            }),
        )

    typed = build_feedback_preview(
        "Θέλω πιο κινούμενες νότες",
        dream_level="D5",
        source="text",
        interpreter="local_llm",
        llm_interpreter=interpreter,
    )
    spoken = build_feedback_preview(
        "Θέλω πιο κινούμενες νότες",
        dream_level="D5",
        source="voice",
        locale="el-GR",
        interpreter="local_llm",
        llm_interpreter=interpreter,
    )

    assert calls == [("Θέλω πιο κινούμενες νότες", "D5")] * 2
    assert typed["actions"] == spoken["actions"]
    assert typed["interpreter"] == spoken["interpreter"] == "local_llm"

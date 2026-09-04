import json

from composition_feedback_v1 import build_composition_feedback
from composition_influence_v1 import DOMAIN_CONTROLS, describe_composition_influence


RATINGS = {
    "musicality": 2.5,
    "material_coherence": 2.5,
    "transition_smoothness": 2.5,
    "variety_without_disconnection": 3.0,
    "synth_material_presence": 2.0,
    "development_over_repetition": 2.5,
    "overall_artistic_impression": 3.0,
}


def test_control_deltas_are_explained_as_whole_composition_domains():
    influence = describe_composition_influence({
        "synthetic_material_weight": 0.08,
        "activity_weight": 0.05,
        "transition_smoothness_weight": 0.08,
        "material_development_weight": 0.08,
    })
    assert set(influence["domain_ids"]) >= {
        "material_selection",
        "synth_and_arpeggios",
        "energy_and_density",
        "material_development",
        "transitions",
        "overall_form",
    }
    assert influence["total_domain_count"] == len(DOMAIN_CONTROLS)


def test_ratings_and_voice_comment_form_one_auditable_composition_event():
    def fake_llm(text, level):
        assert "ασύνδετο" in text
        assert level == "D5"
        return {
            "status": "interpreted",
            "scope": "level",
            "target_level": "D5",
            "target_levels": ["D5"],
            "actions": [{
                "intent": "increase_coherence",
                "control_deltas": {"coherence_weight": 0.07},
            }],
            "combined_control_deltas": {"coherence_weight": 0.07},
            "confidence": 0.92,
            "summary_el": "Ζητά περισσότερες συνδέσεις μεταξύ των υλικών.",
            "ambiguities": [],
            "interpreter": "local_llm",
            "model": "test-model",
        }

    event = build_composition_feedback(
        RATINGS,
        dream_level="D5",
        keep_as_baseline=False,
        comment="Ήταν λίγο ασύνδετο και ήθελα περισσότερες σχέσεις.",
        source="voice",
        locale="el",
        interpreter="local_llm",
        llm_interpreter=fake_llm,
    )
    assert event["listener_input"] == {"source": "voice", "locale": "el"}
    assert event["comment_interpretation"]["interpreter"] == "local_llm"
    assert event["requested_control_deltas"]["coherence_weight"] > 0.07
    assert len(event["composition_influence"]["domain_ids"]) >= 6
    json.dumps(event, ensure_ascii=False)


def test_rating_vector_reaches_every_composition_domain_for_low_mixed_review():
    event = build_composition_feedback(
        RATINGS,
        dream_level="D3",
        keep_as_baseline=False,
        more="περισσότερο synth, arpeggios, ενέργεια και πλουσιότερα layers",
        less="θολούρα, μπάσα και απότομες μεταβάσεις",
    )
    assert set(event["composition_influence"]["domain_ids"]) == set(DOMAIN_CONTROLS)

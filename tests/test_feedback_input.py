import copy
import json

import pytest

from feedback_input_v1 import apply_feedback_preview, build_feedback_preview
from human_feedback_v1 import DEFAULT_LEARNING_PROFILE
from hyponoia_feedback_app import format_preview


def test_text_feedback_previews_greek_intents_without_mutation():
    profile = copy.deepcopy(DEFAULT_LEARNING_PROFILE)
    before = copy.deepcopy(profile)
    preview = build_feedback_preview(
        "Θέλω περισσότερη μουσικότητα, synth και πιο ομαλές μεταβάσεις",
        dream_level="D3",
        source="text",
        profile=profile,
    )

    assert preview["can_apply"] is True
    assert preview["target_levels"] == ["D3"]
    assert {item["intent"] for item in preview["actions"]} >= {
        "increase_musicality",
        "increase_synthetic_material",
        "increase_smoothness",
    }
    assert {item["target_level"] for item in preview["control_changes"]} == {"D3"}
    assert profile == before


def test_voice_transcript_uses_exactly_the_same_interpreter_as_text():
    text = "More arpeggios, more energy and less repetition"
    typed = build_feedback_preview(text, dream_level="D5", source="text")
    spoken = build_feedback_preview(text, dream_level="D5", source="voice", locale="en-GB")

    assert spoken["source"] == "voice"
    assert spoken["locale"] == "en-GB"
    assert spoken["actions"] == typed["actions"]
    assert spoken["control_changes"] == typed["control_changes"]


def test_unrecognised_comment_cannot_be_applied(tmp_path):
    preview = build_feedback_preview("μου άρεσε το μπλε", dream_level="D1")
    assert preview["status"] == "unrecognised"
    assert preview["can_apply"] is False
    with pytest.raises(ValueError, match="no recognised changes"):
        apply_feedback_preview(
            preview,
            profile_path=tmp_path / "learning_profile.json",
            evidence_dir=tmp_path / "human_feedback",
            confirmed=True,
        )


def test_apply_requires_confirmation_and_persists_level_specific_evidence(tmp_path):
    profile_path = tmp_path / "learning_profile.json"
    evidence_dir = tmp_path / "human_feedback"
    preview = build_feedback_preview(
        "Περισσότερα arpeggios και λιγότερη επανάληψη",
        dream_level="D5",
        source="voice",
        locale="el-GR",
    )

    with pytest.raises(ValueError, match="explicitly confirmed"):
        apply_feedback_preview(
            preview,
            profile_path=profile_path,
            evidence_dir=evidence_dir,
        )

    profile, event = apply_feedback_preview(
        preview,
        profile_path=profile_path,
        evidence_dir=evidence_dir,
        confirmed=True,
    )

    assert event["source"] == "voice"
    assert event["target_levels"] == ["D5"]
    assert event["policy"]["shared_text_voice_interpreter"] is True
    assert profile["level_weights"]["D1"] == DEFAULT_LEARNING_PROFILE["level_weights"]["D1"]
    assert profile["level_weights"]["D3"] == DEFAULT_LEARNING_PROFILE["level_weights"]["D3"]
    assert profile["level_weights"]["D5"]["arpeggio_weight"] > 1.0
    assert profile["level_weights"]["D5"]["repetition_control"] > 1.0
    assert profile["history"][-1]["event_id"] == event["event_id"]
    assert json.loads(profile_path.read_text(encoding="utf-8"))["history"][-1]["source"] == "voice"
    assert len(list(evidence_dir.glob("*_free_feedback.json"))) == 1


def test_explicit_global_comment_is_previewed_for_all_levels():
    preview = build_feedback_preview(
        "Σε όλα τα επίπεδα περισσότερη μουσικότητα",
        dream_level="D1",
    )
    assert preview["scope"] == "global"
    assert preview["target_levels"] == ["D1", "D3", "D5"]
    assert {item["target_level"] for item in preview["control_changes"]} == {"D1", "D3", "D5"}


def test_nontechnical_preview_explains_targets_and_actions():
    preview = build_feedback_preview(
        "Περισσότερη μουσικότητα και πιο ομαλές μεταβάσεις",
        dream_level="D1",
    )
    text = format_preview(preview)
    assert "Θα επηρεαστεί: D1" in text
    assert "περισσότερη μουσικότητα" in text
    assert "ομαλότερες μεταβάσεις" in text
    assert "musicality_weight" in text

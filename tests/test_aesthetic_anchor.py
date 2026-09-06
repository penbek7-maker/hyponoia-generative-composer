from aesthetic_anchor_v1 import ANCHOR_CONTROLS, transfer_aesthetic_anchor


def test_d1_anchor_guides_quality_without_cloning_level_identity():
    profile = {
        "level_weights": {
            "D1": {
                "musicality_weight": 1.20,
                "transition_smoothness_weight": 1.16,
                "activity_weight": 1.05,
                "arpeggio_weight": 1.20,
            },
            "D3": {
                "musicality_weight": 1.00,
                "transition_smoothness_weight": 1.00,
                "activity_weight": 1.18,
                "arpeggio_weight": 1.00,
            },
            "D5": {
                "musicality_weight": 1.08,
                "transition_smoothness_weight": 1.02,
                "activity_weight": 1.24,
                "arpeggio_weight": 1.00,
            },
        }
    }
    updated, event = transfer_aesthetic_anchor(
        profile,
        source_level="D1",
        target_levels=("D3", "D5"),
        strength=0.75,
    )
    assert updated["level_weights"]["D3"]["musicality_weight"] > 1.00
    assert updated["level_weights"]["D5"]["transition_smoothness_weight"] > 1.02
    assert updated["level_weights"]["D3"]["activity_weight"] == 1.18
    assert updated["level_weights"]["D5"]["arpeggio_weight"] == 1.00
    assert "arpeggio_weight" not in ANCHOR_CONTROLS
    assert event["source_level"] == "D1"


def test_anchor_rejects_unbounded_strength():
    profile = {"level_weights": {"D1": {}, "D3": {}}}
    try:
        transfer_aesthetic_anchor(
            profile,
            source_level="D1",
            target_levels=("D3",),
            strength=1.2,
        )
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("unbounded anchor strength was accepted")

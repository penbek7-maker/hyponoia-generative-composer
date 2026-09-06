from library_coverage_v1 import (
    coverage_snapshot,
    recording_coverage_factor,
    sync_recording_coverage,
    update_recording_coverage,
)


def test_mutable_library_adds_retires_and_restores_coverage_entries():
    profile = {}
    first = sync_recording_coverage(profile, ["a.wav", "b.wav"])
    assert first == {
        "added": ["a.wav", "b.wav"],
        "restored": [],
        "removed": [],
    }

    second = sync_recording_coverage(profile, ["b.wav", "c.wav"])
    assert second == {
        "added": ["c.wav"],
        "restored": [],
        "removed": ["a.wav"],
    }
    assert set(profile["recording_coverage"]) == {"b.wav", "c.wav"}

    third = sync_recording_coverage(profile, ["a.wav", "b.wav", "c.wav"])
    assert third == {"added": [], "restored": ["a.wav"], "removed": []}


def test_actual_render_usage_updates_coverage_and_prioritises_unheard_material():
    profile = {}
    first = update_recording_coverage(
        profile,
        ["heard.wav", "unheard.wav"],
        {"heard.wav": 4},
    )
    assert first["completed_renders"] == 1
    assert first["heard_recordings"] == 1
    assert first["coverage_ratio"] == 0.5
    assert profile["recording_coverage"]["heard.wav"]["event_selections"] == 4
    assert profile["recording_coverage"]["unheard.wav"]["renders_since_used"] == 1
    assert recording_coverage_factor(
        "unheard.wav", profile
    ) < recording_coverage_factor("heard.wav", profile)

    second = update_recording_coverage(
        profile,
        ["heard.wav", "unheard.wav"],
        {"unheard.wav": 2},
    )
    assert second["heard_recordings"] == 2
    assert second["never_heard_recordings"] == 0
    assert coverage_snapshot(profile)["coverage_ratio"] == 1.0


def test_coverage_priority_is_bounded_and_never_bans_familiar_sources():
    profile = {}
    sync_recording_coverage(profile, ["old.wav", "new.wav"])
    profile["completed_renders"] = 20
    profile["recording_coverage"]["old.wav"].update(
        {"render_appearances": 18, "renders_since_used": 0}
    )
    profile["recording_coverage"]["new.wav"].update(
        {"render_appearances": 0, "renders_since_used": 20}
    )
    new_factor = recording_coverage_factor("new.wav", profile, strength=2.0)
    old_factor = recording_coverage_factor("old.wav", profile, strength=2.0)
    assert 0.68 <= new_factor <= 1.28
    assert 0.68 <= old_factor <= 1.28
    assert new_factor < old_factor

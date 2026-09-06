"""Persistent, library-aware rotation for Hyponoia source recordings.

The composer still chooses a coherent subset for each render. This module
remembers which source recordings were actually heard so that additions enter
the rotation, removed files leave it cleanly, and neglected recordings receive
a bounded advantage in later renders.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def _new_entry() -> dict[str, Any]:
    return {
        "render_appearances": 0,
        "event_selections": 0,
        "renders_since_used": 0,
        "last_used_render": None,
    }


def sync_recording_coverage(
    profile: dict[str, Any], recordings: Iterable[str]
) -> dict[str, list[str]]:
    """Synchronise persistent coverage memory with a mutable sound library."""
    active_names = sorted({str(name) for name in recordings if str(name)})
    active_set = set(active_names)
    coverage = profile.setdefault("recording_coverage", {})
    retired = profile.setdefault("retired_recording_coverage", {})
    profile.setdefault("completed_renders", 0)

    removed = sorted(name for name in coverage if name not in active_set)
    for name in removed:
        retired[name] = coverage.pop(name)

    added = []
    restored = []
    for name in active_names:
        if name in coverage:
            continue
        if name in retired:
            coverage[name] = retired.pop(name)
            restored.append(name)
        else:
            coverage[name] = _new_entry()
            added.append(name)

    return {"added": added, "restored": restored, "removed": removed}


def recording_coverage_factor(
    recording: str,
    profile: Mapping[str, Any] | None,
    *,
    strength: float = 1.0,
) -> float:
    """Return a bounded score multiplier; lower means greater rotation priority."""
    if not profile:
        return 1.0
    coverage = profile.get("recording_coverage", {})
    entry = coverage.get(str(recording))
    if not entry or not coverage:
        return 1.0

    appearances = [
        max(0, int(item.get("render_appearances", 0)))
        for item in coverage.values()
    ]
    mean_appearances = sum(appearances) / max(1, len(appearances))
    own_appearances = max(0, int(entry.get("render_appearances", 0)))
    stale = max(0, int(entry.get("renders_since_used", 0)))
    bounded_strength = max(0.5, min(1.6, float(strength)))

    overuse_penalty = max(
        -0.16,
        min(0.20, (own_appearances - mean_appearances) * 0.045),
    )
    staleness_bonus = min(0.24, stale * 0.035)
    never_heard_bonus = (
        0.10
        if own_appearances == 0 and int(profile.get("completed_renders", 0))
        else 0.0
    )
    exponent = (
        overuse_penalty - staleness_bonus - never_heard_bonus
    ) * bounded_strength
    return float(max(0.68, min(1.28, math.exp(exponent))))


def update_recording_coverage(
    profile: dict[str, Any],
    recordings: Iterable[str],
    usage_by_recording: Mapping[str, int],
) -> dict[str, Any]:
    """Record one completed render using actual audible event selections."""
    sync = sync_recording_coverage(profile, recordings)
    render_number = max(0, int(profile.get("completed_renders", 0))) + 1
    profile["completed_renders"] = render_number
    coverage = profile["recording_coverage"]

    for name, entry in coverage.items():
        selections = max(0, int(usage_by_recording.get(name, 0)))
        if selections:
            entry["render_appearances"] = (
                max(0, int(entry.get("render_appearances", 0))) + 1
            )
            entry["event_selections"] = (
                max(0, int(entry.get("event_selections", 0))) + selections
            )
            entry["renders_since_used"] = 0
            entry["last_used_render"] = render_number
        else:
            entry["renders_since_used"] = (
                max(0, int(entry.get("renders_since_used", 0))) + 1
            )

    snapshot = coverage_snapshot(profile)
    snapshot["library_sync"] = sync
    return snapshot


def coverage_snapshot(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Create an auditable summary without exposing the full persistent profile."""
    coverage = (profile or {}).get("recording_coverage", {})
    total = len(coverage)
    heard = sum(
        1
        for entry in coverage.values()
        if int(entry.get("render_appearances", 0)) > 0
    )
    never_heard = sorted(
        name
        for name, entry in coverage.items()
        if int(entry.get("render_appearances", 0)) == 0
    )
    priority = sorted(
        coverage,
        key=lambda name: (
            int(coverage[name].get("render_appearances", 0)),
            -int(coverage[name].get("renders_since_used", 0)),
            name,
        ),
    )
    return {
        "completed_renders": max(
            0, int((profile or {}).get("completed_renders", 0))
        ),
        "library_recordings": total,
        "heard_recordings": heard,
        "never_heard_recordings": len(never_heard),
        "coverage_ratio": round(heard / total, 6) if total else 0.0,
        "next_rotation_priority": priority[: min(12, len(priority))],
        "policy": (
            "coherent subset per render with bounded priority for underused "
            "recordings"
        ),
    }

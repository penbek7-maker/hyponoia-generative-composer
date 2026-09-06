"""Transfer an accepted D-level's broad aesthetic balance without cloning it."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from hyponoia_stability import atomic_write_json, utc_timestamp


ANCHOR_CONTROLS = (
    "musicality_weight",
    "coherence_weight",
    "richness_weight",
    "transition_smoothness_weight",
    "synthetic_material_weight",
    "low_frequency_control",
    "layer_clarity_weight",
)


def transfer_aesthetic_anchor(
    profile: dict[str, Any],
    *,
    source_level: str,
    target_levels: tuple[str, ...],
    strength: float = 0.75,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blend broad mix/phrasing preferences while preserving D-level identity.

    Activity, exploration, granulation and arpeggio controls deliberately stay
    level-specific. This lets an accepted D1 guide quality without turning D3
    and D5 into copies of the same composition.
    """
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    levels = copy.deepcopy(profile.get("level_weights", {}))
    if source_level not in levels:
        raise ValueError(f"missing source level: {source_level}")
    updated = copy.deepcopy(profile)
    updated_levels = updated.setdefault("level_weights", {})
    changes = []
    for target_level in target_levels:
        if target_level not in updated_levels:
            raise ValueError(f"missing target level: {target_level}")
        for control in ANCHOR_CONTROLS:
            source_value = float(levels[source_level].get(control, 1.0))
            old_value = float(updated_levels[target_level].get(control, 1.0))
            new_value = max(0.5, min(1.8, old_value + (source_value - old_value) * strength))
            updated_levels[target_level][control] = new_value
            changes.append({
                "target_level": target_level,
                "control": control,
                "old_value": old_value,
                "anchor_value": source_value,
                "new_value": new_value,
            })
    event = {
        "schema_version": "aesthetic_anchor_v1",
        "source_level": source_level,
        "target_levels": list(target_levels),
        "strength": float(strength),
        "transferred_controls": list(ANCHOR_CONTROLS),
        "preserved_level_specific_controls": [
            "activity_weight",
            "exploration_weight",
            "repetition_control",
            "structured_granulation_weight",
            "arpeggio_weight",
        ],
        "changes": changes,
    }
    return updated, event


def apply_aesthetic_anchor_file(
    profile_path: str | Path,
    *,
    source_level: str,
    target_levels: tuple[str, ...],
    strength: float = 0.75,
    event_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist one auditable anchor event through the project's atomic writer."""
    path = Path(profile_path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    updated, event = transfer_aesthetic_anchor(
        profile,
        source_level=source_level,
        target_levels=target_levels,
        strength=strength,
    )
    event["timestamp"] = utc_timestamp()
    updated.setdefault("history", []).append(event)
    atomic_write_json(path, updated)
    if event_path is not None:
        atomic_write_json(Path(event_path), event)
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--source", required=True, choices=("D1", "D3", "D5"))
    parser.add_argument("--targets", nargs="+", required=True, choices=("D1", "D3", "D5"))
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--event", type=Path)
    args = parser.parse_args()
    event = apply_aesthetic_anchor_file(
        args.profile,
        source_level=args.source,
        target_levels=tuple(args.targets),
        strength=args.strength,
        event_path=args.event,
    )
    print(json.dumps(event, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

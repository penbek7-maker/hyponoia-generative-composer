"""Composition-wide preference head over Hyponoia's learned audio embeddings."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hyponoia_stability import atomic_write_json, utc_timestamp


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.ndim != 1 or not np.isfinite(values).all() or norm < 1e-12:
        raise ValueError("invalid preference vector")
    return values / norm


def _load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) < 2:
        raise ValueError("embedding file must contain at least two objects")
    return {str(key): _unit(np.asarray(value, dtype=np.float64)) for key, value in payload.items()}


def render_prototype(
    report: dict[str, Any], embeddings: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Summarise one entire render from every embedded object it actually used."""
    vectors = []
    weights = []
    missing = []
    recordings = set()
    for detail in report.get("sample_usage_details", {}).values():
        object_id = str(detail.get("object_id", ""))
        vector = embeddings.get(object_id)
        if vector is None:
            missing.append(object_id)
            continue
        selections = max(1.0, float(detail.get("selection_count", 1.0)))
        exposure = max(0.0, float(detail.get("exposure_sec", 0.0)))
        weight = np.sqrt(selections) * (1.0 + 0.18 * np.log1p(exposure))
        vectors.append(vector)
        weights.append(weight)
        recordings.add(str(detail.get("recording", "")))
    if not vectors:
        raise ValueError("render report has no objects covered by the embeddings")
    prototype = _unit(np.average(np.stack(vectors), axis=0, weights=np.asarray(weights)))
    return prototype, {
        "audio_file": report.get("audio_file"),
        "dream_level": report.get("dream_level"),
        "covered_objects": len(vectors),
        "missing_objects": missing,
        "recordings": sorted(recording for recording in recordings if recording),
    }


def render_structure(report: dict[str, Any]) -> dict[str, Any]:
    """Extract whole-render relationships that fragment embeddings cannot see."""
    role_counts = {
        str(role): max(0.0, float(count))
        for role, count in dict(report.get("role_counts", {})).items()
    }
    role_total = sum(role_counts.values()) or 1.0
    temporal = dict(report.get("temporal_metrics", {}))
    total_selections = max(
        1.0,
        float(report.get("total_sample_selections", sum(role_counts.values()) or 1.0)),
    )
    return {
        "event_rate_per_minute": float(
            temporal.get("event_rate_per_minute", total_selections / 3.0)
        ),
        "foreground_event_rate_per_minute": float(
            temporal.get("foreground_event_rate_per_minute", 0.0)
        ),
        "average_event_duration_sec": float(
            temporal.get("average_event_duration_sec", 0.0)
        ),
        "quick_succession_count": float(temporal.get("quick_succession_count", 0.0)),
        "role_distribution": {
            role: count / role_total for role, count in role_counts.items()
        },
        "unique_recordings": int(len(report.get("recordings", {}))),
    }


def average_structures(structures: list[dict[str, Any]]) -> dict[str, Any]:
    if not structures:
        return {}
    scalar_keys = (
        "event_rate_per_minute",
        "foreground_event_rate_per_minute",
        "average_event_duration_sec",
        "quick_succession_count",
        "unique_recordings",
    )
    roles = sorted(
        {role for item in structures for role in item.get("role_distribution", {})}
    )
    averaged = {
        key: float(np.mean([float(item.get(key, 0.0)) for item in structures]))
        for key in scalar_keys
    }
    averaged["role_distribution"] = {
        role: float(
            np.mean([item.get("role_distribution", {}).get(role, 0.0) for item in structures])
        )
        for role in roles
    }
    return averaged


def train_composition_preference(
    *,
    embeddings_path: str | Path,
    positive_reports: list[str | Path],
    negative_reports: list[str | Path],
    output_path: str | Path,
    positive_target: float = 0.98,
    negative_target: float = 0.30,
    strength: float = 0.42,
) -> dict[str, Any]:
    """Fit a bounded contrastive prototype from whole-render human choices."""
    if not positive_reports or not negative_reports:
        raise ValueError("at least one positive and one contrast report are required")
    if not 0.0 <= negative_target < positive_target <= 1.0:
        raise ValueError("preference targets must satisfy 0 <= negative < positive <= 1")
    if not 0.0 <= strength <= 0.6:
        raise ValueError("strength must be between 0 and 0.6")

    embeddings_path = Path(embeddings_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    embeddings = _load_embeddings(embeddings_path)
    positive_vectors = []
    negative_vectors = []
    positive_evidence = []
    negative_evidence = []
    positive_structures = []
    negative_structures = []
    reference_recordings = set()

    for path in positive_reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        vector, evidence = render_prototype(report, embeddings)
        positive_vectors.append(vector)
        positive_evidence.append(evidence)
        positive_structures.append(render_structure(report))
        reference_recordings.update(evidence["recordings"])
    for path in negative_reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        vector, evidence = render_prototype(report, embeddings)
        negative_vectors.append(vector)
        negative_evidence.append(evidence)
        negative_structures.append(render_structure(report))

    positive = _unit(np.mean(np.stack(positive_vectors), axis=0))
    negative = _unit(np.mean(np.stack(negative_vectors), axis=0))
    separation = float(np.clip(1.0 - np.dot(positive, negative), 0.0, 2.0))
    positive_margins = [
        float(np.dot(vector, positive) - np.dot(vector, negative))
        for vector in positive_vectors
    ]
    negative_margins = [
        float(np.dot(vector, positive) - np.dot(vector, negative))
        for vector in negative_vectors
    ]
    for report_path, evidence in zip(positive_reports, positive_evidence):
        evidence["audio_file"] = Path(str(evidence.get("audio_file", ""))).name
        evidence["source_report"] = "/".join(Path(report_path).parts[-3:])
    for report_path, evidence in zip(negative_reports, negative_evidence):
        evidence["audio_file"] = Path(str(evidence.get("audio_file", ""))).name
        evidence["source_report"] = "/".join(Path(report_path).parts[-3:])

    level_heads = {}
    positive_levels = [item.get("dream_level") for item in positive_evidence]
    negative_levels = [item.get("dream_level") for item in negative_evidence]
    for level in (1, 3, 5):
        positive_indices = [i for i, value in enumerate(positive_levels) if value == level]
        negative_indices = [i for i, value in enumerate(negative_levels) if value == level]
        if not positive_indices or not negative_indices:
            continue
        level_positive_vectors = [positive_vectors[i] for i in positive_indices]
        level_negative_vectors = [negative_vectors[i] for i in negative_indices]
        level_positive = _unit(np.mean(np.stack(level_positive_vectors), axis=0))
        level_negative = _unit(np.mean(np.stack(level_negative_vectors), axis=0))
        positive_level_margins = [
            float(np.dot(vector, level_positive) - np.dot(vector, level_negative))
            for vector in level_positive_vectors
        ]
        negative_level_margins = [
            float(np.dot(vector, level_positive) - np.dot(vector, level_negative))
            for vector in level_negative_vectors
        ]
        level_heads[str(level)] = {
            "positive_prototype": level_positive.tolist(),
            "negative_prototype": level_negative.tolist(),
            "positive_structure": average_structures(
                [positive_structures[i] for i in positive_indices]
            ),
            "contrast_structure": average_structures(
                [negative_structures[i] for i in negative_indices]
            ),
            "reference_recordings": sorted({
                recording
                for i in positive_indices
                for recording in positive_evidence[i]["recordings"]
            }),
            "training_diagnostics": {
                "positive_render_count": len(positive_indices),
                "contrast_render_count": len(negative_indices),
                "mean_training_margin_gap": float(
                    np.mean(positive_level_margins) - np.mean(negative_level_margins)
                ),
            },
        }
    model = {
        "schema_version": "composition_preference_v1",
        "model_type": "contrastive_prototype_head_over_deep_embeddings",
        "trained_at": utc_timestamp(),
        "mode": "assist",
        "embeddings_path": os.path.relpath(embeddings_path, output_path.parent),
        "embedding_dimensions": int(len(positive)),
        "strength": float(strength),
        "positive_target": float(positive_target),
        "negative_target": float(negative_target),
        "positive_prototype": positive.tolist(),
        "negative_prototype": negative.tolist(),
        "positive_evidence": positive_evidence,
        "contrast_evidence": negative_evidence,
        "reference_recordings": sorted(reference_recordings),
        "reference_reuse_penalty": 1.08,
        "positive_structure": average_structures(positive_structures),
        "contrast_structure": average_structures(negative_structures),
        "level_heads": level_heads,
        "training_diagnostics": {
            "positive_contrast_separation": separation,
            "positive_training_margins": positive_margins,
            "contrast_training_margins": negative_margins,
            "mean_training_margin_gap": float(
                np.mean(positive_margins) - np.mean(negative_margins)
            ),
            "positive_render_count": len(positive_vectors),
            "contrast_render_count": len(negative_vectors),
            "covered_positive_objects": sum(item["covered_objects"] for item in positive_evidence),
            "covered_contrast_objects": sum(item["covered_objects"] for item in negative_evidence),
            "policy": "strong human preference; bounded assist; small-set training evidence, not a generalisation claim",
        },
    }
    atomic_write_json(output_path, model)
    return model


@dataclass
class CompositionPreferenceAssist:
    mode: str = "off"
    strength: float = 0.0
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    positive: np.ndarray | None = None
    negative: np.ndarray | None = None
    reference_recordings: set[str] = field(default_factory=set)
    reference_reuse_penalty: float = 1.0
    positive_structure: dict[str, Any] = field(default_factory=dict)
    positive_dream_levels: set[int] = field(default_factory=set)
    level_heads: dict[int, dict[str, Any]] = field(default_factory=dict)
    source_path: str | None = None
    error: str | None = None

    @classmethod
    def disabled(cls) -> "CompositionPreferenceAssist":
        return cls()

    @classmethod
    def from_file(cls, path: str | Path) -> "CompositionPreferenceAssist":
        source = Path(path).expanduser()
        if not source.exists():
            return cls(mode="off", source_path=str(source), error=f"Preference model not found: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "composition_preference_v1":
                raise ValueError("unsupported composition preference schema")
            strength = float(payload.get("strength", 0.42))
            if not 0.0 <= strength <= 0.6:
                raise ValueError("strength must be between 0 and 0.6")
            embedding_path = Path(payload["embeddings_path"]).expanduser()
            if not embedding_path.is_absolute():
                embedding_path = source.parent / embedding_path
            evidence_unique_objects = {
                int(item["dream_level"]): int(item.get("covered_objects", 0))
                for item in payload.get("positive_evidence", [])
                if item.get("dream_level") in (1, 3, 5)
            }
            positive_structure = dict(payload.get("positive_structure", {}))
            if evidence_unique_objects:
                positive_structure.setdefault(
                    "unique_objects",
                    float(np.mean(list(evidence_unique_objects.values()))),
                )
            level_heads = {}
            for level, head in dict(payload.get("level_heads", {})).items():
                parsed_level = int(level)
                if parsed_level not in (1, 3, 5):
                    continue
                positive_structure = dict(head.get("positive_structure", {}))
                if evidence_unique_objects.get(parsed_level, 0) > 0:
                    positive_structure.setdefault(
                        "unique_objects", evidence_unique_objects[parsed_level]
                    )
                level_heads[parsed_level] = {
                    "positive": _unit(np.asarray(head["positive_prototype"], dtype=np.float64)),
                    "negative": _unit(np.asarray(head["negative_prototype"], dtype=np.float64)),
                    "positive_structure": positive_structure,
                    "reference_recordings": set(map(str, head.get("reference_recordings", []))),
                }
            return cls(
                mode=str(payload.get("mode", "off")),
                strength=strength,
                embeddings=_load_embeddings(embedding_path),
                positive=_unit(np.asarray(payload["positive_prototype"], dtype=np.float64)),
                negative=_unit(np.asarray(payload["negative_prototype"], dtype=np.float64)),
                reference_recordings=set(map(str, payload.get("reference_recordings", []))),
                reference_reuse_penalty=float(payload.get("reference_reuse_penalty", 1.08)),
                positive_structure=positive_structure,
                positive_dream_levels={
                    int(item["dream_level"])
                    for item in payload.get("positive_evidence", [])
                    if item.get("dream_level") in (1, 3, 5)
                },
                level_heads=level_heads,
                source_path=str(source.resolve()),
            )
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return cls(mode="off", source_path=str(source), error=str(exc))

    @property
    def active(self) -> bool:
        return (
            self.mode == "assist"
            and bool(self.embeddings)
            and self.positive is not None
            and self.negative is not None
            and self.error is None
        )

    def object_factor(
        self, object_id: str | None, recording: str | None = None, dream_level: int | None = None
    ) -> float:
        """Lower favours the accepted embedding region for this D-level."""
        if not self.active or not object_id:
            return 1.0
        if (
            dream_level in (1, 3, 5)
            and self.positive_dream_levels
            and int(dream_level) not in self.positive_dream_levels
        ):
            return 1.0
        vector = self.embeddings.get(str(object_id))
        if vector is None:
            return 1.0
        head = self.level_heads.get(int(dream_level)) if dream_level in (1, 3, 5) else None
        positive = head["positive"] if head else self.positive
        negative = head["negative"] if head else self.negative
        references = head["reference_recordings"] if head else self.reference_recordings
        margin = float(np.dot(vector, positive) - np.dot(vector, negative))
        factor = float(np.exp(-self.strength * margin))
        # Level-specific heads already learn the accepted material region.
        # Library coverage supplies diversity, so a second reference penalty
        # here would push the composer away from the listener's gold palette.
        if head is None and dream_level in (3, 5) and recording in references:
            factor *= self.reference_reuse_penalty
        return float(max(0.76, min(1.28, factor)))

    def recording_factor(self, objects: list[dict[str, Any]], dream_level: int) -> float:
        if not self.active or not objects:
            return 1.0
        factors = [
            self.object_factor(obj.get("object_id"), obj.get("recording"), dream_level)
            for obj in objects
        ]
        return float(max(0.82, min(1.20, np.mean(factors))))

    def target_event_count(self, dream_level: int, duration_sec: float) -> int | None:
        """Transfer the positive render's economy while preserving level depth."""
        structure = self.structure_for_level(dream_level)
        if not self.active or not structure:
            return None
        rate = float(structure.get("event_rate_per_minute", 0.0))
        if rate <= 0.0:
            return None
        has_level_head = int(dream_level) in self.level_heads if dream_level in (1, 3, 5) else False
        depth = 1.0 if has_level_head else {1: 1.0, 3: 1.08, 5: 1.24}.get(int(dream_level), 1.0)
        target = rate * max(0.0, float(duration_sec)) / 60.0 * depth
        return max(1, int(round(target)))

    def structure_for_level(self, dream_level: int) -> dict[str, Any]:
        """Return the accepted whole-render structure for one dream level."""
        level_head = self.level_heads.get(int(dream_level)) if dream_level in (1, 3, 5) else None
        return level_head["positive_structure"] if level_head else self.positive_structure

    def target_average_event_duration(self, dream_level: int) -> float | None:
        """Learned phrase/layer duration from the accepted complete render."""
        structure = self.structure_for_level(dream_level)
        if not self.active or not structure:
            return None
        target = float(structure.get("average_event_duration_sec", 0.0))
        return target if target > 0.0 else None

    def target_unique_recordings(self, dream_level: int) -> int | None:
        """Minimum per-render source breadth learned from the accepted render."""
        structure = self.structure_for_level(dream_level)
        if not self.active or not structure:
            return None
        target = int(round(float(structure.get("unique_recordings", 0.0))))
        return target if target > 0 else None

    def target_unique_objects(self, dream_level: int) -> int | None:
        """Material-object breadth observed in the accepted complete render."""
        structure = self.structure_for_level(dream_level)
        if not self.active or not structure:
            return None
        target = int(round(float(structure.get("unique_objects", 0.0))))
        return target if target > 0 else None

    def target_role_distribution(self, dream_level: int) -> dict[str, float]:
        """Accepted balance of texture, resonance and foreground functions."""
        structure = self.structure_for_level(dream_level)
        if not self.active or not structure:
            return {}
        distribution = {
            str(role): max(0.0, float(share))
            for role, share in dict(structure.get("role_distribution", {})).items()
        }
        total = sum(distribution.values())
        if total <= 0.0:
            return {}
        return {role: share / total for role, share in distribution.items()}

    def role_factor(self, role: str, dream_level: int) -> float:
        """Softly transfer the positive render's role balance, not its samples."""
        level_head = self.level_heads.get(int(dream_level)) if dream_level in (1, 3, 5) else None
        structure = self.structure_for_level(dream_level)
        if not self.active or not structure:
            return 1.0
        distribution = dict(structure.get("role_distribution", {}))
        share = max(0.0, float(distribution.get(role, 0.0)))
        level_shape = {} if level_head else {
            1: {},
            3: {"gesture": 1.20, "texture": 1.06, "resonance": 0.88},
            5: {"gesture": 2.80, "texture": 0.90, "resonance": 0.25},
        }.get(int(dream_level), {})
        shaped_share = share * level_shape.get(role, 1.0)
        return float(max(0.68, min(1.38, np.sqrt(shaped_share / 0.20))))

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active": self.active,
            "strength": self.strength,
            "embedding_count": len(self.embeddings),
            "reference_recording_count": len(self.reference_recordings),
            "structure_active": bool(self.positive_structure),
            "positive_event_rate_per_minute": self.positive_structure.get(
                "event_rate_per_minute"
            ),
            "positive_average_event_duration_sec": self.positive_structure.get(
                "average_event_duration_sec"
            ),
            "positive_unique_recordings": self.positive_structure.get(
                "unique_recordings"
            ),
            "material_anchor_levels": sorted(self.positive_dream_levels),
            "level_specific_heads": sorted(self.level_heads),
            "source_path": self.source_path,
            "error": self.error,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--positive", required=True, action="append", type=Path)
    parser.add_argument("--contrast", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive-target", type=float, default=0.98)
    parser.add_argument("--negative-target", type=float, default=0.30)
    parser.add_argument("--strength", type=float, default=0.42)
    args = parser.parse_args()
    model = train_composition_preference(
        embeddings_path=args.embeddings,
        positive_reports=args.positive,
        negative_reports=args.contrast,
        output_path=args.output,
        positive_target=args.positive_target,
        negative_target=args.negative_target,
        strength=args.strength,
    )
    print(json.dumps(model["training_diagnostics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

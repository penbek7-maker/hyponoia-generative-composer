"""Bounded, fail-safe use of learned embeddings inside the Gate 1 generator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


VALID_MODES = frozenset({"off", "assist"})


@dataclass
class RepresentationAssist:
    mode: str = "off"
    strength: float = 0.35
    embeddings_path: str | None = None
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def disabled(cls) -> "RepresentationAssist":
        return cls(mode="off")

    @classmethod
    def from_config(cls, config_path: str | Path) -> "RepresentationAssist":
        config_path = Path(config_path)
        if not config_path.exists():
            return cls(mode="off", error=f"Configuration not found: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            mode = str(config.get("mode", "off")).strip().lower()
            if mode not in VALID_MODES:
                raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
            strength = float(config.get("strength", 0.35))
            if not 0.0 <= strength <= 0.5:
                raise ValueError("strength must be between 0.0 and 0.5")
            raw_path = config.get("embeddings_path")
            resolved = None
            if raw_path:
                resolved_path = Path(raw_path).expanduser()
                if not resolved_path.is_absolute():
                    resolved_path = config_path.parent / resolved_path
                resolved = str(resolved_path.resolve())
            assist = cls(mode=mode, strength=strength, embeddings_path=resolved)
            if mode == "assist":
                assist._load_embeddings()
            return assist
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return cls(mode="off", error=str(exc))

    def _load_embeddings(self) -> None:
        if not self.embeddings_path:
            raise ValueError("assist mode requires embeddings_path")
        path = Path(self.embeddings_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or len(payload) < 2:
            raise ValueError("embedding file must contain at least two stable IDs")
        if not all(isinstance(vector, list) for vector in payload.values()):
            raise ValueError("every embedding must be a JSON list")
        dimensions = {len(vector) for vector in payload.values()}
        if len(dimensions) != 1 or next(iter(dimensions), 0) < 1:
            raise ValueError("embeddings must all have one shared dimension")
        for stable_id, vector in payload.items():
            values = np.asarray(vector, dtype=np.float64)
            norm = float(np.linalg.norm(values))
            if values.ndim != 1 or not np.isfinite(values).all() or norm < 1e-12:
                raise ValueError(f"invalid embedding for {stable_id}")
            self.embeddings[str(stable_id)] = values / norm

    @property
    def active(self) -> bool:
        return self.mode == "assist" and bool(self.embeddings) and self.error is None

    def continuity_factor(self, previous_id: str | None, candidate_id: str | None) -> float:
        """Lower is better; the bounded factor cannot dominate Gate 1 scoring."""
        if not self.active or not previous_id or not candidate_id or previous_id == candidate_id:
            return 1.0
        previous = self.embeddings.get(str(previous_id))
        candidate = self.embeddings.get(str(candidate_id))
        if previous is None or candidate is None:
            return 1.0
        similarity = float(np.clip(np.dot(previous, candidate), -1.0, 1.0))
        factor = float(np.exp(-self.strength * (similarity - 0.5)))
        return float(max(0.82, min(1.18, factor)))

    def snapshot(self) -> dict:
        return {
            "mode": self.mode,
            "active": self.active,
            "strength": self.strength,
            "embeddings_path": self.embeddings_path,
            "embedding_count": len(self.embeddings),
            "error": self.error,
        }

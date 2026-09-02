"""Incrementally refresh Hyponoia embeddings with a frozen trained encoder.

Existing vectors are retained by content-derived stable ID. New sound objects
are embedded with the frozen encoder, optionally passed through the existing
human-calibrated metric adapter, and removed objects are pruned. No training or
source-audio mutation occurs in this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from representation_feedback_v2 import DiagonalMetricAdapter
from representation_learning_v1 import (
    EMBEDDING_DIM,
    ContrastiveEncoder,
    SoundObject,
    load_sound_objects,
    log_mel,
    read_fragment,
)
from representation_training_v1 import fixed_frames


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def load_existing_embeddings(path: str | Path) -> dict[str, list[float]]:
    """Load and validate an optional existing embedding map."""
    source = Path(path)
    if not source.exists():
        return {}
    payload = _load_json(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding file must contain a JSON object: {source}")
    if not payload:
        return {}
    if not all(isinstance(vector, list) for vector in payload.values()):
        raise ValueError("Every existing embedding must be a JSON list")
    dimensions = {len(vector) for vector in payload.values()}
    if len(dimensions) != 1 or next(iter(dimensions), 0) < 1:
        raise ValueError("Existing embeddings must share one non-zero dimension")
    matrix = np.asarray(list(payload.values()), dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Existing embeddings must be finite vectors")
    if np.any(np.linalg.norm(matrix, axis=1) < 1e-12):
        raise ValueError("Existing embeddings must have non-zero norm")
    return {str(key): [float(value) for value in vector] for key, vector in payload.items()}


def _load_encoder(path: Path) -> tuple[ContrastiveEncoder, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen encoder not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older supported torch releases.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid frozen encoder checkpoint: {path}")
    embedding_dim = int(checkpoint.get("embedding_dim", EMBEDDING_DIM))
    frames = int(checkpoint.get("frames", 128))
    if embedding_dim < 1 or frames < 1:
        raise ValueError(f"Invalid frozen encoder dimensions: {path}")
    model = ContrastiveEncoder(embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, embedding_dim, frames


def _load_adapter(path: Path | None, embedding_dim: int) -> DiagonalMetricAdapter | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Metric adapter not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid metric adapter checkpoint: {path}")
    adapter_dim = int(checkpoint.get("embedding_dim", embedding_dim))
    if adapter_dim != embedding_dim:
        raise ValueError("Encoder and metric adapter dimensions do not match")
    adapter = DiagonalMetricAdapter(embedding_dim)
    adapter.load_state_dict(checkpoint["model_state_dict"])
    adapter.eval()
    return adapter


def _embed_objects(
    objects: list[SoundObject],
    encoder: ContrastiveEncoder,
    frames: int,
    adapter: DiagonalMetricAdapter | None,
    *,
    batch_size: int = 32,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(objects), batch_size):
            batch = objects[start : start + batch_size]
            features = torch.stack(
                [fixed_frames(log_mel(read_fragment(item)), frames) for item in batch]
            )
            vectors = encoder(features)
            if adapter is not None:
                vectors = adapter(vectors)
            vectors = F.normalize(vectors, dim=1)
            for item, vector in zip(batch, vectors.cpu().numpy()):
                result[item.stable_id] = [float(value) for value in vector]
    return result


def prepare_incremental_embeddings(
    index_path: str | Path,
    memory_dir: str | Path,
    embeddings_path: str | Path,
    *,
    encoder_path: str | Path | None,
    adapter_path: str | Path | None = None,
    batch_size: int = 32,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Prepare a complete embedding payload without writing any files."""
    started = time.time()
    objects = load_sound_objects(index_path, memory_dir)
    unique_objects = {item.stable_id: item for item in objects}
    desired_ids = set(unique_objects)
    existing = load_existing_embeddings(embeddings_path)
    existing_ids = set(existing)
    reused_ids = sorted(desired_ids & existing_ids)
    new_ids = sorted(desired_ids - existing_ids)
    removed_ids = sorted(existing_ids - desired_ids)

    payload = {stable_id: existing[stable_id] for stable_id in reused_ids}
    embedding_dim = len(next(iter(existing.values()))) if existing else None
    used_encoder = False
    used_adapter = False

    if new_ids:
        if encoder_path is None:
            raise FileNotFoundError(
                "New sound objects need a frozen encoder, but encoder_path is not configured"
            )
        encoder, encoder_dim, frames = _load_encoder(Path(encoder_path))
        if embedding_dim is not None and encoder_dim != embedding_dim:
            raise ValueError("Existing embeddings and frozen encoder dimensions do not match")
        adapter = _load_adapter(Path(adapter_path) if adapter_path else None, encoder_dim)
        payload.update(
            _embed_objects(
                [unique_objects[stable_id] for stable_id in new_ids],
                encoder,
                frames,
                adapter,
                batch_size=batch_size,
            )
        )
        embedding_dim = encoder_dim
        used_encoder = True
        used_adapter = adapter is not None

    payload = dict(sorted(payload.items()))
    report = {
        "schema_version": 1,
        "policy": "frozen_encoder_incremental",
        "full_retraining": False,
        "active_sound_objects": len(objects),
        "active_unique_stable_ids": len(desired_ids),
        "previous_embeddings": len(existing_ids),
        "reused_embeddings": len(reused_ids),
        "created_embeddings": len(new_ids),
        "removed_embeddings": len(removed_ids),
        "embedding_dim": embedding_dim,
        "encoder_used": used_encoder,
        "adapter_used": used_adapter,
        "encoder_path": str(Path(encoder_path).resolve()) if encoder_path else None,
        "adapter_path": str(Path(adapter_path).resolve()) if adapter_path else None,
        "embeddings_path": str(Path(embeddings_path).resolve()),
        "elapsed_seconds": round(time.time() - started, 3),
        "status": "refreshed" if new_ids or removed_ids else "unchanged",
    }
    return payload, report


def resolve_configured_path(config_path: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def prepare_from_config(
    index_path: str | Path,
    memory_dir: str | Path,
    config_path: str | Path,
) -> tuple[Path | None, dict[str, list[float]] | None, dict[str, Any]]:
    """Prepare a refresh using the generator's representation configuration."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        return None, None, {
            "schema_version": 1,
            "status": "disabled",
            "reason": "representation configuration not found",
            "full_retraining": False,
        }
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid representation configuration: {config_path}")
    mode = str(config.get("mode", "off")).strip().lower()
    if mode == "off":
        return None, None, {
            "schema_version": 1,
            "status": "disabled",
            "reason": "representation assist is off",
            "full_retraining": False,
        }
    if mode != "assist":
        raise ValueError(f"Unsupported representation mode: {mode}")

    embeddings_path = resolve_configured_path(config_path, config.get("embeddings_path"))
    if embeddings_path is None:
        raise ValueError("Assist mode requires embeddings_path")
    encoder_path = resolve_configured_path(config_path, config.get("encoder_path"))
    adapter_path = resolve_configured_path(config_path, config.get("metric_adapter_path"))
    payload, report = prepare_incremental_embeddings(
        index_path,
        memory_dir,
        embeddings_path,
        encoder_path=encoder_path,
        adapter_path=adapter_path,
    )
    report["config_path"] = str(config_path)
    return embeddings_path, payload, report

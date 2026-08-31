"""Human-calibrated metric learning for Hyponoia representation embeddings.

This Phase 2 module learns a deliberately small diagonal metric on top of the
frozen v1 embeddings.  It never modifies Gate 1 files and is not consumed by
the generator automatically.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TRAINING_LABELS = frozenset({"related", "different"})
ALL_LABELS = frozenset({"related", "different", "unsure"})


@dataclass(frozen=True)
class FeedbackPair:
    left: str
    right: str
    label: str


class DiagonalMetricAdapter(nn.Module):
    """A low-capacity, positive diagonal reweighting of frozen embeddings."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.clamp(-1.5, 1.5).exp()
        return F.normalize(embeddings * scale, dim=-1)


def load_embedding_map(path: str | Path) -> dict[str, list[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) < 2:
        raise ValueError("Embedding file must contain at least two stable IDs")
    if not all(isinstance(vector, list) for vector in payload.values()):
        raise ValueError("Every embedding must be a JSON list")
    dimensions = {len(vector) for vector in payload.values()}
    if len(dimensions) != 1 or next(iter(dimensions), 0) < 1:
        raise ValueError("Embedding vectors must all have one non-zero dimension")
    matrix = np.asarray(list(payload.values()), dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must form a finite 2D matrix")
    return payload


def load_feedback(path: str | Path, available_ids: set[str]) -> list[FeedbackPair]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ValueError("Feedback file has no answers")
    pairs: list[FeedbackPair] = []
    for key, label in answers.items():
        if label not in ALL_LABELS:
            raise ValueError(f"Unsupported feedback label: {label!r}")
        if not isinstance(key, str) or key.count("__") != 1:
            raise ValueError(f"Malformed feedback pair: {key!r}")
        left, right = key.split("__")
        missing = {stable_id for stable_id in (left, right) if stable_id not in available_ids}
        if missing:
            raise ValueError(f"Feedback references unknown stable IDs: {sorted(missing)}")
        pairs.append(FeedbackPair(left, right, label))
    return pairs


def load_feedback_history(
    paths: str | Path | Iterable[str | Path], available_ids: set[str]
) -> tuple[list[FeedbackPair], list[Path]]:
    """Merge append-only review files; a later answer replaces the same pair."""
    if isinstance(paths, (str, Path)):
        resolved_paths = [Path(paths)]
    else:
        resolved_paths = [Path(path) for path in paths]
    if not resolved_paths:
        raise ValueError("At least one feedback file is required")
    merged: dict[tuple[str, str], FeedbackPair] = {}
    for path in resolved_paths:
        for pair in load_feedback(path, available_ids):
            merged[(pair.left, pair.right)] = pair
    return list(merged.values()), resolved_paths


def _tensor_view(embedding_map: dict[str, list[float]]) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    stable_ids = list(embedding_map)
    vectors = F.normalize(torch.tensor([embedding_map[key] for key in stable_ids], dtype=torch.float32), dim=1)
    return stable_ids, vectors, {stable_id: index for index, stable_id in enumerate(stable_ids)}


def _fit_adapter(
    vectors: torch.Tensor,
    indices: dict[str, int],
    pairs: list[FeedbackPair],
    *,
    steps: int,
    learning_rate: float,
    regularization: float,
    negative_margin: float,
    seed: int,
) -> tuple[DiagonalMetricAdapter, list[float]]:
    training_pairs = [pair for pair in pairs if pair.label in TRAINING_LABELS]
    if not training_pairs or {pair.label for pair in training_pairs} != TRAINING_LABELS:
        raise ValueError("Metric adaptation requires both related and different feedback")
    torch.manual_seed(seed)
    model = DiagonalMetricAdapter(vectors.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    left = torch.tensor([indices[pair.left] for pair in training_pairs])
    right = torch.tensor([indices[pair.right] for pair in training_pairs])
    positive = torch.tensor([pair.label == "related" for pair in training_pairs])
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        adapted = model(vectors)
        similarity = (adapted[left] * adapted[right]).sum(dim=1)
        positive_loss = (1.0 - similarity[positive]).square().mean()
        negative_loss = F.relu(similarity[~positive] - negative_margin).square().mean()
        identity_loss = model.log_scale.square().mean()
        loss = positive_loss + negative_loss + regularization * identity_loss
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


def _pair_scores(
    vectors: torch.Tensor,
    indices: dict[str, int],
    pairs: list[FeedbackPair],
) -> tuple[list[float], list[str]]:
    labeled = [pair for pair in pairs if pair.label in TRAINING_LABELS]
    scores = [float(torch.dot(vectors[indices[pair.left]], vectors[indices[pair.right]])) for pair in labeled]
    return scores, [pair.label for pair in labeled]


def _auc(scores: list[float], labels: list[str]) -> float | None:
    related = [score for score, label in zip(scores, labels) if label == "related"]
    different = [score for score, label in zip(scores, labels) if label == "different"]
    if not related or not different:
        return None
    wins = sum(a > b for a in related for b in different)
    ties = sum(a == b for a in related for b in different)
    return (wins + 0.5 * ties) / (len(related) * len(different))


def summarize_scores(scores: list[float], labels: list[str]) -> dict:
    grouped = defaultdict(list)
    for score, label in zip(scores, labels):
        grouped[label].append(score)
    related_mean = float(np.mean(grouped["related"])) if grouped["related"] else None
    different_mean = float(np.mean(grouped["different"])) if grouped["different"] else None
    return {
        "pairs": len(scores),
        "related_mean_similarity": related_mean,
        "different_mean_similarity": different_mean,
        "separation": related_mean - different_mean if related_mean is not None and different_mean is not None else None,
        "auc": _auc(scores, labels),
    }


def leave_one_anchor_out(
    vectors: torch.Tensor,
    indices: dict[str, int],
    pairs: list[FeedbackPair],
    **fit_kwargs,
) -> dict:
    labeled = [pair for pair in pairs if pair.label in TRAINING_LABELS]
    anchors = sorted({pair.left for pair in labeled})
    held_out_scores: list[float] = []
    held_out_labels: list[str] = []
    folds = []
    for fold_number, anchor in enumerate(anchors):
        training = [pair for pair in labeled if pair.left != anchor]
        held_out = [pair for pair in labeled if pair.left == anchor]
        if {pair.label for pair in training} != TRAINING_LABELS:
            continue
        model, _history = _fit_adapter(vectors, indices, training, seed=fit_kwargs["seed"] + fold_number, **{
            key: value for key, value in fit_kwargs.items() if key != "seed"
        })
        with torch.no_grad():
            adapted = model(vectors)
        scores, labels = _pair_scores(adapted, indices, held_out)
        held_out_scores.extend(scores)
        held_out_labels.extend(labels)
        folds.append({"anchor": anchor, "held_out_pairs": len(scores)})
    result = summarize_scores(held_out_scores, held_out_labels)
    result["folds"] = folds
    return result


def adapt_embeddings_from_feedback(
    embeddings_path: str | Path,
    feedback_path: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    *,
    steps: int = 400,
    learning_rate: float = 0.03,
    regularization: float = 0.8,
    negative_margin: float = 0.45,
    seed: int = 20260831,
) -> dict:
    if steps < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError("Invalid adaptation hyperparameters")
    embedding_map = load_embedding_map(embeddings_path)
    pairs, feedback_paths = load_feedback_history(feedback_path, set(embedding_map))
    stable_ids, vectors, indices = _tensor_view(embedding_map)
    labeled = [pair for pair in pairs if pair.label in TRAINING_LABELS]
    baseline_scores, baseline_labels = _pair_scores(vectors, indices, labeled)
    fit_kwargs = {
        "steps": steps,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "negative_margin": negative_margin,
        "seed": seed,
    }
    cross_validation = leave_one_anchor_out(vectors, indices, labeled, **fit_kwargs)
    model, history = _fit_adapter(vectors, indices, labeled, **fit_kwargs)
    with torch.no_grad():
        adapted = model(vectors)
    final_scores, final_labels = _pair_scores(adapted, indices, labeled)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_output = output_dir / "embeddings_v2.json"
    embeddings_payload = {
        stable_id: [float(value) for value in row]
        for stable_id, row in zip(stable_ids, adapted.numpy())
    }
    embeddings_output.write_text(json.dumps(embeddings_payload, indent=2, sort_keys=True), encoding="utf-8")
    adapter_output = output_dir / "metric_adapter_v2.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_dim": vectors.shape[1],
        "adapter_type": "positive_diagonal",
    }, adapter_output)

    label_counts = Counter(pair.label for pair in pairs)
    manifest = {
        "phase": "Hyponoia representation feedback calibration v2",
        "gate1_integration": False,
        "generator_integration": False,
        "base_embeddings": str(Path(embeddings_path).resolve()),
        "feedback_files": [str(path.resolve()) for path in feedback_paths],
        "embeddings": embeddings_output.name,
        "adapter": adapter_output.name,
        "embedding_count": len(stable_ids),
        "embedding_dim": vectors.shape[1],
        "feedback_counts": dict(sorted(label_counts.items())),
        "training_pairs": len(labeled),
        "uncertain_pairs_ignored_for_training": label_counts["unsure"],
        "hyperparameters": fit_kwargs,
        "baseline_on_labeled_pairs": summarize_scores(baseline_scores, baseline_labels),
        "leave_one_anchor_out": cross_validation,
        "final_fit_on_labeled_pairs": summarize_scores(final_scores, final_labels),
        "final_loss": history[-1],
        "scale_min": float(model.log_scale.detach().clamp(-1.5, 1.5).exp().min()),
        "scale_max": float(model.log_scale.detach().clamp(-1.5, 1.5).exp().max()),
    }
    if not all(math.isfinite(value) for value in (manifest["final_loss"], manifest["scale_min"], manifest["scale_max"])):
        raise ValueError("Non-finite adaptation result")
    (output_dir / "feedback_training_manifest_v2.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--feedback", type=Path, required=True, action="append",
        help="Listening feedback JSON; repeat the option to learn cumulatively",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regularization", type=float, default=0.8)
    parser.add_argument("--negative-margin", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    manifest = adapt_embeddings_from_feedback(
        args.embeddings,
        args.feedback,
        args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        negative_margin=args.negative_margin,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

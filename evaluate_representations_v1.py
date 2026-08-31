"""Offline health evaluation for learned Hyponoia sound-object embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


NUMERIC_FEATURES = (
    "duration", "energy", "brightness", "noise", "attack", "pitch_midi",
    "pitch_confidence", "pitch_motion", "harmonicity", "spectral_bandwidth",
    "zero_crossing_rate", "tail_length", "resonance_strength",
    "transient_score", "ambient_score", "texture_score", "drone_score",
    "gesture_strength", "musicality", "richness",
)
CATEGORICAL_FEATURES = ("register", "gesture_type", "phrase_role", "emotional_function")
REVIEW_FAMILIES = (
    ("synthetic", "Συνθετικός / synth χαρακτήρας", ("synthetic_score",)),
    ("tonal_drone", "Τονικός / drone χαρακτήρας", ("drone_score", "harmonicity", "pitch_confidence")),
    ("transient_gesture", "Κρουστικός / gesture χαρακτήρας", ("transient_score", "gesture_strength")),
    ("texture_ambient", "Υφή / ambient χαρακτήρας", ("texture_score", "ambient_score")),
)


def load_metadata(index_path: str | Path) -> dict[str, dict]:
    records = json.loads(Path(index_path).read_text(encoding="utf-8"))
    metadata = {}
    for record in records:
        for item in record.get("objects", []):
            stable_id = item["stable_id"]
            if stable_id not in metadata:
                metadata[stable_id] = {
                    "recording": record["recording"],
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "duration": item.get("duration"),
                    "features": item.get("features", {}),
                }
    return metadata


def robust_feature_matrix(ids: list[str], metadata: dict[str, dict]) -> np.ndarray:
    rows = []
    for stable_id in ids:
        item = metadata[stable_id]
        features = item["features"]
        rows.append([item.get("duration") if name == "duration" else features.get(name) for name in NUMERIC_FEATURES])
    matrix = np.asarray(rows, dtype=np.float64)
    matrix[~np.isfinite(matrix)] = np.nan
    matrix = matrix[:, np.isfinite(matrix).any(axis=0)]
    if matrix.shape[1] == 0:
        raise ValueError("No numeric acoustic features available for evaluation")
    medians = np.nanmedian(matrix, axis=0)
    missing = np.where(np.isnan(matrix))
    matrix[missing] = medians[missing[1]]
    q25, q75 = np.percentile(matrix, [25, 75], axis=0)
    scale = q75 - q25
    scale[scale < 1e-9] = 1.0
    return (matrix - medians) / scale


def select_balanced_review_anchors(
    ids: list[str], metadata: dict[str, dict], limit: int = 20
) -> list[tuple[int, str, str, float]]:
    """Round-robin high-scoring anchors across four audible sound families."""
    if limit < 1:
        return []
    candidates: dict[str, list[tuple[float, str, int]]] = {}
    labels = {}
    for key, label, feature_names in REVIEW_FAMILIES:
        labels[key] = label
        rows = []
        for index, stable_id in enumerate(ids):
            features = metadata[stable_id]["features"]
            values = [float(features.get(name) or 0.0) for name in feature_names]
            rows.append((max(values), stable_id, index))
        candidates[key] = sorted(rows, key=lambda row: (-row[0], row[1]))

    selected: list[tuple[int, str, str, float]] = []
    used_ids: set[str] = set()
    used_recordings: set[str] = set()
    while len(selected) < min(limit, len(ids)):
        added = False
        for key, _label, _feature_names in REVIEW_FAMILIES:
            choice = next((
                row for row in candidates[key]
                if row[1] not in used_ids and metadata[row[1]]["recording"] not in used_recordings
            ), None)
            if choice is None:
                choice = next((row for row in candidates[key] if row[1] not in used_ids), None)
            if choice is None:
                continue
            score, stable_id, index = choice
            selected.append((index, key, labels[key], score))
            used_ids.add(stable_id)
            used_recordings.add(metadata[stable_id]["recording"])
            added = True
            if len(selected) >= min(limit, len(ids)):
                break
        if not added:
            break
    return selected


def evaluate_embeddings(
    index_path: str | Path,
    embeddings_path: str | Path,
    output_path: str | Path,
    *,
    neighbors: int = 5,
    sampled_pairs: int = 50_000,
    seed: int = 20260831,
    exclude_same_recording: bool = True,
) -> dict:
    embedding_map = json.loads(Path(embeddings_path).read_text(encoding="utf-8"))
    metadata = load_metadata(index_path)
    ids = [stable_id for stable_id in embedding_map if stable_id in metadata]
    if len(ids) <= neighbors:
        raise ValueError("Not enough matched embeddings for neighbor evaluation")
    embeddings = np.asarray([embedding_map[key] for key in ids], dtype=np.float64)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    recordings = np.asarray([metadata[key]["recording"] for key in ids], dtype=object)
    similarity = embeddings @ embeddings.T
    if exclude_same_recording:
        similarity[recordings[:, None] == recordings[None, :]] = -np.inf
    else:
        np.fill_diagonal(similarity, -np.inf)
    if int(np.isfinite(similarity).sum(axis=1).min()) < neighbors:
        raise ValueError("Not enough eligible cross-recording neighbours")
    neighbor_indices = np.argpartition(-similarity, neighbors, axis=1)[:, :neighbors]
    neighbor_indices = np.take_along_axis(
        neighbor_indices,
        np.argsort(-np.take_along_axis(similarity, neighbor_indices, axis=1), axis=1),
        axis=1,
    )

    numeric = robust_feature_matrix(ids, metadata)
    rng = np.random.default_rng(seed)
    left_parts, right_parts, pair_count = [], [], 0
    while pair_count < sampled_pairs:
        left = rng.integers(0, len(ids), sampled_pairs)
        right = rng.integers(0, len(ids), sampled_pairs)
        eligible = left != right
        if exclude_same_recording:
            eligible &= recordings[left] != recordings[right]
        left_parts.append(left[eligible])
        right_parts.append(right[eligible])
        pair_count += int(eligible.sum())
    left = np.concatenate(left_parts)[:sampled_pairs]
    right = np.concatenate(right_parts)[:sampled_pairs]
    embedding_similarity = np.sum(embeddings[left] * embeddings[right], axis=1)
    acoustic_distance = np.linalg.norm(numeric[left] - numeric[right], axis=1)
    correlation = float(spearmanr(embedding_similarity, -acoustic_distance).statistic)

    neighbor_distance = float(np.linalg.norm(numeric[:, None, :] - numeric[neighbor_indices], axis=2).mean())
    random_distance = float(acoustic_distance.mean())
    labels = {}
    for name in CATEGORICAL_FEATURES:
        values = np.asarray([metadata[key]["features"].get(name) for key in ids], dtype=object)
        observed = float((values[:, None] == values[neighbor_indices]).mean())
        baseline = float((values[left] == values[right]).mean())
        labels[name] = {
            "neighbor_agreement": observed,
            "random_baseline": baseline,
            "lift": observed / baseline if baseline else None,
        }

    examples = []
    for index, category_key, category, category_score in select_balanced_review_anchors(ids, metadata):
        anchor_id = ids[index]
        examples.append({
            "anchor": {
                "stable_id": anchor_id,
                "review_category_key": category_key,
                "review_category": category,
                "review_category_score": category_score,
                **metadata[anchor_id],
            },
            "neighbors": [
                {
                    "stable_id": ids[j],
                    "cosine_similarity": float(similarity[index, j]),
                    **metadata[ids[j]],
                }
                for j in neighbor_indices[index]
            ],
        })

    result = {
        "neighbor_policy": "cross_recording_only" if exclude_same_recording else "all_recordings",
        "matched_embeddings": len(ids),
        "embedding_dim": int(embeddings.shape[1]),
        "neighbors": neighbors,
        "sampled_pairs": int(len(left)),
        "embedding_acoustic_spearman": correlation,
        "mean_neighbor_acoustic_distance": neighbor_distance,
        "mean_random_acoustic_distance": random_distance,
        "neighbor_distance_ratio": neighbor_distance / random_distance,
        "categorical_neighbor_agreement": labels,
        "review_sampling_policy": "balanced_four_sound_families",
        "example_neighbors": examples,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--sampled-pairs", type=int, default=50_000)
    parser.add_argument("--allow-same-recording", action="store_true")
    args = parser.parse_args()
    result = evaluate_embeddings(
        args.index, args.embeddings, args.output,
        neighbors=args.neighbors, sampled_pairs=args.sampled_pairs,
        exclude_same_recording=not args.allow_same_recording,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "example_neighbors"}, indent=2))


if __name__ == "__main__":
    main()

"""Reproducible contrastive training for Hyponoia Phase 2.

Outputs are intentionally separate from Gate 1 and are never consumed by the
generator automatically.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from representation_learning_v1 import (
    EMBEDDING_DIM,
    ContrastiveEncoder,
    SoundObject,
    augment,
    contrastive_loss,
    export_embeddings,
    load_sound_objects,
    log_mel,
    read_fragment,
    seed_everything,
)


def fixed_frames(x: torch.Tensor, frames: int) -> torch.Tensor:
    """Center crop or zero-pad a log-mel representation."""
    current = x.shape[-1]
    if current > frames:
        start = (current - frames) // 2
        return x[..., start : start + frames]
    if current < frames:
        return torch.nn.functional.pad(x, (0, frames - current))
    return x


class ContrastiveSoundDataset(Dataset):
    def __init__(self, objects: list[SoundObject], frames: int = 128, cache_dir: str | Path | None = None):
        self.objects = objects
        self.frames = frames
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.objects)

    def representation(self, item: SoundObject) -> torch.Tensor:
        cache_path = self.cache_dir / f"{item.stable_id}.npy" if self.cache_dir else None
        if cache_path and cache_path.exists():
            return torch.from_numpy(np.load(cache_path, allow_pickle=False))
        x = fixed_frames(log_mel(read_fragment(item)), self.frames)
        if cache_path:
            np.save(cache_path, x.numpy(), allow_pickle=False)
        return x

    def __getitem__(self, index: int):
        item = self.objects[index]
        x = self.representation(item)
        return item.stable_id, augment(x), augment(x)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_representation_model(
    index_path: str | Path,
    memory_dir: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    frames: int = 128,
    seed: int = 20260831,
    max_objects: int | None = None,
    device: str = "auto",
) -> dict:
    """Train and export an isolated checkpoint, embeddings, and run manifest."""
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs must be >= 1 and batch_size must be >= 2")
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = load_sound_objects(index_path, memory_dir)
    if max_objects is not None:
        objects = objects[:max_objects]
    if len(objects) < 2:
        raise ValueError("At least two sound objects are required")

    selected_device = choose_device(device)
    dataset = ContrastiveSoundDataset(objects, frames=frames, cache_dir=output_dir / "feature_cache")
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=loader_generator)
    model = ContrastiveEncoder(EMBEDDING_DIM).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    started = time.time()

    model.train()
    for _epoch in range(epochs):
        losses = []
        for _ids, first, second in loader:
            first, second = first.to(selected_device), second.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(model(first), model(second))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))

    checkpoint_path = output_dir / "encoder_v1.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "frames": frames,
            "seed": seed,
            "epochs": epochs,
            "loss_history": history,
        },
        checkpoint_path,
    )

    # Stable IDs are content-derived; identical IDs intentionally share one vector.
    unique_objects = list({item.stable_id: item for item in objects}.values())
    model.eval()
    vectors = []
    with torch.no_grad():
        for item in unique_objects:
            x = dataset.representation(item).unsqueeze(0)
            vectors.append(model(x).squeeze(0))
    embeddings = torch.stack(vectors)
    embeddings_path = output_dir / "embeddings_v1.json"
    export_embeddings([item.stable_id for item in unique_objects], embeddings, embeddings_path)

    manifest = {
        "phase": "Hyponoia representation learning v1",
        "gate1_integration": False,
        "index_path": str(Path(index_path).resolve()),
        "memory_dir": str(Path(memory_dir).resolve()),
        "objects_seen": len(objects),
        "unique_stable_ids": len(unique_objects),
        "embedding_dim": EMBEDDING_DIM,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "frames": frames,
        "seed": seed,
        "device": str(selected_device),
        "loss_history": history,
        "elapsed_seconds": round(time.time() - started, 3),
        "checkpoint": checkpoint_path.name,
        "embeddings": embeddings_path.name,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    manifest = train_representation_model(
        args.dataset_root / "memory_index_v3.json",
        args.dataset_root / "alpha_memory",
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        frames=args.frames,
        seed=args.seed,
        max_objects=args.max_objects,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


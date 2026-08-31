# Hyponoia Phase 2 — Representation Learning v1

This is an experimental layer isolated from the Gate 1 baseline. It reads the
existing stable sound-object IDs and audio slices, creates normalized log-mel
representations, and defines a small convolutional encoder for self-supervised
contrastive learning. Its output contract is a separate JSON mapping from stable
ID to a 32-dimensional embedding.

## Current boundary

- No Gate 1 generator, Critic, feedback, OSC, or Max/MSP file is modified.
- Importing the module performs no training and changes no project data.
- The encoder is not yet connected to the generator.
- Full-library training is intentionally deferred until this integration and its
  tests are reviewed.

## Environment and tests

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-representation.txt
.venv/bin/python -m pytest
```

## Training checkpoint

`representation_training_v1.py` adds a reproducible, isolated contrastive
training command. It writes its feature cache, encoder checkpoint, embeddings,
and manifest only beneath the requested output directory. The generated
embeddings are still **not** connected to the Gate 1 generator.

```bash
.venv/bin/python representation_training_v1.py \
  --dataset-root /path/to/private/project \
  --output-dir /path/to/private/phase2_run \
  --epochs 10 --batch-size 32
```

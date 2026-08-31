# Hyponoia Phase 2 — Representation Learning

This is an experimental layer isolated from the Gate 1 baseline. It reads the
existing stable sound-object IDs and audio slices, creates normalized log-mel
representations, and defines a small convolutional encoder for self-supervised
contrastive learning. Its output contract is a separate JSON mapping from stable
ID to a 32-dimensional embedding.

## Current boundary

- No Gate 1 generator, Critic, feedback, OSC, or Max/MSP file is modified.
- Importing the module performs no training and changes no project data.
- The encoder is not yet connected to the generator.
- Learned embeddings remain opt-in experimental artifacts until listening
  acceptance; they are never consumed by Gate 1 automatically.

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

Evaluate the learned geometry against the existing acoustic descriptors:

```bash
.venv/bin/python evaluate_representations_v1.py \
  --index /path/to/private/project/memory_index_v3.json \
  --embeddings /path/to/private/phase2_run/embeddings_v1.json \
  --output /path/to/private/phase2_run/evaluation_v1.json
```

Build a local listening review with extracted private clips (never committed):

```bash
.venv/bin/python build_listening_review_v1.py \
  --index /path/to/private/project/memory_index_v3.json \
  --memory-dir /path/to/private/project/alpha_memory \
  --evaluation /path/to/private/phase2_run/evaluation_v1.json \
  --output-dir /path/to/private/listening_review
```

## Human calibration v2

`representation_feedback_v2.py` uses a completed listening-review JSON to learn
a small positive diagonal metric on top of the frozen v1 embeddings. `related`
pairs are pulled closer, `different` pairs are separated, and `unsure` answers
are retained in the report but deliberately excluded from training. The adapter
is low-capacity and regularized to limit drift from the self-supervised model.

The command performs leave-one-anchor-out validation before fitting the final
adapter. It writes a new checkpoint, embeddings, and manifest beneath a new
output directory. It does not overwrite v1 artifacts and does not connect the
result to the generator.

```bash
.venv/bin/python representation_feedback_v2.py \
  --embeddings /path/to/private/phase2_run/embeddings_v1.json \
  --feedback /path/to/hyponoia_listening_feedback.json \
  --output-dir /path/to/private/phase2_feedback_v2
```

Repeat `--feedback /path/to/another_review.json` to learn cumulatively from
later reviews. If the same pair is reviewed again, the newest answer wins.

Re-run the cross-recording evaluator and listening review against
`embeddings_v2.json` before any generator integration.

## Guarded generator assist

`representation_assist_v1.py` provides the first opt-in bridge to the existing
generator. The committed `representation_config.json` keeps `mode` set to
`off`, which is exactly neutral and preserves Gate 1 selection. In `assist`
mode, the learned geometry adds only a bounded continuity factor between two
different consecutive sound objects; role, harmony, human-feedback weights,
sample exploration, and anti-repetition controls remain authoritative.

The first accepted policy applies assist only inside the same validated role
family (`gesture`, `texture`, `impact`, or `noise`). `resonance`/tonal/drone
continuity stays neutral until a later listening set demonstrates reliable
generalization for that family.

The bridge fails closed: missing or malformed configuration or embeddings
produce a neutral factor instead of interrupting a render. Every render report
records whether the assist was active, its strength, and the embedding count.

# Hyponoia Phase 2 — Representation Learning

This is an experimental layer isolated from the Gate 1 baseline. It reads the
existing stable sound-object IDs and audio slices, creates normalized log-mel
representations, and defines a small convolutional encoder for self-supervised
contrastive learning. Its output contract is a separate JSON mapping from stable
ID to a 32-dimensional embedding.

## Current boundary

- The Gate 1 release remains unchanged on `main`; all Phase 2 work stays on the
  `phase2-representation-learning` branch.
- Importing the module performs no training and changes no project data.
- Learned embeddings are connected only through the opt-in guarded assist and
  are never consumed by Gate 1 automatically.
- Composition feedback now affects bounded generator decisions and mix controls
  only for the rated D1, D3 or D5 profile.
- The first controlled D1 listening loop was accepted as a temporary baseline
  on 1 September 2026. It does not make D3 or D5 inherit D1 preferences.

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

## Change your sound library

On macOS, double-click `Open Hyponoia Library.command`.

1. Choose any folder containing WAV files.
2. Press **Preview changes**.
3. Check the counts for added, changed, removed, renamed and unchanged files.
4. Press **Update library**.

Hyponoia recommends about 100 source recordings for a useful personal palette,
but this is not a hard minimum. You may start with fewer and add, replace,
rename or remove WAV files whenever you want. Source WAV files are never moved,
deleted or rewritten. Unchanged analyses are reused; new or modified recordings
are analysed, and the previous generated index is backed up before activation.

The selected folder is saved locally in `hyponoia_user_config.json`. The next
composition automatically uses that library. Personal audio, the local path
configuration and backups are ignored by Git and are not uploaded to GitHub.

The same update is available from Terminal when needed:

```bash
.venv/bin/python update_library_v1.py \
  --library /path/to/Hyponoia\ Library
```

## Unified composition feedback foundation

`composition_feedback_v1.py` accepts the seven 1–5 listener ratings used by
the Deep Learning listening workflow plus `more`, `less` and free-comment
text. It supports the first bounded Greek and English musical intents and
routes every update only to the rated D1, D3 or D5 profile. Ratings can request
musicality, coherence, smooth transitions, controlled variety, synth presence
and development. Text can additionally request clearer layers and less
low-frequency masking. The Critic remains diagnostic and cannot override an
explicit human rating.

These controls are now connected to audible, bounded decisions: musical
selection, density, development, synth presence, low-frequency masking, layer
clarity, event activity and role-aware releases. The accepted D1-D listening
baseline scored 3.5/5; it is evidence for the next training cycle, not a claim
that the preference model is finished.

The next candidate pass also recognises explicit requests for arpeggios, a
clearer/less muddy mix and different sustained layers. Arpeggios are generated
as bounded phrases, not as one continuous bed. Long texture/resonance sources
rotate between D1/D3/D5 families and are windowed only after explicit diversity
feedback. The shared reverb keeps most bass and low-mid energy dry. These
changes remain candidates until human listening accepts them; D1-D stays the
protected reference.

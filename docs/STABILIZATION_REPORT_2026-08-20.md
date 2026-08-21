# Hyponoia v2 Stabilisation Report — 20 August 2026

## Outcome

The public Python baseline was frozen at `363bb3d08410f0c2daffce9c42c6ee2b9e84bb93` and repaired on the local branch `codex/stabilization-2026-08-20`. The physiological/Max architecture was preserved; no Deep Learning component was added.

## Corrected baseline

- The full audio path now targets 48 kHz.
- Recordings and objects receive content-derived stable IDs, while positional legacy IDs remain available for safe migration.
- The original sample-learning profile is migrated without silently deleting unresolved evidence.
- Human ratings are the primary learning signal. Human-minus-Critic residuals remain separate diagnostic evidence.
- Low quality ratings now request a musically corrective control direction; they no longer use Critic disagreement as a blind parameter sign.
- Positive/negative human evidence respectively reinforces/weakens the samples used in the render.
- Sample credit is differentiated by selection count, exposure time, gain, role, and uncertainty.
- English comments use a deterministic, bounded MVP phrase schema and produce logged before/after controls.
- Critic v2.1 removes hard ceiling behaviour and records raw discontinuity/bloom components plus count-, exposure-, and role-based usage metrics.
- Critic calibration can report variance, MAE, signed error, and rank correlation against human ratings.
- Source recordings are cached during generation.
- OSC remains responsive while rendering, uses the active Python executable, and reports busy state.
- Feedback events use unique IDs and atomic JSON writes.
- Palette families are deterministic across Python processes.
- The profile builder accepts an arbitrary reference file.
- Runtime and test dependencies are pinned.

## Validation

Command: `.venv/bin/pytest -q`

Initial result: **13 passed** on 20 August 2026. After the gentle-transition and balanced-material refinements, the expanded suite reports **16 passed**.

Covered behaviours: 48 kHz resampling, stable IDs, migration, deterministic phrase actions, correct rating direction, positive/negative sample learning, differentiated credit, profile generation, memory inventory, recording cache, Critic score variation, ordered D-level activity, brightness-aware envelopes, and balanced material-plan limits.

The private Mac baseline successfully indexed 103 recordings into 2,823 sound objects and completed multiple real three-minute D1/D5 generation, Critic, human-rating, controlled-comment, and sample-credit cycles. Personal audio and learning data remain outside the repository.

## Data availability boundary

The public repository intentionally contains no personal WAV library, `memory_index_v3.json`, historical learning profiles, render reports, Critic reports, or feedback events. The test suite therefore uses generated audio fixtures. Before Gate 1 can be declared fully passed, the same suite plus one controlled positive/negative feedback cycle must run against the private 103-file baseline and its real historical profiles.

## Next gate

21 August 2026: controlled baseline validation and Critic v2.1 calibration. Deep Learning remains blocked until the real-data Gate 1 checks pass.

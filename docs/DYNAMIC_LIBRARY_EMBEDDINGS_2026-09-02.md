# Dynamic library embeddings — 2 September 2026

## Accepted update contract

The Update Library flow now treats musical memory and learned representations
as one guarded update when representation assist is active.

- Source WAV files are read-only.
- Stable IDs whose audio content remains present keep their existing vectors.
- New stable IDs are embedded with the frozen contrastive encoder.
- The current positive diagonal metric adapter is applied to new vectors when
  the active configuration provides one.
- Stable IDs no longer present in the active library are removed.
- Full encoder or adapter retraining is never started by a library update.
- Invalid/missing required artifacts stop the update before the working memory
  or embedding map is replaced.
- Generated indexes, configurations and changed embedding maps are backed up.

## Configuration

`representation_config.json` keeps the feature opt-in. In `assist` mode it may
provide paths relative to the configuration file or absolute local paths:

```json
{
  "mode": "assist",
  "strength": 0.35,
  "embeddings_path": "phase2_artifacts/embeddings_v2.json",
  "encoder_path": "phase2_artifacts/encoder_v1.pt",
  "metric_adapter_path": "phase2_artifacts/metric_adapter_v2.pt",
  "refresh_policy": "frozen_encoder_incremental"
}
```

The committed configuration remains `off` because trained artifacts and user
audio are not part of the source repository. The final user package must either
ship a validated general encoder or guide the user through one explicit initial
model setup. After that one setup, ordinary library changes use the incremental
path described above.

## Verification boundary

Automated tests cover reuse, addition, removal, relative-path configuration,
optional metric adaptation and failure without a configured encoder. A separate
real-WAV integration check uses the existing trained Phase 2 encoder and adapter
on an isolated temporary project; it never modifies the private training run.

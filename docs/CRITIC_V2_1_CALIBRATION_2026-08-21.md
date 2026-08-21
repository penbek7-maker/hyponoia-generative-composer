# Critic v2.1 calibration note — 21 August 2026

## Evidence before the v2.1 formula change

Nine real Hyponoia renders had matching Critic and human ratings for all six
dimensions. The principal failure was coherence compression.

| Dimension | Critic mean | Human mean | Critic variance | Human variance | MAE | Rank correlation |
|---|---:|---:|---:|---:|---:|---:|
| Musicality | 74.03 | 66.67 | 2.67 | 116.67 | 8.45 | 0.45 |
| Coherence | 93.72 | 73.11 | 0.26 | 36.32 | 20.61 | 0.04 |
| Richness | 54.51 | 62.22 | 18.26 | 145.06 | 11.93 | 0.11 |
| Transitions | 60.54 | 65.00 | 207.93 | 150.00 | 10.27 | 0.68 |
| Bloom quality | 78.50 | 68.44 | 248.71 | 188.25 | 17.81 | 0.47 |
| Overall | 72.74 | 65.22 | 15.04 | 131.51 | 11.35 | -0.12 |

These figures are a small-sample diagnostic baseline, not evidence of a
generalisable perceptual model.

## v2.1 changes

- Coherence combines local continuity, macro-arc smoothness, developmental
  trajectory and material balance.
- Musicality and richness now include development and render-report evidence,
  rather than relying only on frame-level acoustic descriptors.
- Structural novelty compares role and section proportions only with earlier
  renders at the same D-level.
- Low historical structural novelty produces an explicit comment and a small,
  bounded overall penalty.
- The raw features and component values remain stored in every report.

## Generator response

Three consecutive D5 reports used almost completely different stable sample
IDs, yet their role and section distributions remained close. The perceived
similarity was therefore treated as a form/orchestration issue. The generator
now selects one of three D5 formal energy arcs and lets explicit activity,
musicality and synthetic-material controls affect D5 gesture selection and
formal contrast. D1 and D3 paths are unchanged.

## Remaining acceptance condition

The implementation is not release-ready until the new Critic is run on real
audio, paired with a new independent human rating, and followed by an updated
dimension-specific calibration report. Max/MSP path routing and live OSC remain
separate blockers for the overall GATE 1 decision.

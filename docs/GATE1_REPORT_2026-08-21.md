# Hyponoia GATE 1 Validation — 21 August 2026

## Decision

**GATE 1: PASS.** The automated and private-data baselines are stable, and the
final live Max/Python path and OSC round trip passed on 21 August 2026. No Deep
Learning is claimed in this release.

## Automated baseline

`.venv/bin/python -m pytest -q` reports **39 passed, 0 failed**. The three
warnings come from deprecated standard-library imports inside `audioread`; they
do not represent Hyponoia test failures.

The suite now proves the complete controlled learning chain: human feedback
changes the signed learned value of used samples, and repeated positive or
negative evidence changes the exact weighted-selection probability in the
expected direction. Credit remains differentiated by selection count, exposure,
gain, role and uncertainty.

## D1/D3/D5 feedback isolation

Numeric ratings update the shared aesthetic profile. Text comments are routed
to a separate D1, D3 or D5 profile, inferred from the current render or an
explicit level prefix. The generator combines the shared controls with only the
active level profile. Only explicit global language changes all three levels.

Acceptance tests apply three different comments to D1, D3 and D5 and verify
that the other profiles remain unchanged. A global comment then verifies that
all three profiles change together.

Version 2.5.1 additionally verifies that `more musical`, `more rhythmic`, `more
synthetic` and `greater bloom` become four bounded D5-only actions. A correction
event can be appended without duplicating the original sample credit when an
older interpreter handled only part of a stored comment.

The private read-only validation now passes with **103 recordings**, **2,823
sound objects**, stable IDs, 48 kHz records, all three D-level render reports,
the controlled signed-feedback cycle, level-specific comments and unique
append-only feedback IDs.

The level-isolation check compares each controlled comment with the current
learned profile rather than assuming untouched `1.0` defaults. This allows the
validator to remain read-only and correct after real D-level learning has
already accumulated.

## Critic v2.1

Nine paired real-rating events confirmed the pre-v2.1 failure. Coherence had a
Critic mean of 93.72 against a human mean of 73.11, variance 0.26 against 36.32,
MAE 20.61 and rank correlation 0.04. Overall rank correlation was -0.12. The
Critic could therefore rank a rejected render above a preferred render.

Critic v2.1 replaces the smoothness-only coherence formula with local
continuity, macro-form arc smoothness, developmental trajectory and material
balance. It also reports historical structural novelty from role and section
proportions. The three current D5 renders shared almost no sample IDs, but had
very similar role/form distributions; this explains why changing samples alone
did not solve their perceived similarity.

The first D5 update added bounded energy/musicality drive, stronger selection of
musical synthetic gestures and three alternative formal energy arcs. A private
103-recording render proved that event count alone was insufficient: despite 184
layers and 62 gestures, the listener still perceived the result as D1 because
the stretches, emergence envelopes, delay spacing and ambient bed remained slow.

Generator revision `2026-08-21-d5-temporal-energy-2` therefore maps D5 activity
learning to actual temporal behaviour. It shortens D5-only time stretches,
envelopes and delay spacing, reduces the slow ambient wash as activity rises,
and adds smooth role-aware internal motion to develop the selected material.
It also records event-rate, foreground-rate, duration and quick-succession
metrics in each render report. D1 and D3 retain their established behaviour.

Private listening then showed that v2.2 over-corrected: it produced more audible
events, but some phrases became too short and disconnected. Two listener-chosen
references were analysed for pulse, local spectral continuity, abrupt energy
drops, loudness and macro-dynamics. The preferred Hyponoia reference measured
approximately -17.4 LUFS and only 0.33 abrupt energy/spectral drops per minute;
the two references placed their pulse estimates between 123 and 129 BPM.

Revision `2026-08-21-d5-reference-continuity-3` therefore bounds layer-count
growth, expresses activity through a soft shared pulse and internal motion,
preserves longer transformed phrases, overlaps related role lanes and applies a
final cosine attack/release guard after every other effect. The private audio is
not distributed; only non-identifying aggregate targets are stored in reports.

The calibration tool now computes each dimension only when that same dimension
has a human rating. Reanalysis of four available current pairs produced
coherence variance 34.084319, MAE 2.7625 and rank correlation 0.774597; the
v2.1 calibration update passed. This remains a small research sample rather
than a general perceptual-validity claim.

## Aesthetic bridge v2.5.1

The listener-preferred 20 August D5 is the primary aesthetic reference. The
bridge restores its open palette, elastic timing, long phrases and ambient body
while keeping bounded D5 energy, synthetic selection and the final continuity
guard. The accepted-direction private render used 142 selections from 32 sound
objects and 12 source recordings. The concentration of repeated objects is
documented and intentionally deferred: it will be retested with the enlarged
library and additional feedback rather than hidden by a premature hard cap.

Selection remains open to unfamiliar material through feature-based weighted
choice, recording-history penalties, within-render usage penalties and an
underused-object exploration bonus.

## Max and physiological regression

The Python OSC receiver is non-blocking and preserves the harmony state. The
Max/Python integration now uses the following verified routes:

- D1, D3 and D5 messages are sent to port 7401.
- Python responses return to Max on port 7402.
- The physiological stream remains on port 5001 with `/bands`, `/bands_z`,
  `/state` and `/state_name` routes.

The active Max patch now routes `/generator/path` into `preload 1 $1` and waits
for `/generator/ready` before playback. The receiver guarantees that the path is
sent first, preventing the previous hard-coded-path mismatch and local preload
race.

A live D5 request from Max reached Python on port 7401, rendered 143 layers,
updated `output/current.wav`, returned its absolute path on port 7402 and then
played the new result automatically in Max. Port 5001 remained bound for the
physiological stream throughout the test. The live Max regression therefore
passed.

## Remaining integration work

1. Re-run sample-distribution analysis after the library is enlarged and more
   human feedback has accumulated.
2. Begin the separately scoped representation-learning phase; do not describe
   the current explainable learner as Deep Learning.

The full machine-readable evidence is in
`reports/gate1_validation_2026-08-21.json`.

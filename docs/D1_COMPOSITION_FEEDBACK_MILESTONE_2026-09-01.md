# D1 composition-feedback milestone — 1 September 2026

## Outcome

The first complete D1 composition-feedback loop is accepted as a temporary
listening baseline. Gate 1 remains unchanged on `main`; this milestone belongs
only to `phase2-representation-learning`.

## What became audible

- 1–5 ratings and Greek/English text persist as bounded D-level controls.
- D1 feedback does not leak into D3 or D5.
- Musicality, coherence, richness, exploration, repetition, development and
  synthetic-material preferences affect material and event planning.
- Low-frequency control reduces masking instead of deleting bass globally.
- Layer clarity opens stereo placement and reduces excessive reverb masking.
- Explicit smoothness feedback adds role-aware releases, damped tails and
  related-layer overlap.
- Every render report records the active controls, representation-assist state
  and the derived audio-feedback factors.

## Controlled listening progression

| Render | Listener outcome |
| --- | --- |
| D1-A smoke | 2.5/5 overall; not accepted |
| D1-B full | 2/5 overall; not accepted |
| D1-C | 3/5; synth and energy improved; exits still abrupt |
| D1-D | 3.5/5; synth and energy preserved; exits improved; accepted temporarily |

D1-C and D1-D used the same seed, six-recording material plan, 84 events and
the same D1 learning profile. D1-D changed the structural release policy only.
Average event duration increased from 14.11 to 17.60 seconds, and average
foreground duration from 1.84 to 3.22 seconds.

## Validation

- 69 tests passed on Python 3.12.10.
- Full render: 180 seconds, stereo, 48 kHz, finite samples.
- Representation assist: active, strength 0.35, 2,566 calibrated embeddings.
- Personal WAVs, embeddings, feedback profiles and render artifacts remain
  local and are not committed to GitHub.

## Next learning step

The accepted D1-D is a reference label, not a final model. Future renders must
be compared against it. More diverse ratings and libraries can improve the
preference model, but improvements must continue to pass listening evaluation;
training alone is never treated as proof of artistic improvement.

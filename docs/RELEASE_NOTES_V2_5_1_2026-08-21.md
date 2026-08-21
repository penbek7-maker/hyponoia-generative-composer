# Hyponoia v2.5.1 — Gate 1 code baseline

This release candidate consolidates the stable 48 kHz memory/generation path,
Critic v2.1, explainable human-guided learning and non-blocking Max OSC bridge.

## Learning

- Human scores are the primary target; Critic values remain auxiliary evidence.
- Numeric ratings update shared aesthetic controls.
- Text comments update only the inferred or explicit D1, D3 or D5 profile.
- `more musical`, `more rhythmic`, `more synthetic` and `greater bloom` now
  produce four separate, bounded control updates.
- Sample credit is signed and weighted by role, exposure, gain and selection
  count, with append-only feedback IDs and atomic persistence.

## Generator

- Sound objects are selected from analysed acoustic/musical features and learned
  sample values, not from filenames alone.
- Each render uses a logged random seed for controlled stochastic variation.
- Underused objects receive an exploration advantage; frequently used source
  recordings receive history and within-render penalties.
- The v2.5 aesthetic bridge returns to the listener-preferred open palette,
  elastic timing, long phrases and ambient body while retaining the final
  continuity guard and gentle D5 rhythmic/synthetic motion.
- No hard repetition cap is imposed. Repetition distribution will be evaluated
  again with the enlarged sound library and additional human feedback.

## Validation

- 39 automated tests pass; three warnings come from upstream `audioread`
  deprecations.
- The private baseline previously passed with 103 recordings and 2,823 sound
  objects. Private WAV, memory and learning files remain excluded from Git.
- The live Max/Python round trip passed: Max requested D5, Python rendered and
  updated `output/current.wav`, and Max received the dynamic path before the
  ready trigger and played the new result automatically.
- The physiological OSC input remained bound to port 5001 during the test.

Deep learning is not claimed in this release. The current implementation is an
explainable adaptive baseline for the next representation-learning phase.

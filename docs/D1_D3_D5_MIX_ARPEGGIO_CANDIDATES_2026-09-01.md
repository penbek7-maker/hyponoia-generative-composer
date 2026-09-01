# D1/D3/D5 mix and arpeggio candidates — 1 September 2026

## Listener evidence

The first D3-A and D5-A reviews requested less mix blur, fewer repeated long
layers, more audible synthesizer material, real arpeggios and greater
musicality. D3 and D5 received separate bounded profiles. D1-D remains the
accepted temporary baseline.

## Candidate implementation

- Added explicit arpeggio and long-layer-diversity controls.
- Greek/English feedback now recognises arpeggios, muddy/blurred mixes and
  repeated sustained layers.
- Added a phrase-based, level-specific arpeggio layer. It is silent at neutral
  weight and bounded after an explicit listener request.
- Added deterministic D-level sustained-source rotation, within-render reuse
  penalties and level-specific windows for very long texture/resonance objects.
- Reduced low and low-mid energy in the shared reverb return and lowered the
  candidate room amount.
- Kept all representation assist guards and the Gate 1 main baseline intact.

## Validation

- 76 tests passed on Python 3.12.
- D1-E experimental attempts did not beat the accepted D1-D diagnostic
  reference, so D1-D remains protected.
- D3-C and D5-C are 180-second, stereo, 48 kHz listening candidates with active
  representation assist (2,566 calibrated embeddings, strength 0.35).
- Human listening remains authoritative; Critic scores are diagnostic only.

## Next decision

The listener accepted the corrected direction as much better overall. The next
D5 requests a small gradual increase in synth presence and a little more
palette variety; those controls were stored without forcing another render.
The next implementation milestone is the user-controlled Update Library flow.

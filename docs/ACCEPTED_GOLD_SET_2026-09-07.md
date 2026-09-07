# Accepted Hyponoia gold set

The aesthetic search is frozen. These are the only positive composition-level
anchors for D1, D3 and D5. Later experimental renders must not be added as
positive evidence without a new explicit listening decision.

| Level | Accepted file | SHA-256 | Role |
| --- | --- | --- | --- |
| D1 | `Hyponoia_D1_SynthGranular_v3.wav` | `a86ea1cf96217e0e64b3d01153d3a9aeb6020f892640eaebb7078a20cec39e0e` | positive preference anchor |
| D3 | `Hyponoia_D3_SpectralBackground_APPROVED.wav` | `5e9e47f4789cfdb6eb37697a69ed42a410d2d53ee322743f14b9275b3b1588f6` | positive preference anchor; `spectral_halo` background |
| D5 | `Hyponoia_D5_MusicalForeground_v17.wav` | `4847697081f5e66d353a227047bce0221a3ee032571e7805f06818cc7e8b67a5` | positive preference anchor |

The local preference model is trained from the corresponding render reports,
not by copying the waveforms. It uses the learned 32-dimensional sound-object
embeddings together with whole-render structural summaries. Rejected renders
remain contrast evidence.

This is a bounded assistive preference layer, not an end-to-end claim that
three compositions are enough to generalise every musical decision.

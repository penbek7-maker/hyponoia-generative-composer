# External Listener Feedback

## Review 01 — 11 September 2026

The first external review identified four priorities:

1. A narrow, tinnitus-like tone recurred too consistently.
2. A similar rising/falling pitch gesture appeared around 75% of each composition.
3. Pitch control was not sufficiently clear to the listener.
4. Future learning should draw on a broader, documented acousmatic-composition corpus.

## Release response

- Replaced the fixed pure-tone spectral bloom with a source-derived bloom made
  from the active library and the current composition.
- Removed the compulsory 7.2/9.3/11.8/13.5 kHz oscillator layer and its pitch
  drift. This does not filter or attenuate high frequencies already present in
  source recordings or selected synth material.
- Made the bloom window render-specific rather than fixed near 75%.
- Kept source recordings untouched; the change does not remove natural high
  frequencies or apply blanket de-noising.
- Exposed tonal centre through named root notes and scale strength in the main UI.
- Added regression tests ensuring silence cannot acquire a generated bloom tone
  and different source material produces different bloom content.
- Removed the compulsory low-frequency oscillator bed after the follow-up review
  identified a lower continuous pitch.
- Removed the remaining oscillator-based synth/arpeggio signature after it was
  heard as the same sine timbre across D1/D3/D5. Musical phrases are now cut
  from the selected source mix and developed through scale-related resampling,
  stretching, structured granulation, repetition, overlap and long windows.
- Added a source-derived granular bed: overlapping, windowed grains are sampled
  from the selected composition material. Only this added under-layer is
  body-shaped; the original source mix remains untouched.
- Restored intensity without restoring the common sine signature by separating
  the selected material into sustained and synth-like buses. These drive
  evolving source-derived drones and pads, while a motif bus creates denser
  rhythmic/arpeggio-like phrases. All three layers are reported separately.
- Extended the final event-edge guard from D5 to D1/D3/D5 and added a precise
  event timeline to every render report. This allows a reported sound at a given
  time to be traced to its source recording without blanket de-clicking.

## Research response

The request for broader training becomes a separately governed corpus study.
Only licensed, public-domain, Creative Commons, or directly permitted works may
be used. Composer-disjoint evaluation and provenance records are required before
any result can support an academic claim.

## Internal artistic review — phrase-shaped beds

The v21 direction was accepted as good for D1 and viable for D3. The remaining
issue was not treated as preference training alone: background layers sounded as
if material had been placed in a loop, while foreground events needed more
musical phrasing and clearer completion.

Release response:

- Preserve the approved v21 palette, synth preference and exploration settings.
- Replace the continuously perceived bed envelope with a finite sequence of
  phrases located by the learned whole-work energy and density trajectories.
- Give each bed phrase an asymmetric arrival, body and longer release so that it
  completes before another phrase answers it.
- Drive pad colour and presence from learned energy/brightness trajectories
  instead of a periodic whole-render sine-shaped modulation.
- Keep strong spectral, granular and tonal transformations restricted to learned
  formal development points; recurrence remains recognisable and finite.

## Internal artistic review — learned tonal blooms

The phrase-shaped beds improved continuity, but the result still needed the
musical confidence and electroacoustic flowering of the approved D1/D3 anchors.
The response is not a blanket ban on sine, synthesis or drones: it is a ban on
one compulsory oscillator signature recurring in every render.

Release response:

- Derive several tonal candidates from spectral peaks in the synth-like material
  selected from the active user library; do not use a fixed fallback pitch.
- When a scale is active, place those detected pitches gently inside that scale.
- Let D1/D3/D5 use one, two or three finite tonal blooms respectively, selected
  by learned whole-composition energy/density context rather than a fixed time.
- Shape pitch micro-motion from the source amplitude and give each bloom its own
  arrival, body, release and increasingly active spatial trajectory.
- Retain source-derived gestures, textures, granulation, drones and pads. The
  tonal layer supports their phrasing; it does not replace them or filter the
  original recordings.

## Internal artistic review — accepted-anchor evolution and pan

The original 103-sample library restored the electroacoustic identity and was
accepted as close to the first release. The final adjustment is intentionally
small: retain that render balance while giving phrases a clearer developmental
approach and less generic spatial motion.

Release response:

- Keep the level-specific D1/D3/D5 positive composition heads as the material,
  role-distribution and duration anchors.
- Preserve the existing learned development moments: accepted D1/D3 structure
  and owned-work energy, density, brightness and tonal trajectories continue to
  govern where material changes. A tested additional preparatory transformation
  was rejected because it altered the accepted morphology too strongly.
- Use the pan curves already measured from the owned complete works as a bounded
  contribution to phrase and gesture trajectories. Generic spatial motion is
  retained, so the learned contour guides rather than copies a work.

## Internal artistic review — complete-timeline artist-profile A/B

A controlled D3 comparison used the same 103-recording library, seed, C minor
harmony, material plan, event counts and mastering. Version A retained 24
evenly distributed eight-second embedding observations per owned work. Version
B embedded the complete timeline of all five works (32 minutes 42 seconds).
The artist preferred A.

Release decision:

- Retain A and the existing selective artist profile as the release default.
- Whole-work energy, density, brightness, tonal and pan trajectories continue
  to be analysed from the complete audio; only the deep timbral prototype uses
  the 24 representative observations.
- Do not activate B. Complete-timeline embedding increased the recurrence
  control from 1.209 to 1.268 and changed the aggregate prototype, but the
  additional coverage did not improve the artist's musical judgement.
- Preserve the complete-timeline profile only as isolated research evidence;
  it must not overwrite the approved D1/D3/D5 foundation.

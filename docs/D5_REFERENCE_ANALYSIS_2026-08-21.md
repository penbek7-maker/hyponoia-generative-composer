# D5 reference analysis

Two listener-selected audio references were analysed locally. Neither audio
file is included in the code package or intended for redistribution. Only the
aggregate temporal and dynamic targets below are retained.

## Measured characteristics

| Measure | Hyponoia volume/continuity reference | External rhythm/form reference |
| --- | ---: | ---: |
| Duration | 180.0 s | 788.7 s |
| Integrated loudness | -17.4 LUFS | -19.7 LUFS |
| Loudness range | 10.4 LU | 18.0 LU |
| Estimated pulse | 129.2 BPM | 123.1 BPM |
| Adjacent spectral similarity | 0.9955 | 0.9710 |
| Abrupt energy/spectral drops | 0.33/min | 0.84/min |

## Generator targets

- soft shared pulse between 122 and 129 BPM, without rigid beat quantisation;
- Hyponoia-like overall body around -17.4 LUFS;
- no more than 0.5 abrupt energy/spectral drops per minute as an acceptance
  target;
- continuous texture/resonance lanes with overlapping handovers;
- foreground phrases grouped rhythmically but finished with musical releases;
- energy through internal motion, development and synthetic transformation,
  rather than unrestricted layer count;
- preservation of D1 and D3 behaviour.

## Implementation

Generator revision `2026-08-21-d5-reference-continuity-3` adds soft-grid
positioning, lane-continuity scheduling, bounded D5 event growth, section-aware
internal motion and a final role-specific cosine attack/release guard. A
synthetic end-to-end smoke render measured -17.9 LUFS, 0 abrupt drop events per
minute and 0.9907 average adjacent spectral similarity. Private real-library
listening subsequently supplied the acceptance decision for the aesthetic
bridge release.

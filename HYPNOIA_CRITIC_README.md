# Hyponoia Critic v2.1 — How to use

## 1. Render first

Example:

```bash
python3 generator_v3_memory_bloom_smooth.py 5
```

It will save something like:

```text
output/Hyponoia_v3_memory_D5_20260709_153000.wav
```

## 2. Run the internal critic

```bash
python3 critic_v2.py output/Hyponoia_v3_memory_D5_20260709_153000.wav
```

This creates:

```text
critic_reports/Hyponoia_v3_memory_D5_20260709_153000_critic.json
```

## 3. Give human feedback

```bash
python3 human_feedback_v1.py critic_reports/Hyponoia_v3_memory_D5_20260709_153000_critic.json
```

You will score the same criteria as the system:

- musicality
- coherence
- richness
- transitions
- bloom_quality
- overall

Each score is 0–100.

## 4. Learning output

The script creates/updates:

```text
learning_profile.json
```

Human feedback is the primary target. The Critic score remains separate and is logged as diagnostic evidence. To assess calibration after collecting ratings, run:

```bash
python3 critic_calibration.py
```

This produces variance, MAE, signed error, and rank-correlation metrics in `critic_calibration_report.json`.

## Critic v2.1 interpretation

Critic v2.1 no longer equates coherence with the absence of abrupt amplitude or
spectral jumps. Its coherence score combines:

- short-term continuity;
- macro-form arc smoothness;
- development through the beginning, middle and ending;
- balance of samples, exposure and musical roles.

The report also contains `structural_history`, which compares role and section
proportions with earlier renders at the same D-level. This is a transparent
structural-novelty diagnostic, not a claim that different sample IDs guarantee
a perceptually different composition.

The Critic remains an auxiliary heuristic. A high Critic score never overrides
a lower human rating, and calibration must be reported with the paired sample
count for each dimension.

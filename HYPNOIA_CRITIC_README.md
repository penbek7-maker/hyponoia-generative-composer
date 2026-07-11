# Hyponoia Critic v1 — How to use

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
python3 critic_v1.py output/Hyponoia_v3_memory_D5_20260709_153000.wav
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

Next step: connect `learning_profile.json` back into the generator.

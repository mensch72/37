# Training results for the learned player of "37"

This documents the self-play run that produced the shipped weights
([`../web/weights/policy.json`](../web/weights/policy.json)). Numbers are win share
of **decided** games with the learned player rotated through all three seats; the
uniform baseline is **33.3%** (one seat in three). Reproduce with:

```bash
pip install numpy
python3 training/train_ppo.py --iters 170 --games-per-iter 24 \
    --out web/weights/policy.json --report training/training_report.json
python3 training/evaluate.py --games 300 --games-2ply 120
```

## What was trained

A small policy/value MLP (191 path-distance-augmented features → 192 → 128 → policy
head of 74 logits + a scalar value head), trained with **PPO + GAE** under
**population-based self-play** against a growing pool of frozen checkpoints, plus
occasional greedy opponents. This is the concrete realisation of the improvements
[FINDINGS section 7](../sandbox/results/FINDINGS.md) predicted would beat the
original A2C ceiling. Run: 170 iterations × 24 games ≈ 4 080 self-play games,
~12 minutes on one CPU core.

## Final strength (300 games each, 120 for 2-ply)

| player | vs 2 random | vs 2 greedy (1-ply heuristic) |
|---|---|---|
| raw policy head (argmax) | 73.7% | 3.3% |
| **policy + 1-ply value search** (shipped "Standard") | **99.3%** | **61.7%** |
| policy + 2-ply value search ("Strong") | 100.0% | 70.8% |

*(baseline = 33.3%)*

**Headline:** the shipped learned player **beats the greedy 1-ply heuristic** — the
explicit goal of the issue — reaching ~62% (1-ply search) and ~71% (2-ply search)
of decided games against two greedy opponents, roughly double the fair-share
baseline. The original design-session policies only reached *parity* (~33%) with
greedy; adding path-distance features and a shallow search over the learned value
head clears that ceiling, exactly as FINDINGS section 7 anticipated.

## Progress over self-play

From [`training_report.json`](training_report.json), the shipped configuration
(policy + 1-ply value search) vs two greedy opponents rose over training, and the
current policy came to beat a frozen copy of the **first checkpoint**:

| iteration | vs random (raw) | vs greedy (+1-ply search) | vs first checkpoint |
|---|---|---|---|
| 20 | 28% | 54% | 33% |
| 60 | 70% | 64% | 33% |
| 100 | 73% | 70% | 67% |
| 140 | 73% | 54% | 100% |
| 170 | 80% | 66% | 67% |

## Honest limitations

- **The value head, not the policy head, carries the strength.** The raw policy
  (argmax, no search) is *weak* against greedy (3.3%) even though it crushes random
  play (73.7%). The learned **value** function is accurate enough that a 1–2 ply
  search over it beats greedy handily; the policy head mainly serves as a move-
  ordering prior for that search. A stronger policy head would need more compute,
  hex-aware convolutions (only partially addressed here via path features), and
  longer training than this ~12-minute CPU run.
- **Small samples per checkpoint during training** (90 games) make the per-iteration
  columns noisy; the 300-game final evaluation above is the reliable measurement.
- **3-player self-play has no equilibrium guarantee** (FINDINGS section 7).
  Population-based play against past checkpoints mitigates but does not eliminate
  the risk of cycling; the frozen-checkpoint column is the guard against silent
  regressions.
- Everything here is measured against the scripted greedy floor and self-play, not
  against strong human play — treat it as design evidence, per the FINDINGS caveats.

## Files

- [`hexlife37.py`](hexlife37.py) — environment (beak economy, superko, elimination),
  the path-distance features (mirrored exactly by `web/js/features.js`), the greedy
  baseline and the network.
- [`train_ppo.py`](train_ppo.py) — PPO + GAE, population-based self-play, periodic
  evaluation, weight export.
- [`evaluate.py`](evaluate.py) — the stable multi-game evaluation that produced the
  table above ([`eval_summary.json`](eval_summary.json)).
- [`training_report.json`](training_report.json) — per-iteration trajectory of the
  shipped run.

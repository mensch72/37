# Findings from the design session

All simulations use the scripted 1-ply `greedy` player unless stated; board is 37 cells
(radius 3) unless stated. "decided" = someone connected; the rest are repetition draws
or move-cap timeouts. Chi-square is against a uniform 1/3 seat split (crit. 5.99 at 5%).

## 1. Choosing the automaton rule

Chosen: birth at exactly 4 with majority colour, survival 1–4 (**B4/S1-4**).

| rule | decided | avg plies | density | births+deaths per ply | note |
|---|---|---|---|---|---|
| **B4 / S1-4 (chosen)** | 92–93% | 35 | 0.43 | 2.6 | balanced |
| B4 / S1-3 | 76–83% | 130–160 | 0.31 | 3.2 | calm but a grind; chains erode as fast as built |
| B3 majority, 1-1-1 by lot / S1-3 | 99–100% | 33–42 | 0.48 | 6.0–6.5 | fast but turbulent; 14% of births by dice (18–19% in the opening) |

Birth at 4 with death at 5 gives structures exactly one point of slack: chains persist
long enough to cross, yet stay killable with commitment.

Two early bugs, both fatal if left in:

1. With isolation death and a *single* starting stone, the opening stone dies before the
   player moves again — nothing can ever be built. Fixed by starting each player with a
   2-cell still life.
2. A "last colour standing" win must not be checked before all three have moved, or the
   first player trivially wins at ply 1. The kill condition was later dropped entirely:
   under B4 births, annihilation essentially never happens (0 extinctions in 2700
   sampled positions without an economy).

## 2. Board size

| | 37 cells (R=3) | 61 cells (R=4) |
|---|---|---|
| greedy decided | 80% | 93% |
| greedy avg plies | 41 | 70 |
| random decided | 97% | 70% |
| cells changed per ply | 3.9 | 4.0 |

37 chosen: about half the physical operations per game, minimal crossing 7 not 9, more
robust to weak play.

## 3. Starting position

All candidates verified as still lifes. n = 1000 games each:

| start | decided | seats 1/2/3 (%) | chi2 | plies |
|---|---|---|---|---|
| tangential dominoes d=1 | 92% | 29.0 / 32.8 / 38.2 | 11.9 | 39 |
| tangential dominoes d=2 (run 1) | 92% | 34.0 / 32.7 / 33.3 | 0.2 | 54 |
| tangential dominoes d=2 (run 2, fresh seeds) | 92% | 33.1 / 37.6 / 29.3 | 9.7 | 54 |
| radial d=1 | 93% | 43.4 / 27.5 / 29.0 | 43.1 | 41 |
| adjacent ring | 91% | 34.0 / 38.1 / 27.9 | 14.4 | 37 |
| **ring, opposite (chosen)** | 93% | 35.0 / 35.8 / 29.2 | 7.1 | 34 |
| ring, opposite, turned (D3) | 92% | 40.6 / 32.6 / 26.8 | 26.8 | 53 |

- Starting attached to your own side is catastrophic for fairness (43% for seat 1).
  Distance from your own goal at the start is what dilutes first-mover advantage.
- Higher spatial symmetry (turned D3 ring) does **not** buy fairness and makes games
  ~60% longer, because both your starting cells sit on your own mid-line.
- Seat bias exists in almost every configuration, is of order ±4–10 pp, and its
  **direction depends on playing style** (the learned policy favoured different seats
  than greedy on the same setup). It cannot be designed away by geometry — hence the
  rotating-seat match rule.

## 4. Dynamics of the chosen rule (no economy)

Over 2700 played positions, 30 games:

- births 1.94/ply, deaths 2.36/ply (crowding 2.32, isolation 0.04), net +0.33 including
  the player's own move; about 4 cells change state per move.
- isolation death almost never fires; its role is to constrain *placement* (no growth
  into open space), not to kill.
- density rises from 0.25 in the opening to a plateau near 0.65 by ply 140.
- birth composition: 2-1-1 plurality 37%, 3-1 majority 34%, **2-2 → third colour 23%**,
  4-0 own 6%. ~60% of births have mixed parentage; every game contains at least one
  third-colour birth.
- colour shares (sorted per position) average 0.43 / 0.33 / 0.25; max/min ratio 1.86.
  No colour ever went extinct.

## 5. Economy: budget vs beak

### Budget (rejected)

Finite pool; deaths and removals refund; births are paid from the pool and mint a
permanent new stone when the pool is empty.

| | budget 6 | budget 4 |
|---|---|---|
| decided | 91% | 90% |
| plies | 47 | 58 |
| mints per game | 9.8 | 14.8 |
| **turns where the mover could not add** | 14% | **26%** |
| first minter wins | 48% | 42% |

Rejected: the pool jams — at budget 4 the mover cannot place on a quarter of all turns.
Depleting budgets (no refunds) are worse: density collapses to 0.25–0.30 and 70–90% of
games end undecided.

### Beak (chosen)

Births come from the box, deaths return to it; the beak only exchanges cells with the
board.

| | beak-4 | budget-4 | no economy |
|---|---|---|---|
| decided | 95–100% | 90% | 92–93% |
| seat chi2 across runs | 1.1, 2.0, 3.0, 3.7 | 5.4, 6.7 | 5.3, 7.1, 11.4 |
| turns unable to act | 0.2% | 26% | 0% |
| plies | 40–45 | 58 | 34 |

Further beak measurements (n = 1000, S1-4):

- move mix 63% drops / 37% picks; only 7% of picks rescued a cell that would have died,
  so the defensive use of picking was **not** found by the 1-ply player — headroom for
  stronger play.
- mean total cells per player stays near 6 for ~80 plies, then diverges: final totals
  average 10.9 (leader) vs 3.5 (laggard).
- extinction in 2% of games with greedy play but **93% under random play** — permanent
  deaths punish weak play hard.

### Beak capacity sweep

| | cap/start 4 | cap/start 3 | cap/start 2 |
|---|---|---|---|
| decided | 100% | 100% | 96% |
| plies | 45 | 45 | 55 |
| picks share of moves | 38% | 42% | 46% |
| reaches 7 total | 100%, median ply 5 | 100%, ply 10 | 98%, ply 14 |
| first to 7 total wins | 36% | 54% | **61%** |
| extinction games | 2% | 3% | 7% |
| mixed-parentage births | 91% | 91% | 91% |

The last row refutes the expectation that a small beak forces reliance on foreign
parents: mixed parentage is 91% at every capacity. The parasitic ecology was always the
core; a small beak changes how many times you must breed, not how.

### Capacity is inert; starting content is the dial

Assignments rotated over seats, n = 450 each:

| setup | win shares |
|---|---|
| caps 4/3/2, starts 4/3/2 | 50 / 33 / 17 |
| **caps 4/3/2, starts all 2** | **32 / 35 / 33** |
| caps all 4, starts 4/3/2 | 48 / 31 / 22 |

Capacity alone has no measurable effect. Separately confirmed: an unlimited cap is
indistinguishable from cap = start (players never hold more than they started with).
So the rules can simply say "unlimited beak, N cells at the start".

### Beak does not rescue S1-3

With permanent deaths and death at ≥4 neighbours the ecosystem drains away: 96% of games
end with extinctions, 3% decided, density 0.18.

## 6. Learned policies and forced outcomes

Environment: 37-cell board; canonicalised observation (board rotated so the acting
player's axis is always "x" and colour channel 0 is always "self"); action space
74 = drop@i | pick@i with a legality mask; terminal rewards +1 / −0.5 / −0.5, draw −0.1.
The 120° symmetry lets one shared policy play all three seats, so self-play is exact.

- Base game, 20k self-play episodes, 128-unit MLP, A2C: **98%** win share vs two random
  opponents, **32%** vs two greedy opponents (uniform baseline 33%) — parity with the
  1-ply heuristic, not better.
- The policy failed to transfer to unseen starting positions (0–1% decided on ring starts
  when trained on tangential dominoes) — pure distribution shift.

### Can a player force a stalemate?

Probes against a frozen base policy, one seat learning:

| condition | draws / stalls |
|---|---|
| repetition = draw, base vs base | 33% |
| repetition = draw, spoiler trained to draw | **73%** |
| **superko** (repetition illegal), base vs base | 1% |
| superko, unadapted spoiler | 9% |
| superko, spoiler retrained to stall | **1%** |

Under the actual rule (repetition illegal, no legal move = elimination), a spoiler
trained *specifically to stall* got monotonically worse at stalling (8 → 5 → 2 → 1%)
while also losing games. Within this policy class stalemate cannot be forced — but only
because repetition is illegal rather than a draw. The superko clause is structural.

An exploiter trained to maximise its own wins reached 25% against two base policies
(baseline ≈ 22%) — no evidence of a forcible win either.

Zugzwang never occurred in thousands of superko games; the clause is a deterrent
backstop, not a live rule.

## 7. Open questions

- Stronger agents: greedy gets Dijkstra distances for free while the network must learn
  them from raw one-hots. Path features, hex-aware convolutions, or a shallow search over
  a learned value head (only ~23 legal moves, so 2-ply ≈ 500 evaluations) should all beat
  the current ceiling.
- Population-based self-play: symmetric shared-policy self-play in a 3-player general-sum
  game has no equilibrium guarantee and may converge to something exploitable.
- Hoarding: with an unlimited beak nothing stops a player picking their whole colony into
  the beak — safe from death, unable to win. The 1-ply player never does this; a human
  might. If it becomes a real spoiler tactic, reinstating a cap (say 6) is the fix.
- Whether the beak adds depth or only bookkeeping is a playtesting question. For depth:
  picking has defensive uses the scripted player never found. Against: an economy-blind
  heuristic plays the beak game about as well as the plain game.

# 37

**37** is a three-player abstract board game on a hexagon of 37 cells driven by a
Conway-like cellular automaton. Each player owns one pair of opposite sides of the
board and wins by joining their two sides with an unbroken chain of their own
colour. The shortest possible winning chain is 7 cells — hence the name: **3**7
cells, **3** players, **7**-cell path.

This repository implements 37 as a **static web app** (served from GitHub Pages)
that runs entirely in the browser — including a **learned computer player** trained
by self-play — plus the training pipeline and the design exploration it grew from.

> Play it: once GitHub Pages is enabled for this repository (Settings → Pages →
> Build and deployment → GitHub Actions), the app is published from the
> [`web/`](web/) directory by [`.github/workflows/pages.yml`](.github/workflows/pages.yml).
> To run locally: `cd web && python3 -m http.server 8000` then open
> <http://localhost:8000/>.

## The game

The board is a boulder, the colours are strains of lichen, and the players are
birds that pick up and drop flecks of lichen. On your turn you either **drop** a
fleck from your beak onto an empty cell or **pick** one of your own cells back into
your beak; then the whole boulder grows at once. Birth is the same in every rule
variant — an empty cell with **exactly 4** neighbours is **born**, its colour the
majority of its four parents, the plurality in a 2-1-1, or — in a 2-2 tie — **the
absent third colour** — but there are two selectable **survival** rules:

- **calm (S1-5, the default):** a living cell dies only if it is **isolated** (0
  neighbours) or **completely surrounded** (all 6 neighbours alive); it survives
  with 1–5 neighbours. Only interior cells can ever be surrounded, since rim cells
  have at most 4 neighbours. This roughly halves the number of cell changes between
  your turns, so the board is easier to follow.
- **volatile (S1-4):** a living cell also dies of **crowding** at **5 or 6**
  neighbours (survives with 1–4). Faster, more churn, the harder option.

Newborn flecks come from the box and dead flecks return to it; your beak only ever
exchanges flecks with the board (the *beak economy*). Your beak starts with **N**
flecks, which is the whole difficulty dial, and is **capped at 6** so no bird can
hoard or run a "centre pump" indefinitely. A move may not recreate a whole
position that has already occurred (superko); a player with no cells on the board
and an empty beak is eliminated, and so is a player **all four of whose home-side
cells are occupied by other colours** — that side is permanently blocked, so they
can no longer connect and are declared out (their cells stay as terrain).

### Difficulty dial

`N` sets how many net births you must win from the automaton before a win is
possible (`7 − N − 2`):

| setting | beak at start | total cells | births needed |
|---|---|---|---|
| novice | unlimited | — | 0 |
| advanced (standard) | 4 | 6 | 1 |
| expert | 2 | 4 | 3 |
| handicap | 4 / 3 / 2 per player | 6 / 5 / 4 | 1 / 2 / 3 |

Except at *novice* (whose beak is unlimited by design), the beak is capped at 6 —
one above the largest holding ever seen in ordinary play — which costs nothing
normally but blocks indefinite hoarding and the centre pump.

The full canonical rules and the reasoning behind every choice are in
[`sandbox/README.md`](sandbox/README.md) and
[`sandbox/results/FINDINGS.md`](sandbox/results/FINDINGS.md).

## The computer player is learned, not scripted

The opponent is a small neural **policy/value network trained by self-play**, run
client-side as plain numeric code over exported weights (in
[`web/weights/`](web/weights)). It is *not* a hand-written heuristic. Because a
policy trained on one ruleset does **not** transfer to another, a **separate weight
file is trained per rule variant** — `policy-calm.json` and `policy-volatile.json`
— and the app loads the one matching the selected rule, falling back to the legacy
`policy.json` if a variant file is missing. The scripted 1-ply `greedy` heuristic
from the design session is kept only as a training/evaluation baseline (the "floor
on competence" from FINDINGS), alongside three scripted strategies discovered in
human play — an **endpoint-claimer**, a **rim-arc blocker** and a **centre
pumper** — which are now part of the training opponent pool and the strength
report.

The training pipeline in [`training/`](training) implements the improvements that
[FINDINGS section 7](sandbox/results/FINDINGS.md) predicted would beat the original
A2C ceiling (which only reached parity with greedy):

- **path-distance features** — the network is given the same Dijkstra path
  information the greedy heuristic gets for free (see `compute_features` in
  [`training/hexlife37.py`](training/hexlife37.py));
- **PPO with GAE** instead of vanilla A2C;
- **population-based self-play** against frozen past checkpoints, rather than pure
  symmetric self-play (which has no equilibrium guarantee in a 3-player
  general-sum game);
- a **shallow max-n search over the learned value head** at inference time (only
  ~23 legal moves, so 1–2 ply is a few hundred evaluations), implemented in
  [`web/js/ai.js`](web/js/ai.js).

The one shared policy plays all three seats via the canonicalisation trick from
`marl_hexlife.py`: the board is rotated so the acting player's axis is fixed and
the colours are relabelled so channel 0 is always "self".

### Measured strength

See [Training and evaluation](#training-and-evaluation) below — the numbers are
reported directly from the training run that produced the shipped weights
([`training/training_report.json`](training/training_report.json)).

## UI

The app shows the board, each player's beak contents and total cell count, whose
turn it is, and — importantly for a cellular-automaton game — what the growth step
just did (cells that were **born** get a dashed ring, cells that **died** get a
small cross in the colour of the fleck that died). When a player wins by
connection, the UI also traces one concrete winning chain between that player's two
sidelines. The board is a dark "rock" grey with thin cell borders, and each
player's pair of sides is marked by a coloured line just off the outer cells.
Colours use the red / yellow / blue palette (colour-blind safe, avoids the
red–green axis) and are distinguished by **shape** as well as hue, using only
60°-rotationally-symmetric marks (Red hexagon, Yellow star, Blue circle). A
move-history scrubber lets you replay the game, undo the live position, or return
to any replayed state, and a move-speed slider paces the computer players so a
human can follow the action.

Any combination of the three seats can be human or computer (local hot-seat plus
computer opponents); there are no accounts, servers or online multiplayer.

## Repository layout

```
web/                 the static web app (this is what GitHub Pages serves)
  index.html
  css/style.css
  js/engine.js       core rules engine (shared by app and tests)
  js/features.js     canonicalised + path-distance features (mirrors Python)
  js/ai.js           policy/value inference + shallow value-head search
  js/ui.js           SVG board renderer + player panel
  js/main.js         app controller (game loop, history, growth log)
  weights/             exported trained weights (client-side inference)
    policy-calm.json     for the calm (S1-5, default) rule
    policy-volatile.json for the volatile (S1-4) rule
    policy.json          legacy single-rule weights (fallback)
training/            self-play training pipeline
  hexlife37.py       environment + features + baselines + scripted strategies + network
  train_ppo.py       PPO + GAE + population-based self-play + export + eval
  evaluate.py        per-strategy strength table
  training_report-*.json  strength trajectory per rule variant
tests/               unit tests
  gen_reference.py   generates reference vectors from sandbox/code/beak3.py
  engine.test.mjs    JS growth step vs the Python reference (+ invariants)
  features.test.mjs  JS features vs Python features (weight-transfer safety)
sandbox/             the original design exploration (rules, reference Python)
```

## Testing

The growth step is the easy thing to get subtly wrong (the simultaneous update and
the 2-2 birth rule in particular), so it is unit-tested against the Python
reference implementation on thousands of randomly generated positions.

```bash
python3 tests/gen_reference.py 3000   # regenerate reference vectors (needs the sandbox Python)
npm test                              # run the JS engine + feature-parity tests
```

`tests/engine.test.mjs` replays every reference vector through the JavaScript
`growth()` and asserts identical output, checks board topology, the still-life
start, connection/elimination and that random games always terminate.
`tests/features.test.mjs` checks that the browser feature extractor reproduces the
Python trainer's features exactly, which is what lets weights trained in Python run
unchanged in the browser.

## Training and evaluation

To reproduce (or improve) the shipped policies, train one per rule variant (the
`--variant` flag selects the survival rule and the matching default output paths):

```bash
pip install numpy
# calm (S1-5, the default) and volatile (S1-4), advanced beak of 4, cap 6:
python3 training/train_ppo.py --variant calm     # -> web/weights/policy-calm.json
python3 training/train_ppo.py --variant volatile # -> web/weights/policy-volatile.json
# then the per-strategy strength table for each:
python3 training/evaluate.py --variant calm
python3 training/evaluate.py --variant volatile
```

The script prints, and records to the report, the win share of decided games (the
uniform baseline is 33%) for the current policy in each seat against:

- two **random** opponents,
- two **greedy** (1-ply heuristic) opponents — both for the raw policy and for the
  shipped configuration with the 1-ply value search,
- two frozen copies of the **first checkpoint**, to show progress over self-play.

`evaluate.py` additionally reports each configuration (raw / 1-ply / 2-ply search)
against every scripted strategy — greedy, endpoint-claimer, rim-arc blocker and
centre pumper — since a policy that cannot beat the endpoint-claimer is not ready
to ship.

See [`training/RESULTS.md`](training/RESULTS.md) for the numbers from the run that
produced the committed weights, and an honest discussion of what the learned player
can and cannot do.

## Licence

MIT — see [LICENSE](LICENSE).

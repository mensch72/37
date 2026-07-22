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
your beak; then the whole boulder grows at once:

- a living cell with **0** neighbours dies of isolation;
- a living cell with **1–4** neighbours survives;
- a living cell with **5–6** neighbours dies of crowding;
- an empty cell with **exactly 4** neighbours is **born**. Its colour is the
  majority of its four parents, the plurality in a 2-1-1, or — in a 2-2 tie — **the
  absent third colour**.

Newborn flecks come from the box and dead flecks return to it; your beak only ever
exchanges flecks with the board (the *beak economy*). Your beak starts with **N**
flecks, which is the whole difficulty dial. A move may not recreate a whole
position that has already occurred (superko); a player with no cells on the board
and an empty beak is eliminated.

### Difficulty dial

`N` sets how many net births you must win from the automaton before a win is
possible (`7 − N − 2`):

| setting | beak at start | total cells | births needed |
|---|---|---|---|
| novice | unlimited | — | 0 |
| advanced (standard) | 4 | 6 | 1 |
| expert | 2 | 4 | 3 |
| handicap | 4 / 3 / 2 per player | 6 / 5 / 4 | 1 / 2 / 3 |

The full canonical rules and the reasoning behind every choice are in
[`sandbox/README.md`](sandbox/README.md) and
[`sandbox/results/FINDINGS.md`](sandbox/results/FINDINGS.md).

## The computer player is learned, not scripted

The opponent is a small neural **policy/value network trained by self-play**, run
client-side as plain numeric code over exported weights
([`web/weights/policy.json`](web/weights)). It is *not* a hand-written heuristic.
The scripted 1-ply `greedy` heuristic from the design session is kept only as a
training/evaluation baseline (the "floor on competence" from FINDINGS).

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
fading cross). Colours use the red / yellow / blue palette (colour-blind safe,
avoids the red–green axis) and are distinguished by **shape** as well as hue
(Red ▲, Yellow ●, Blue ■). A move-history scrubber lets you replay the game.

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
  weights/policy.json  exported trained weights (client-side inference)
training/            self-play training pipeline
  hexlife37.py       environment + features + baselines + network
  train_ppo.py       PPO + GAE + population-based self-play + export + eval
  training_report.json  strength trajectory of the shipped run
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

To reproduce (or improve) the shipped policy:

```bash
pip install numpy
python3 training/train_ppo.py --iters 170 --games-per-iter 24 \
    --out web/weights/policy.json --report training/training_report.json
```

The script prints, and records to the report, the win share of decided games (the
uniform baseline is 33%) for the current policy in each seat against:

- two **random** opponents,
- two **greedy** (1-ply heuristic) opponents — both for the raw policy and for the
  shipped configuration with the 1-ply value search,
- two frozen copies of the **first checkpoint**, to show progress over self-play.

See [`training/RESULTS.md`](training/RESULTS.md) for the numbers from the run that
produced the committed weights, and an honest discussion of what the learned player
can and cannot do.

## Licence

MIT — see [LICENSE](LICENSE).

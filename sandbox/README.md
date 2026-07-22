# sandbox — design exploration for "37"

Raw output of the design session in which **37** was invented, simulated and tuned.
Kept as produced: exploratory scripts, benchmark numbers, figures, rules PDF.
Not a library — evidence and reference material.

## What 37 is

A 3-player abstract board game on a hexagon of 37 cells driven by a Conway-like
cellular automaton. Each player owns one pair of opposite sides; you win by joining
your two sides with an unbroken chain of your own colour. The shortest possible
winning chain is 7 cells — hence the name: 37 cells, 3 players, 7-cell path.

## Canonical rules

- **Board**: hexagon of side 4 = 37 cells (cube coordinates), up to 6 neighbours.
- **Start**: each player has 2 cells on *opposite neighbours of the centre*, so the six
  starting cells form a tricolour ring. This is a still life.
- **Move**, then the growth step: either **drop** a cell from your beak onto an empty
  cell, or **pick** one of your cells off the board into your beak.
- **Growth step** — all cells update simultaneously, read from the position *before*
  the step, counting living neighbours of *any* colour:
  - occupied, 0 neighbours → dies (isolation)
  - occupied, 1–4 → survives
  - occupied, 5–6 → dies (crowding)
  - empty, exactly 4 → a cell is **born**
- **Birth colour** from the 4 parents: 4-0 or 3-1 → majority; 2-1-1 → plurality;
  **2-2 → the absent third colour**.
- **Beak economy**: newborn cells come from the box, dead cells return to the box —
  births and deaths never touch a beak. Your beak starts with **N** cells; N is the
  whole difficulty dial. Beak capacity is measurably irrelevant, so treat it as
  unlimited.
- **Elimination**: no cells on the board and none in the beak → out.
- **Repetition**: a move may not recreate a whole position (board + all beaks + player
  to move) that already occurred. No legal move → out.
- **Match play**: 3 games with rotated seating (a small seat effect exists and its
  direction depends on playing style).

### Difficulty dial

| setting | beak at start | total cells | births needed before a win is possible |
|---|---|---|---|
| novice | unlimited | — | 0 |
| advanced (standard) | 4 | 6 | 1 |
| expert | 2 | 4 | 3 |
| handicap | 4 / 3 / 2 per player | 6 / 5 / 4 | 1 / 2 / 3 |

A winning chain needs 7 cells, so a starting total of `N+2` implies at least
`7 − N − 2` net births must be won from the automaton. Handicap measured at roughly
48 : 31 : 22 win shares for starts 4 / 3 / 2.

### Theme

The board is a boulder, the colours are lichen strains, the players are birds that
pick up and drop flecks of lichen. Palette red / yellow / blue (colour-blind safe).

## Contents

- `code/` — simulation scripts as written during the session
- `results/FINDINGS.md` — every benchmark run, with numbers and conclusions
- `results/RAW_OUTPUT_NOTES.md` — provenance notes for the scripts
- `figures/`, `rules/` — figures and the printable rules PDF (binary; added separately)

## Caveats on every number here

- `greedy_player` is a **1-ply** search with a hand-tuned heuristic (Dijkstra distance
  between your own two sides, own distance minus opponents'). A floor on competence.
- Learned policies are small numpy MLPs trained with vanilla A2C for tens of thousands
  of self-play episodes. They beat random play comfortably and reach parity with the
  greedy player — no more.
- Sample sizes are stated per experiment; seat fairness uses chi-square against a
  uniform 1/3 split (critical value 5.99 at 5%).

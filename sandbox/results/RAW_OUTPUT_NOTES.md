# Notes on provenance

- `code/hex3life.py` — original engine. Board radius is a module constant `R`; most later
  experiments load this file and textually replace `R = 4` with `R = 3`. Ugly, but it is
  how the numbers in FINDINGS.md were produced.
- `code/beak.py`, `beak2.py`, `beak3.py` — the beak model. `beak3.py` is the final one and
  takes the beak capacity and the number of games as command-line arguments.
- `code/bigbench.py`, `onebench.py`, `onebench2.py` — starting-position sweeps.
- `code/marl_hexlife.py` — MARL environment + A2C self-play for the base (no economy) game.
- `code/marl_budget.py` — same for the budget variant, plus the spoiler/exploiter probes.
- Trained weights (`*.npz`, numpy archives of the MLP parameters) and the figures and rules
  PDF are binary and were delivered outside this commit; add them by hand if wanted.
- The B3-with-lottery variant was benchmarked in a browser widget rather than a script,
  and that code was not preserved; its numbers are in FINDINGS.md section 1.
- Several scripts `exec()` each other's source. This is deliberate scaffolding from an
  interactive session, not a pattern to imitate.

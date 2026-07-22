"""Generate reference test vectors from the Python reference implementation.

Loads the final beak-economy growth step (sandbox/code/beak3.py) exactly as it was
used to produce the FINDINGS numbers, runs it on many randomly generated positions,
and writes tests/vectors.json. The Node test (engine.test.mjs) replays the same
inputs through the JavaScript engine and asserts identical outputs — this is the
check demanded by issue #1 that the simultaneous update and the 2-2 birth rule are
ported correctly.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "sandbox", "code")

# Reproduce beak3.py's loading of the R=3 engine and its trans()/helpers.
src = open(os.path.join(CODE, "hex3life.py")).read().replace("R = 4", "R = 3").split("if __name__")[0]
M = {}
exec(src, M)  # noqa: S102 - trusted local reference code
CELLS = M["CELLS"]
IDX = M["IDX"]
N = M["N"]
EMPTY = M["EMPTY"]
NBRS = M["NBRS"]
assert N == 37, N

beak_src = open(os.path.join(CODE, "beak3.py")).read()
# beak3.py opens hex3life.py relative to CWD; run from the code dir for the exec.
os.chdir(CODE)
# Exec the whole module with a non-__main__ name so its benchmark block is skipped
# but trans() and helpers are defined. (Splitting on "if __name__" is unsafe here
# because that exact string appears inside the file's source text.)
B = {"__name__": "_beak3_ref"}
exec(beak_src, B)  # noqa: S102 - trusted local reference code
trans = B["trans"]


def board_to_list(board):
    return [int(v) for v in board]


def random_board(rng):
    """Random position: each cell empty or one of three colours, biased toward
    the densities seen in real play so births actually fire."""
    board = [EMPTY] * N
    for i in range(N):
        r = rng.random()
        if r < 0.45:
            board[i] = EMPTY
        else:
            board[i] = rng.randint(0, 2)
    return board


def make_vectors(n, seed=12345):
    rng = random.Random(seed)
    out = []
    # Always include the canonical still-life start and a few hand-made 2-2 setups.
    board0 = [EMPTY] * N
    ROT = lambda c: (c[2], c[0], c[1])
    cells = [(1, 0, -1), (-1, 0, 1)]
    for p in range(3):
        for c in cells:
            board0[IDX[c]] = p
        cells = [ROT(c) for c in cells]
    fixed = [board0]

    for board in fixed:
        res = trans(board[:], 4)
        out.append({"in": board_to_list(board), "out": board_to_list(res)})

    for _ in range(n):
        board = random_board(rng)
        res = trans(board[:], 4)
        out.append({"in": board_to_list(board), "out": board_to_list(res)})
    return out


def cells_meta():
    return {
        "cells": [list(c) for c in CELLS],
        "nbrs": [list(nb) for nb in NBRS],
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    vectors = make_vectors(n)
    data = {"meta": cells_meta(), "vectors": vectors}
    with open(os.path.join(HERE, "vectors.json"), "w") as f:
        json.dump(data, f)
    print(f"wrote {len(vectors)} vectors to tests/vectors.json")

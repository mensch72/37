"""Thorough evaluation of a trained policy for the RESULTS writeup.

Loads exported weights, rebuilds the network and reports stable win shares (more
games than the per-iteration training evals) for:

  * the raw policy head (argmax),
  * the shipped configuration: policy + 1-ply value-head search,
  * policy + 2-ply value-head max-n search (the browser "Strong" option),

against a random opponent and each scripted strategy agent (a plain path-distance
greedy agent, the endpoint-claimer, the rim-arc blocker and the centre pumper —
the tactics found in human play, issue #12). The uniform baseline is 33.3% (one
seat out of three). A policy that cannot beat the endpoint-claimer is not ready.
"""
import argparse
import json
import os
import random

import numpy as np

import hexlife37 as H
from hexlife37 import (Env, Net, N, SCRIPTED, VARIANTS, DEFAULT_CAP,
                       winner_after, compute_features)


def load_net(path):
    w = json.load(open(path))
    net = Net(nin=w["nin"], h1=w["h1"], h2=w["h2"])
    net.P["W1"] = np.array(w["W1"]); net.P["b1"] = np.array(w["b1"])
    net.P["W2"] = np.array(w["W2"]); net.P["b2"] = np.array(w["b2"])
    net.P["Wp"] = np.array(w["Wp"]); net.P["bp"] = np.array(w["bp"])
    net.P["Wv"] = np.array(w["Wv"]); net.P["bv"] = np.array([w["bv"]])
    return net


def leaf_value(net, board, beaks, persp):
    w = winner_after(board, persp)
    if w is not None:
        return 1.0 if w == persp else -0.5
    _, v, _ = net.forward(compute_features(board, beaks, persp))
    return v


def maxn(net, board, beaks, player, depth, root, surv_max, topk=8):
    """Shallow max-n over the value head. board/beaks are real-space; returns the
    value for `root`. Restricted to the top-k policy moves to bound cost."""
    mask = H.canonical_mask(board, beaks, player)
    acts = [a for a in range(2 * N) if mask[a]]
    if not acts:
        return leaf_value(net, board, beaks, root)
    logits, _, _ = net.forward(compute_features(board, beaks, player))
    lg = logits.copy(); lg[~mask] = -1e9
    order = sorted(acts, key=lambda a: -lg[a])[:topk]
    best_player, best_root = -1e9, leaf_value(net, board, beaks, root)
    for a in order:
        real = int(H.PERM[player][a % N])
        nb, nk = board[:], beaks[:]
        if a < N:
            nb[real] = player; nk[player] -= 1
        else:
            nb[real] = H.EMPTY; nk[player] += 1
        nb = H.growth(nb, surv_max)
        w = winner_after(nb, player)
        if w is not None:
            pv = 1.0 if w == player else -0.5
            rv = 1.0 if w == root else -0.5
        elif depth <= 1:
            pv = leaf_value(net, nb, nk, player)
            rv = pv if player == root else leaf_value(net, nb, nk, root)
        else:
            nxt = (player + 1) % 3
            rv = maxn(net, nb, nk, nxt, depth - 1, root, surv_max, topk)
            pv = rv if player == root else maxn(net, nb, nk, nxt, depth - 1, player, surv_max, topk)
        if pv > best_player:
            best_player, best_root = pv, rv
    return best_root


def choose(net, env, mode):
    p = env.player
    if mode == "raw":
        pi, _, _, _ = net.policy(env.obs(), env.legal_mask())
        pi = pi * env.legal_mask()
        return int(np.argmax(pi))
    depth = 1 if mode == "search1" else 2
    best_a, best_v = None, -1e9
    for a in env.legal_actions():
        board, beaks = env._simulate(p, a)
        w = winner_after(board, p)
        if w == p:
            return a
        if w is not None:
            v = -1e6
        elif depth == 1:
            v = leaf_value(net, board, beaks, p)
        else:
            v = maxn(net, board, beaks, (p + 1) % 3, depth - 1, p, env.surv_max)
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def play_vs(net, mode, opp, ngames, beak_start, cap, surv_max, seed=2024):
    rng = random.Random(seed)
    wins = dec = 0
    for g in range(ngames):
        seat = g % 3
        env = Env(beak_start=beak_start, cap=cap, surv_max=surv_max)
        while not env.done:
            p = env.player
            if not env.legal_mask().any():
                break
            if p == seat:
                a = choose(net, env, mode)
            elif opp == "random":
                acts = env.legal_actions(); a = acts[rng.randint(0, len(acts) - 1)]
            else:
                a = SCRIPTED[opp](env, rng)
            env.step(a)
        w = getattr(env, "winner", None)
        if w is not None:
            dec += 1
            wins += (w == seat)
    return 100.0 * wins / max(dec, 1), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="calm")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--beak", type=int, default=4)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--games-2ply", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    if args.weights is None:
        args.weights = os.path.join(here, "..", "web", "weights", f"policy-{args.variant}.json")
    if args.out is None:
        args.out = os.path.join(here, f"eval_summary-{args.variant}.json")
    surv_max = VARIANTS[args.variant]
    beak_start = (args.beak, args.beak, args.beak)

    net = load_net(args.weights)
    # opponents: random plus each scripted strategy agent.
    opps = ["random", "greedy", "endpoint", "rim_arc", "pump"]
    labels = {"random": "vs random", "greedy": "vs greedy (path-dist)",
              "endpoint": "vs endpoint-claimer", "rim_arc": "vs rim-arc blocker",
              "pump": "vs centre pumper"}
    rows = []
    print(f"Evaluating {args.weights}  variant={args.variant} cap={args.cap} "
          f"beak={args.beak}  (uniform baseline = 33.3%)\n")
    header = f"  {'player':52s} | " + " | ".join(f"{labels[o]:>22s}" for o in opps)
    print(header)
    for mode, label, ng in [
        ("raw", "raw policy (argmax)", args.games),
        ("search1", "policy + 1-ply value search (shipped 'Standard')", args.games),
        ("search2", "policy + 2-ply value search ('Strong')", args.games_2ply),
    ]:
        cells = {}
        for o in opps:
            wr, dr = play_vs(net, mode, o, ng, beak_start, args.cap, surv_max)
            cells[o] = dict(win_pct=round(wr, 1), decided=dr, games=ng)
        rows.append(dict(mode=mode, label=label, games=ng, vs=cells))
        print(f"  {label:52s} | " + " | ".join(f"{cells[o]['win_pct']:20.1f}%" for o in opps))
    json.dump({"weights": args.weights, "variant": args.variant, "cap": args.cap,
               "beak": args.beak, "baseline_pct": 33.3, "opponents": opps,
               "rows": rows}, open(args.out, "w"), indent=2)
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()

"""Thorough evaluation of a trained policy for the RESULTS writeup.

Loads exported weights (web/weights/policy.json), rebuilds the network and reports
stable win shares (more games than the per-iteration training evals) for:

  * the raw policy head (argmax),
  * the shipped configuration: policy + 1-ply value-head search,
  * policy + 2-ply value-head max-n search (the browser "Strong" option),

against two random opponents and two greedy (1-ply heuristic) opponents. The
uniform baseline is 33.3% (one seat out of three). Also reports the shipped policy
vs a frozen earlier checkpoint if one is supplied.
"""
import argparse
import json
import os
import random

import numpy as np

import hexlife37 as H
from hexlife37 import Env, Net, N, greedy_action, winner_after, compute_features


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


def maxn(net, env_like, board, beaks, player, depth, root, topk=8):
    """Shallow max-n over the value head. board/beaks are real-space; returns the
    value for `root`. Restricted to the top-k policy moves to bound cost."""
    # legal canonical actions on this position (economy mask only; superko ignored
    # inside the search for speed — the environment enforces it at the real move).
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
        nb = H.growth(nb)
        w = winner_after(nb, player)
        if w is not None:
            pv = 1.0 if w == player else -0.5
            rv = 1.0 if w == root else -0.5
        elif depth <= 1:
            pv = leaf_value(net, nb, nk, player)
            rv = pv if player == root else leaf_value(net, nb, nk, root)
        else:
            nxt = (player + 1) % 3
            rv = maxn(net, env_like, nb, nk, nxt, depth - 1, root, topk)
            pv = rv if player == root else maxn(net, env_like, nb, nk, nxt, depth - 1, player, topk)
        if pv > best_player:
            best_player, best_root = pv, rv
    return best_root


def choose(net, env, mode):
    p = env.player
    if mode == "raw":
        pi, _, _, _ = net.policy(env.obs(), env.legal_mask())
        pi = pi * env.legal_mask()
        return int(np.argmax(pi))
    # value search over legal (superko-checked) moves
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
            v = maxn(net, env, board, beaks, (p + 1) % 3, depth - 1, p)
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def play_vs(net, mode, opp, ngames, opp_net=None, seed=2024):
    rng = random.Random(seed)
    wins = dec = 0
    for g in range(ngames):
        seat = g % 3
        env = Env()
        while not env.done:
            p = env.player
            if not env.legal_mask().any():
                break
            if p == seat:
                a = choose(net, env, mode)
            elif opp == "greedy":
                a = greedy_action(env, rng)
            elif opp == "random":
                acts = env.legal_actions(); a = acts[rng.randint(0, len(acts) - 1)]
            else:
                a = choose(opp_net, env, "raw")
            env.step(a)
        w = getattr(env, "winner", None)
        if w is not None:
            dec += 1
            wins += (w == seat)
    return 100.0 * wins / max(dec, 1), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "..", "web", "weights", "policy.json"))
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--games-2ply", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_summary.json"))
    args = ap.parse_args()

    net = load_net(args.weights)
    rows = []
    print(f"Evaluating {args.weights} (uniform baseline = 33.3%)\n")
    for mode, label, ng in [
        ("raw", "raw policy (argmax)", args.games),
        ("search1", "policy + 1-ply value search (shipped 'Standard')", args.games),
        ("search2", "policy + 2-ply value search ('Strong')", args.games_2ply),
    ]:
        wr, dr = play_vs(net, mode, "random", ng)
        wg, dg = play_vs(net, mode, "greedy", ng)
        rows.append(dict(mode=mode, label=label, games=ng,
                         vs_random=round(wr, 1), vs_greedy=round(wg, 1)))
        print(f"  {label:52s} | vs random {wr:5.1f}% | vs greedy {wg:5.1f}%")
    json.dump({"weights": args.weights, "baseline_pct": 33.3, "rows": rows}, open(args.out, "w"), indent=2)
    print("\nsaved ->", args.out)


if __name__ == "__main__":
    main()

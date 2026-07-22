"""Self-play training for the learned player of "37".

Implements the improvements listed in FINDINGS section 7 over the original vanilla
A2C policies (which only reached parity with the 1-ply greedy heuristic):

  * path-distance features (see hexlife37.compute_features),
  * PPO with Generalised Advantage Estimation instead of vanilla A2C,
  * population-based self-play against frozen past checkpoints rather than pure
    symmetric self-play (which has no equilibrium guarantee in a 3-player game).

The shallow value-head search suggested there is implemented on the inference side
(web/js/ai.js); this script also reports the raw-policy strength with and without a
1-ply value search so the numbers are comparable.

Usage:
    python3 training/train_ppo.py --iters 200 --games-per-iter 24 \
        --out web/weights/policy.json
"""
import argparse
import collections
import json
import os
import random
import time

import numpy as np

import hexlife37 as H
from hexlife37 import Env, Net, N, greedy_action


def gae(values, reward, gamma, lam):
    """GAE over one seat's own-decision sequence. Rewards are 0 except the final
    own-decision, which receives `reward`; bootstrap value after terminal is 0."""
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        v_next = 0.0 if t == T - 1 else values[t + 1]
        r = reward if t == T - 1 else 0.0
        delta = r + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    ret = adv + np.asarray(values, dtype=np.float32)
    return adv, ret


def sample_action(net, x, mask, rng):
    pi, v, _, logits = net.policy(x, mask)
    a = int(rng.choice(2 * N, p=pi))
    logp = np.log(pi[a] + 1e-12)
    return a, logp, v


def collect(cur, pool, rng, games, beak_start, gamma, lam):
    """Play `games` self-play games; return a batch of the current policy's own
    decisions with GAE advantages. Seats not controlled by `cur` are played by a
    frozen opponent sampled from `pool` (or greedy/current)."""
    batch = collections.defaultdict(list)
    stats = {"decided": 0, "plies": []}
    for gi in range(games):
        env = Env(beak_start=beak_start)
        # Assign seats. Always keep at least one 'cur' seat for data. Opponents
        # are sampled from the population pool (frozen nets), plus occasional
        # greedy and current, for population-based self-play.
        seat_kind = []
        for s in range(3):
            r = rng.random()
            if r < 0.5 or not pool:
                seat_kind.append("cur")
            elif r < 0.8:
                seat_kind.append(("net", pool[int(rng.integers(len(pool)))]))
            else:
                seat_kind.append("greedy")
        if "cur" not in seat_kind:
            seat_kind[int(rng.integers(3))] = "cur"

        # per-seat trajectory of the current policy
        traj = {s: [] for s in range(3)}
        x = env.obs()
        while not env.done:
            p = env.player
            mask = env.legal_mask()
            if not mask.any():
                break
            kind = seat_kind[p]
            if kind == "cur":
                a, logp, v = sample_action(cur, x, mask, rng)
                traj[p].append([x, mask, a, logp, v])
            elif kind == "greedy":
                a = greedy_action(env, rng)
            else:  # frozen net opponent
                a, _, _ = sample_action(kind[1], x, mask, rng)
            x, r, done = env.step(a)
            if done:
                break
        stats["plies"].append(env.ply)
        if getattr(env, "winner", None) is not None:
            stats["decided"] += 1
        # rewards
        rew = np.full(3, -0.5, dtype=np.float32)
        w = getattr(env, "winner", None)
        if w is None:
            rew[:] = -0.1
        else:
            rew[w] = 1.0
        for s in range(3):
            if seat_kind[s] != "cur" or not traj[s]:
                continue
            values = [row[4] for row in traj[s]]
            adv, ret = gae(values, float(rew[s]), gamma, lam)
            for k, row in enumerate(traj[s]):
                batch["x"].append(row[0])
                batch["mask"].append(row[1])
                batch["a"].append(row[2])
                batch["logp"].append(row[3])
                batch["adv"].append(adv[k])
                batch["ret"].append(ret[k])
    return batch, stats


def ppo_update(net, batch, epochs, minibatch, clip, lr, ent_coef, vf_coef):
    X = np.asarray(batch["x"], dtype=np.float32)
    M = np.asarray(batch["mask"])
    A = np.asarray(batch["a"], dtype=np.int64)
    OLDLP = np.asarray(batch["logp"], dtype=np.float32)
    ADV = np.asarray(batch["adv"], dtype=np.float32)
    RET = np.asarray(batch["ret"], dtype=np.float32)
    n = len(A)
    if n == 0:
        return
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch):
            mb = idx[start:start + minibatch]
            grads = {k: np.zeros_like(v) for k, v in net.P.items()}
            for j in mb:
                x, mask, a = X[j], M[j], int(A[j])
                logits, vraw, (xi, h1, h2) = net.forward(x)
                lg = logits.copy()
                lg[~mask] = -1e9
                lg -= lg.max()
                e = np.exp(lg)
                pi = e / e.sum()
                v = np.tanh(vraw)
                logp = np.log(pi[a] + 1e-12)
                ratio = np.exp(logp - OLDLP[j])
                adv = ADV[j]
                # clipped surrogate; gradient of -min(...) wrt logits
                unclipped = ratio * adv
                clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
                use_unclipped = unclipped <= clipped
                # d(logp)/d(logits) = onehot(a) - pi ; d(ratio)=ratio*d(logp)
                dpi_dlogits = -pi.copy()
                dpi_dlogits[a] += 1.0
                if use_unclipped:
                    dsurr = ratio * adv  # d/dlogp of ratio*adv
                else:
                    # clipped region: if ratio within band, same; if clamped, 0
                    if (1 - clip) <= ratio <= (1 + clip):
                        dsurr = ratio * adv
                    else:
                        dsurr = 0.0
                dlogits_pol = -(dsurr) * dpi_dlogits  # maximise surrogate -> minimise neg
                # entropy bonus: H = -sum pi log pi ; dH/dlogits = -pi*(logpi+H)
                logpi = np.log(np.clip(pi, 1e-12, 1))
                Hent = -(pi * logpi).sum()
                dlogits_ent = -ent_coef * (-pi * (logpi + Hent))
                dlogits = dlogits_pol + dlogits_ent
                dlogits[~mask] = 0.0
                # value loss: vf_coef*(v-ret)^2 ; v=tanh(vraw)
                dv = vf_coef * 2 * (v - RET[j]) * (1 - v * v)
                # backprop
                grads['Wp'] += np.outer(dlogits, h2)
                grads['bp'] += dlogits
                grads['Wv'] += dv * h2
                grads['bv'][0] += dv
                dh2 = net.P['Wp'].T @ dlogits + dv * net.P['Wv']
                dz2 = dh2 * (1 - h2 * h2)
                grads['W2'] += np.outer(dz2, h1)
                grads['b2'] += dz2
                dh1 = net.P['W2'].T @ dz2
                dz1 = dh1 * (1 - h1 * h1)
                grads['W1'] += np.outer(dz1, xi)
                grads['b1'] += dz1
            for k in grads:
                grads[k] /= max(len(mb), 1)
            net.update(grads, lr)


def evaluate(net, opp, ngames, beak_start=(4, 4, 4), value_search=False, opp_net=None):
    """Learned policy (argmax, optionally with 1-ply value search) in each seat vs
    two opponents. Returns win share of decided games (uniform baseline 1/3)."""
    rng = random.Random(777)
    wins = dec = 0
    for g in range(ngames):
        seat = g % 3
        env = Env(beak_start=beak_start)
        while not env.done:
            p = env.player
            mask = env.legal_mask()
            if not mask.any():
                break
            if p == seat:
                a = choose_policy_action(net, env, value_search)
            elif opp == "greedy":
                a = greedy_action(env, rng)
            elif opp == "random":
                acts = env.legal_actions()
                a = acts[rng.randint(0, len(acts) - 1)]
            else:  # net opponent
                oa = choose_policy_action(opp_net, env, False)
                a = oa
            env.step(a)
        w = getattr(env, "winner", None)
        if w is not None:
            dec += 1
            if w == seat:
                wins += 1
    return wins / max(dec, 1)


def choose_policy_action(net, env, value_search):
    p = env.player
    mask = env.legal_mask()
    if not value_search:
        pi, _, _, _ = net.policy(env.obs(), mask)
        pi = pi * mask
        return int(np.argmax(pi))
    # 1-ply value search over legal moves, value from mover's perspective.
    best_a, best_v = None, -1e9
    for a in env.legal_actions():
        board, beaks = env._simulate(p, a)
        w = H.winner_after(board, p)
        if w == p:
            return a
        if w is not None:
            v = -1e6
        else:
            x = H.compute_features(board, beaks, p)
            _, vv, _ = net.forward(x)
            v = vv
        if v > best_v:
            best_v, best_a = v, a
    return best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--pool-every", type=int, default=15)
    ap.add_argument("--pool-max", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "web", "weights", "policy.json"))
    ap.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "training_report.json"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = Net(seed=args.seed)
    pool = []
    first_ckpt = None
    history = []
    t0 = time.time()

    for it in range(1, args.iters + 1):
        batch, stats = collect(net, pool, rng, args.games_per_iter,
                               (4, 4, 4), args.gamma, args.lam)
        ppo_update(net, batch, args.epochs, args.minibatch, args.clip,
                   args.lr, args.ent_coef, args.vf_coef)
        if it % args.pool_every == 0:
            snap = Net(seed=0)
            snap.P = {k: v.copy() for k, v in net.P.items()}
            pool.append(snap)
            if len(pool) > args.pool_max:
                pool.pop(0)
            if first_ckpt is None:
                first_ckpt = snap
        if it % args.eval_every == 0 or it == args.iters:
            wr = evaluate(net, "random", args.eval_games)
            wg = evaluate(net, "greedy", args.eval_games)
            wgs = evaluate(net, "greedy", args.eval_games, value_search=True)
            wc = evaluate(net, "net", args.eval_games, opp_net=first_ckpt) if first_ckpt else float("nan")
            dec = 100 * stats["decided"] / args.games_per_iter
            pl = np.mean(stats["plies"])
            rec = dict(iter=it, decided=dec, plies=float(pl), vs_random=wr,
                       vs_greedy=wg, vs_greedy_search=wgs, vs_first_ckpt=wc,
                       secs=time.time() - t0)
            history.append(rec)
            print(f"it {it:4d} | selfplay decided {dec:3.0f}% len {pl:4.0f} | "
                  f"vs random {100*wr:3.0f}% | vs greedy {100*wg:3.0f}% "
                  f"(+search {100*wgs:3.0f}%) | vs first-ckpt {100*wc:3.0f}% | "
                  f"{time.time()-t0:5.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(net.export(), f)
    with open(args.report, "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print("saved weights ->", args.out)
    print("saved report  ->", args.report)


if __name__ == "__main__":
    main()

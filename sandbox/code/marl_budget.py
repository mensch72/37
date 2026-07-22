"""Budget-6 variant of "37" as MARL env + probes for forced draws/wins.

Obs: 111 board one-hots (canonicalised) + 3 hand sizes (own/next/prev, /10).
Rules: adds need hand>0; deaths & removals refund; births draw from hand,
minting (permanent supply growth) when hand empty. Draw on repetition
(board+hands+player seen twice) or 200 plies.
Probes:
  base     - shared-policy self-play (rewards +1/-0.5/-0.5, draw -0.1)
  spoiler  - one seat, reward draw +1 / decisive -0.5, vs frozen base
  exploiter- one seat, base rewards, vs frozen base

Note: this is the *budget* economy, which was ultimately rejected in favour of
the beak economy (see ../results/FINDINGS.md section 5). It is kept because the
environment design, the canonicalisation and the spoiler/exploiter probes are the
reusable parts.
"""
import numpy as np, random, collections, time, sys

src = open('hex3life.py').read().replace('R = 4', 'R = 3').split("if __name__")[0]
G = {}; exec(src, G)
N, EMPTY, IDX, CELLS, NBRS = G['N'], G['EMPTY'], G['IDX'], G['CELLS'], G['NBRS']
ROT = lambda c: (c[2], c[0], c[1])
PERM = np.zeros((3, N), dtype=int)
for i, c in enumerate(CELLS):
    r = c
    for p in range(3):
        PERM[p][i] = IDX[r]; r = ROT(r)
BOARD0 = [EMPTY]*N
_cells = [(1, 0, -1), (-1, 0, 1)]
for _p in range(3):
    for _c in _cells: BOARD0[IDX[_c]] = _p
    _cells = [ROT(c) for c in _cells]

def trans(board, pools):
    new = board[:]; pools = pools[:]
    births = []
    for i in range(N):
        cnt = [0, 0, 0]
        for j in NBRS[i]:
            if board[j] != EMPTY: cnt[board[j]] += 1
        n = sum(cnt)
        if board[i] != EMPTY:
            if not (1 <= n <= 4):
                new[i] = EMPTY; pools[board[i]] += 1
        elif n == 4:
            s = sorted(range(3), key=lambda q: -cnt[q])
            c = s[2] if (cnt[s[0]] == 2 and cnt[s[1]] == 2) else s[0]
            births.append((i, c))
    for i, c in births:
        if pools[c] > 0: pools[c] -= 1     # else: mint (no pool change)
        new[i] = c
    return new, pools

class Env:
    def reset(self):
        self.board = BOARD0[:]; self.pools = [4, 4, 4]
        self.t = 0; self.seen = collections.Counter()
        return self.obs()
    @property
    def player(self): return self.t % 3
    def obs(self):
        p = self.player
        o = np.zeros(3*N + 3, dtype=np.float64)
        for i in range(N):
            v = self.board[PERM[p][i]]
            if v != EMPTY: o[((v - p) % 3)*N + i] = 1.0
        for k in range(3):
            o[3*N + k] = self.pools[(p + k) % 3] / 10.0
        return o
    def mask(self):
        p = self.player
        m = np.zeros(2*N, dtype=bool)
        can_add = self.pools[p] > 0
        for i in range(N):
            v = self.board[PERM[p][i]]
            if v == EMPTY and can_add: m[i] = True
            elif v == p: m[N + i] = True
        return m
    def step(self, a):
        p = self.player
        real = PERM[p][a % N]
        if a < N: self.board[real] = p; self.pools[p] -= 1
        else:     self.board[real] = EMPTY; self.pools[p] += 1
        self.board, self.pools = trans(self.board, self.pools)
        self.t += 1
        if G['winner_after'](self.board, p) is not None:
            w = next(q for q in range(3) if G['connected'](self.board, q))
            r = np.full(3, -0.5); r[w] = 1.0
            return None, r, True, 'win'
        key = (tuple(self.board), tuple(self.pools), self.t % 3)
        self.seen[key] += 1
        if self.seen[key] >= 2 or self.t >= 200:
            return None, np.full(3, -0.1), True, 'draw'
        return self.obs(), np.zeros(3), False, None

class Net:
    def __init__(self, nin=3*N+3, nh=128, nout=2*N, seed=0):
        rs = np.random.RandomState(seed)
        self.P = dict(W1=rs.randn(nh, nin)*np.sqrt(2/nin), b1=np.zeros(nh),
                      W2=rs.randn(nout, nh)*0.01*np.sqrt(2/nh), b2=np.zeros(nout),
                      w3=rs.randn(nh)*0.01*np.sqrt(2/nh), b3=np.zeros(1))
        self.adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.P.items()}
        self.ts = 0
    def forward(self, x, mask):
        h = np.tanh(self.P['W1'] @ x + self.P['b1'])
        lg = self.P['W2'] @ h + self.P['b2']; lg[~mask] = -1e9
        z = lg - lg.max(); e = np.exp(z); pi = e / e.sum()
        v = float(self.P['w3'] @ h) + float(self.P['b3'][0])
        return pi, v, h
    def update(self, g, lr):
        self.ts += 1
        c1 = 1 - 0.9**self.ts; c2 = 1 - 0.999**self.ts
        for k, gr in g.items():
            m, v = self.adam[k]
            m *= 0.9; m += 0.1*gr; v *= 0.999; v += 0.001*gr*gr
            self.P[k] -= lr*(m/c1)/(np.sqrt(v/c2)+1e-8)

def zerog(net): return {k: np.zeros_like(v) for k, v in net.P.items()}

def backprop(net, grads, x, m, a, pi, v, h, A, target_v, beta_ent, vcoef):
    dlog = pi * A; dlog[a] -= A
    logpi = np.log(np.clip(pi, 1e-12, 1)); H = -(pi*logpi).sum()
    dlog += beta_ent * pi * (logpi + H); dlog[~m] = 0.0
    dv = vcoef * 2 * (v - target_v)
    grads['W2'] += np.outer(dlog, h); grads['b2'] += dlog
    grads['w3'] += dv*h; grads['b3'] += dv
    dh = net.P['W2'].T @ dlog + dv*net.P['w3']
    dz = dh*(1-h*h)
    grads['W1'] += np.outer(dz, x); grads['b1'] += dz

def train_base(episodes, batch=24, lr=7e-4, gamma=0.985, beta=0.012, vc=0.25, seed=0):
    rng = np.random.RandomState(seed); net = Net(seed=seed); env = Env()
    stats = collections.deque(maxlen=300); ep = 0; t0 = time.time()
    while ep < episodes:
        grads = zerog(net); nst = 0
        for _ in range(batch):
            traj = [[], [], []]; x = env.reset()
            while True:
                p = env.player; m = env.mask()
                pi, v, h = net.forward(x, m)
                a = rng.choice(2*N, p=pi)
                traj[p].append((x, m, a, pi, v, h))
                x, r, done, kind = env.step(a)
                if done: break
            stats.append(kind == 'win')
            for p in range(3):
                T = len(traj[p])
                for k in range(T):
                    xx, mm, aa, pp, vv, hh = traj[p][k]
                    Rk = (gamma**(T-1-k))*r[p]
                    backprop(net, grads, xx, mm, aa, pp, vv, hh, Rk-vv, Rk, beta, vc)
                    nst += 1
            ep += 1
        for k in grads: grads[k] /= max(nst, 1)
        net.update(grads, lr)
        if ep % 2500 < batch:
            print(f"  base ep {ep}: selfplay decided {100*np.mean(stats):.0f}% "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return net

def train_seatwise(frozen, episodes, mode, batch=24, lr=7e-4, gamma=0.985,
                   beta=0.012, vc=0.25, seed=1):
    """mode='spoiler': reward draw +1, decisive -0.5 (for learner seat).
       mode='exploit': win +1, lose -0.5, draw -0.1."""
    rng = np.random.RandomState(seed); net = Net(seed=seed+10)
    net.P = {k: v.copy() for k, v in frozen.P.items()}   # warm start
    net.adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in net.P.items()}
    env = Env(); ep = 0; res = collections.deque(maxlen=300); t0 = time.time()
    while ep < episodes:
        grads = zerog(net); nst = 0
        for _ in range(batch):
            seat = ep % 3
            traj = []; x = env.reset()
            while True:
                p = env.player; m = env.mask()
                use = net if p == seat else frozen
                pi, v, h = use.forward(x, m)
                a = rng.choice(2*N, p=pi)
                if p == seat: traj.append((x, m, a, pi, v, h))
                x, r, done, kind = env.step(a)
                if done: break
            if mode == 'spoiler':
                R = 1.0 if kind == 'draw' else -0.5
                res.append(kind == 'draw')
            else:
                R = r[seat]
                res.append(r[seat] == 1.0)
            T = len(traj)
            for k in range(T):
                xx, mm, aa, pp, vv, hh = traj[k]
                Rk = (gamma**(T-1-k))*R
                backprop(net, grads, xx, mm, aa, pp, vv, hh, Rk-vv, Rk, beta, vc)
                nst += 1
            ep += 1
        for k in grads: grads[k] /= max(nst, 1)
        net.update(grads, lr)
        if ep % 2500 < batch:
            print(f"  {mode} ep {ep}: success {100*np.mean(res):.0f}% "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return net

def head_to_head(learner, frozen, n, mode, seed=999):
    rng = np.random.RandomState(seed); env = Env()
    succ = 0; outc = collections.Counter()
    for g in range(n):
        seat = g % 3; x = env.reset()
        while True:
            p = env.player
            use = learner if p == seat else frozen
            pi, _, _ = use.forward(x, env.mask())
            a = rng.choice(2*N, p=pi)
            x, r, done, kind = env.step(a)
            if done: break
        outc[kind] += 1
        if mode == 'spoiler' and kind == 'draw': succ += 1
        if mode == 'exploit' and kind == 'win' and r[seat] == 1.0: succ += 1
    return succ/n, outc

if __name__ == '__main__':
    eps = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    net = train_base(eps)
    np.savez('budget_base.npz', **net.P)
    dr, outc = head_to_head(net, net, 300, 'spoiler')
    print(f"base vs base: draw rate {100*dr:.0f}% ({dict(outc)})")

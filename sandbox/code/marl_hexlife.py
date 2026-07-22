"""3-player hex-Life MARL environment + self-play A2C (numpy, manual grads).

Env: R=3 board (37 cells), B4 (majority / 2-2 -> third), S1-4, tang-d1 dominoes,
connection-only win. Action space: 74 = add@i (0..36) | remove@i (37..73), masked.
Repetition (position+player seen twice) or 200 plies -> draw.
Rewards (terminal only): winner +1, losers -0.5, draw -0.1 for all.

Symmetry: seat p's observation is rotated by rho^{-p} (rho = 120 deg) and
colour-relabelled so channel 0 = self, 1 = next seat, 2 = previous seat.
One shared policy therefore plays all seats (fully symmetric self-play).
"""
import numpy as np, random, collections, sys, time

# ---------- game core (R=3) ----------
src = open('hex3life.py').read().replace('R = 4', 'R = 3').split("if __name__")[0]
G = {}; exec(src, G)
N, EMPTY, IDX, CELLS, NBRS = G['N'], G['EMPTY'], G['IDX'], G['CELLS'], G['NBRS']
assert N == 37

ROT = lambda c: (c[2], c[0], c[1])
PERM = np.zeros((3, N), dtype=int)          # PERM[p][canon_i] = real cell idx
for i, c in enumerate(CELLS):
    r = c
    for p in range(3):
        PERM[p][i] = IDX[r]
        r = ROT(r)                           # rho^p applied to canonical cell

class Env:
    def __init__(self):
        self.reset()
    def reset(self):
        self.board = G['domino_init']('tang', 1)
        self.t = 0
        self.seen = collections.Counter()
        self.done = False
        return self.obs()
    @property
    def player(self):
        return self.t % 3
    def obs(self):
        """(3*N,) float: canonical one-hot of {self, next, prev} occupancy."""
        p = self.player
        o = np.zeros((3, N), dtype=np.float32)
        for i in range(N):
            v = self.board[PERM[p][i]]
            if v != EMPTY:
                o[(v - p) % 3, i] = 1.0
        return o.reshape(-1)
    def mask(self):
        p = self.player
        m = np.zeros(2 * N, dtype=bool)
        for i in range(N):
            v = self.board[PERM[p][i]]
            if v == EMPTY: m[i] = True
            elif v == p:   m[N + i] = True
        return m
    def step(self, a):
        """a in [0,74): canonical action. Returns (obs, rewards[3], done)."""
        p = self.player
        real = PERM[p][a % N]
        mv = ('add', real) if a < N else ('rem', real)
        self.board, _ = G['apply_move'](self.board, p, mv, 4, 'strict', frozenset())
        self.t += 1
        if G['winner_after'](self.board, p) is not None:
            w = next(q for q in range(3) if G['connected'](self.board, q))
            r = np.full(3, -0.5); r[w] = 1.0
            self.done = True; return None, r, True
        key = (tuple(self.board), self.t % 3)
        self.seen[key] += 1
        if self.seen[key] >= 2 or self.t >= 200:
            self.done = True; return None, np.full(3, -0.1), True
        return self.obs(), np.zeros(3), False

# ---------- tiny MLP with policy + value heads ----------
class Net:
    def __init__(self, nin=3*N, nh=128, nout=2*N, seed=0):
        rs = np.random.RandomState(seed)
        s1 = np.sqrt(2.0/nin); s2 = np.sqrt(2.0/nh)
        self.P = dict(
            W1=rs.randn(nh, nin)*s1, b1=np.zeros(nh),
            W2=rs.randn(nout, nh)*s2*0.01, b2=np.zeros(nout),
            w3=rs.randn(nh)*s2*0.01, b3=np.zeros(1))
        self.adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.P.items()}
        self.tstep = 0
    def forward(self, x, mask):
        h = np.tanh(self.P['W1'] @ x + self.P['b1'])
        logits = self.P['W2'] @ h + self.P['b2']
        logits[~mask] = -1e9
        z = logits - logits.max()
        e = np.exp(z); pi = e / e.sum()
        v = float(self.P['w3'] @ h) + float(self.P['b3'][0])
        return pi, v, h
    def update(self, grads, lr):
        self.tstep += 1
        b1c = 1 - 0.9**self.tstep; b2c = 1 - 0.999**self.tstep
        for k, g in grads.items():
            m, v = self.adam[k]
            m *= 0.9;   m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            self.P[k] -= lr * (m/b1c) / (np.sqrt(v/b2c) + 1e-8)

def zeros_like_params(net):
    return {k: np.zeros_like(v) for k, v in net.P.items()}

# ---------- training ----------
def run_episode(env, net, rng, sample=True):
    """Returns per-seat trajectories [(x, mask, a, pi, v, h)], rewards[3], plies."""
    traj = [[], [], []]
    x = env.reset()
    while True:
        p = env.player
        m = env.mask()
        pi, v, h = net.forward(x, m)
        a = rng.choice(2*N, p=pi) if sample else int(np.argmax(pi))
        traj[p].append((x, m, a, pi, v, h))
        x, r, done = env.step(a)
        if done:
            return traj, r, env.t

def train(episodes=4000, batch=24, lr=7e-4, gamma=0.985, beta_ent=0.012,
          vcoef=0.25, seed=0, eval_every=500):
    rng = np.random.RandomState(seed)
    net = Net(seed=seed)
    env = Env()
    stats = collections.deque(maxlen=200)
    t0 = time.time()
    ep = 0
    while ep < episodes:
        grads = zeros_like_params(net)
        nsteps = 0
        for _ in range(batch):
            traj, rew, plies = run_episode(env, net, rng)
            stats.append((rew.max() == 1.0, plies))
            for p in range(3):
                T = len(traj[p])
                R = rew[p]
                for k in reversed(range(T)):
                    x, m, a, pi, v, h = traj[p][k]
                    Rk = (gamma ** (T-1-k)) * R
                    A = Rk - v
                    # policy grad on logits: (pi - onehot)*A  (masked logits fixed)
                    dlog = pi * A; dlog[a] -= A
                    # entropy bonus
                    logpi = np.log(np.clip(pi, 1e-12, 1))
                    H = -(pi * logpi).sum()
                    dlog += beta_ent * pi * (logpi + H)
                    dlog[~m] = 0.0
                    dv = vcoef * 2 * (v - Rk)
                    # backprop
                    grads['W2'] += np.outer(dlog, h); grads['b2'] += dlog
                    grads['w3'] += dv * h;            grads['b3'] += dv
                    dh = net.P['W2'].T @ dlog + dv * net.P['w3']
                    dz = dh * (1 - h*h)
                    grads['W1'] += np.outer(dz, x);   grads['b1'] += dz
                    nsteps += 1
            ep += 1
        for k in grads: grads[k] /= max(nsteps, 1)
        net.update(grads, lr)
        if ep % eval_every < batch:
            wr = evaluate(net, 'random', 45); wg = evaluate(net, 'greedy', 21)
            dec = 100*np.mean([d for d, _ in stats])
            pl = np.mean([l for _, l in stats])
            print(f"ep {ep:5d} | selfplay decided {dec:.0f}%, len {pl:.0f} | "
                  f"vs 2 random: {100*wr:.0f}% | vs 2 greedy: {100*wg:.0f}% | "
                  f"{time.time()-t0:.0f}s", flush=True)
    return net

def evaluate(net, opp, ngames):
    """Learned policy in each seat vs two scripted opponents; returns win share
    of decided games (uniform baseline = 1/3)."""
    rng = np.random.RandomState(123)
    pyrng = random.Random(123)
    wins = dec = 0
    for g in range(ngames):
        seat = g % 3
        env = Env()
        x = env.obs()
        while True:
            p = env.player
            if p == seat:
                pi, _, _ = net.forward(x, env.mask())
                a = int(np.argmax(pi))
                x, r, done = env.step(a)
            else:
                f = G['greedy_player'] if opp == 'greedy' else G['random_player']
                mv = f(env.board, p, 4, pyrng, 'strict', frozenset())
                canon = int(np.where(PERM[p] == mv[1])[0][0])
                a = canon if mv[0] == 'add' else N + canon
                x, r, done = env.step(a)
            if done:
                if r.max() == 1.0:
                    dec += 1
                    if r[seat] == 1.0: wins += 1
                break
    return wins / max(dec, 1)

if __name__ == '__main__':
    eps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    net = train(episodes=eps)
    print("final: vs 2 random", evaluate(net, 'random', 90),
          "| vs 2 greedy", evaluate(net, 'greedy', 45))
    np.savez('hexlife_policy.npz', **net.P)

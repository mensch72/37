"""Environment, features and network for training the learned player of "37".

This mirrors web/js/engine.js and web/js/features.js exactly so that a policy
trained here runs unchanged in the browser. The growth step is the final
beak-economy rule (validated against sandbox/code/beak3.py by the Node tests); the
economy, superko-as-illegal, elimination and connection win match the JS Game.

The observation is canonicalised (marl_hexlife.py trick) and augmented with the
path-distance features FINDINGS section 7 asks for. Action space is 2N = 74
(drop@i | pick@i) with a legality mask.
"""
import collections
import heapq
import numpy as np

R = 3
EMPTY = -1
SURV_MAX_VOLATILE = 4  # S1-4 ("volatile")
SURV_MAX_CALM = 5      # S1-5 ("calm", the default)
SURV_MAX = SURV_MAX_VOLATILE  # legacy alias (the reference rule)
DEFAULT_SURV_MAX = SURV_MAX_CALM
VARIANTS = {"calm": SURV_MAX_CALM, "volatile": SURV_MAX_VOLATILE}
DEFAULT_CAP = 6
DIRS = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]

CELLS = [(x, y, z) for x in range(-R, R + 1) for y in range(-R, R + 1)
         for z in [-x - y] if abs(z) <= R]
N = len(CELLS)
assert N == 37
IDX = {c: i for i, c in enumerate(CELLS)}
NBRS = [[IDX[(c[0] + d[0], c[1] + d[1], c[2] + d[2])] for d in DIRS
         if (c[0] + d[0], c[1] + d[1], c[2] + d[2]) in IDX] for c in CELLS]

ROT = lambda c: (c[2], c[0], c[1])
PERM = np.zeros((3, N), dtype=int)
for i, c in enumerate(CELLS):
    r = c
    for p in range(3):
        PERM[p][i] = IDX[r]
        r = ROT(r)
INV_PERM = np.zeros((3, N), dtype=int)
for p in range(3):
    for i in range(N):
        INV_PERM[p][PERM[p][i]] = i

SIDES = []
for ax in range(3):
    plus = [i for i, c in enumerate(CELLS) if c[ax] == R]
    minus = [i for i, c in enumerate(CELLS) if c[ax] == -R]
    SIDES.append((plus, minus))


def initial_board():
    board = [EMPTY] * N
    cells = [(1, 0, -1), (-1, 0, 1)]
    for p in range(3):
        for c in cells:
            board[IDX[c]] = p
        cells = [ROT(c) for c in cells]
    return board


def growth(board, surv_max=DEFAULT_SURV_MAX):
    new = board[:]
    births = []
    for i in range(N):
        cnt = [0, 0, 0]
        for j in NBRS[i]:
            v = board[j]
            if v != EMPTY:
                cnt[v] += 1
        n = cnt[0] + cnt[1] + cnt[2]
        if board[i] != EMPTY:
            if not (1 <= n <= surv_max):
                new[i] = EMPTY
        elif n == 4:
            s = sorted(range(3), key=lambda q: -cnt[q])
            if cnt[s[0]] == 2 and cnt[s[1]] == 2:
                c = s[2]
            else:
                c = s[0]
            births.append((i, c))
    for i, c in births:
        new[i] = c
    return new


def connected(board, p):
    plus, minus = SIDES[p]
    minus_set = set(minus)
    stack = [i for i in plus if board[i] == p]
    seen = set(stack)
    while stack:
        i = stack.pop()
        if i in minus_set:
            return True
        for j in NBRS[i]:
            if board[j] == p and j not in seen:
                seen.add(j)
                stack.append(j)
    return False


def winner_after(board, mover):
    conn = [connected(board, p) for p in range(3)]
    if any(conn):
        return mover if conn[mover] else conn.index(True)
    return None


def canonical_board(board, p):
    cb = [EMPTY] * N
    for i in range(N):
        v = board[PERM[p][i]]
        cb[i] = EMPTY if v == EMPTY else (v - p) % 3
    return cb


def _dist_field(cb, c, sources):
    dist = [10 ** 9] * N
    pq = []
    for i in sources:
        if cb[i] == c:
            d = 0
        elif cb[i] == EMPTY:
            d = 1
        else:
            continue
        if d < dist[i]:
            dist[i] = d
            heapq.heappush(pq, (d, i))
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        for j in NBRS[i]:
            if cb[j] == c:
                nd = d
            elif cb[j] == EMPTY:
                nd = d + 1
            else:
                continue
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(pq, (nd, j))
    return dist


def _path_best(cb, c):
    d_plus = _dist_field(cb, c, SIDES[c][0])
    best = min((d_plus[i] for i in SIDES[c][1]), default=10 ** 9)
    return d_plus, best


OCC = 3 * N
EMPTYF = N
PATHF = N
SCALARS = 6
FEAT_LEN = OCC + EMPTYF + PATHF + SCALARS  # 191


def compute_features(board, beaks, p):
    cb = canonical_board(board, p)
    f = np.zeros(FEAT_LEN, dtype=np.float32)
    for i in range(N):
        v = cb[i]
        if v == EMPTY:
            f[OCC + i] = 1.0
        else:
            f[v * N + i] = 1.0
    d_plus, best = _path_best(cb, 0)
    d_minus = _dist_field(cb, 0, SIDES[0][1])
    base = OCC + EMPTYF
    if best < 10 ** 9:
        for i in range(N):
            if cb[i] not in (1, 2) and d_plus[i] + d_minus[i] == best:
                f[base + i] = 1.0
    sb = base + PATHF
    _, pd_next = _path_best(cb, 1)
    _, pd_prev = _path_best(cb, 2)
    cap = lambda d: d if d < 10 ** 9 else 20
    f[sb + 0] = 1.0 / (1.0 + cap(best))
    f[sb + 1] = 1.0 / (1.0 + cap(pd_next))
    f[sb + 2] = 1.0 / (1.0 + cap(pd_prev))
    f[sb + 3] = beaks[p] / 8.0
    f[sb + 4] = beaks[(p + 1) % 3] / 8.0
    f[sb + 5] = beaks[(p + 2) % 3] / 8.0
    return f


def canonical_mask(board, beaks, p, cap=10 ** 9):
    cb = canonical_board(board, p)
    m = np.zeros(2 * N, dtype=bool)
    can_drop = beaks[p] > 0
    can_pick = beaks[p] < cap
    for i in range(N):
        if can_drop and cb[i] == EMPTY:
            m[i] = True
        elif can_pick and cb[i] == 0:
            m[N + i] = True
    return m


def canonical_action_to_real(a, p):
    cell = int(PERM[p][a % N])
    return ("drop", cell) if a < N else ("pick", cell)


def position_key(board, beaks, player):
    return (tuple(board), tuple(beaks), player)


class Env:
    """Beak-economy environment with superko-as-illegal and elimination.

    Rewards are terminal only: winner +1, other live players -0.5, and a small
    negative for the rare stuck/timeout case. Turns are taken by live players in
    seat order, skipping eliminated seats.
    """

    def __init__(self, beak_start=(4, 4, 4), cap=DEFAULT_CAP, max_plies=200,
                 surv_max=DEFAULT_SURV_MAX):
        self.beak_start = tuple(beak_start)
        self.cap = cap
        self.max_plies = max_plies
        self.surv_max = surv_max
        self.reset()

    def reset(self):
        self.board = initial_board()
        self.beaks = list(self.beak_start)
        self.to_move = 0
        self.alive = [True, True, True]
        self.ply = 0
        self.history = {position_key(self.board, self.beaks, self.to_move)}
        self.done = False
        self._skip_dead()
        return self.obs()

    @property
    def player(self):
        return self.to_move

    def obs(self):
        return compute_features(self.board, self.beaks, self.to_move)

    def cell_count(self, p):
        return sum(1 for v in self.board if v == p)

    def is_eliminated(self, p):
        return self.beaks[p] == 0 and self.cell_count(p) == 0

    def is_blocked(self, p):
        # A player whose whole side is occupied by other colours can never connect
        # (issue #12). A side counts as blocked only when every cell on it is
        # occupied and none is p's colour.
        for side in SIDES[p]:
            if all(self.board[i] != EMPTY and self.board[i] != p for i in side):
                return True
        return False

    def _simulate(self, p, a):
        board = self.board[:]
        beaks = self.beaks[:]
        real = int(PERM[p][a % N])
        if a < N:
            board[real] = p
            beaks[p] -= 1
        else:
            board[real] = EMPTY
            beaks[p] += 1
        return growth(board, self.surv_max), beaks

    def _next_live(self, board, beaks, frm):
        count = [0, 0, 0]
        for v in board:
            if v != EMPTY:
                count[v] += 1
        alive_now = [beaks[q] > 0 or count[q] > 0 for q in range(3)]
        for step in range(1, 4):
            q = (frm + step) % 3
            if alive_now[q]:
                return q
        return frm

    def legal_actions(self):
        p = self.to_move
        m = canonical_mask(self.board, self.beaks, p, self.cap)
        out = []
        for a in range(2 * N):
            if not m[a]:
                continue
            board, beaks = self._simulate(p, a)
            if winner_after(board, p) is not None:
                out.append(a)
                continue
            nxt = self._next_live(board, beaks, p)
            if position_key(board, beaks, nxt) not in self.history:
                out.append(a)
        return out

    def legal_mask(self):
        m = np.zeros(2 * N, dtype=bool)
        for a in self.legal_actions():
            m[a] = True
        return m

    def _skip_dead(self):
        guard = 0
        while not self.done and guard < 6:
            guard += 1
            if not self.alive[self.to_move]:
                pass  # already out (blocked-side or a prior elimination)
            elif self.is_eliminated(self.to_move):
                self.alive[self.to_move] = False
            elif len(self.legal_actions()) == 0:
                self.alive[self.to_move] = False
            else:
                return
            if sum(self.alive) <= 1:
                self.done = True
                self.winner = self.alive.index(True) if any(self.alive) else None
                return
            for step in range(1, 4):
                q = (self.to_move + step) % 3
                if self.alive[q]:
                    self.to_move = q
                    break

    def step(self, a):
        p = self.to_move
        real = int(PERM[p][a % N])
        if a < N:
            self.board[real] = p
            self.beaks[p] -= 1
        else:
            self.board[real] = EMPTY
            self.beaks[p] += 1
        self.board = growth(self.board, self.surv_max)
        self.ply += 1
        for q in range(3):
            if self.alive[q] and (self.is_eliminated(q) or self.is_blocked(q)):
                self.alive[q] = False
        w = winner_after(self.board, p)
        if w is not None:
            self.done = True
            self.winner = w
            r = np.full(3, -0.5)
            r[w] = 1.0
            return None, r, True
        if sum(self.alive) <= 1:
            self.done = True
            self.winner = self.alive.index(True) if any(self.alive) else None
            r = np.full(3, -0.5)
            if self.winner is not None:
                r[self.winner] = 1.0
            return None, r, True
        if self.ply >= self.max_plies:
            self.done = True
            self.winner = None
            return None, np.full(3, -0.1), True
        self.to_move = self._next_live(self.board, self.beaks, p)
        self.history.add(position_key(self.board, self.beaks, self.to_move))
        self._skip_dead()
        if self.done:
            r = np.full(3, -0.5)
            if getattr(self, "winner", None) is not None:
                r[self.winner] = 1.0
            return None, r, True
        return self.obs(), np.zeros(3), False


# ---- greedy 1-ply baseline (the FINDINGS floor), real-space ----
def _path_dist_real(board, p):
    plus, minus = SIDES[p]
    minus_set = set(minus)
    dist = [10 ** 9] * N
    pq = []
    for i in plus:
        if board[i] == p:
            dist[i] = 0
            heapq.heappush(pq, (0, i))
        elif board[i] == EMPTY:
            dist[i] = 1
            heapq.heappush(pq, (1, i))
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        if i in minus_set:
            return d
        for j in NBRS[i]:
            if board[j] == p:
                nd = d
            elif board[j] == EMPTY:
                nd = d + 1
            else:
                continue
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(pq, (nd, j))
    return 10 ** 9


def greedy_action(env, rng):
    """1-ply heuristic mirroring beak3.greedy, in canonical action space."""
    p = env.to_move
    best_a, best_s = None, None
    for a in env.legal_actions():
        board, beaks = env._simulate(p, a)
        w = winner_after(board, p)
        if w == p:
            return a
        if w is not None:
            s = -1e8
        else:
            my = _path_dist_real(board, p)
            opp = sorted(_path_dist_real(board, q) for q in range(3) if q != p)
            s = -3 * my + opp[0] + 0.3 * opp[1] + 0.01 * rng.random()
        if best_s is None or s > best_s:
            best_s, best_a = s, a
    return best_a


# ---- scripted strategy opponents found in human play (issue #12 section 2) ----
# All return a canonical action from env.legal_actions(); each isolates one of the
# strategies the old economy-blind training never saw, so the learned policy is
# forced to face them during training and is scored against them at evaluation.

RIM = [i for i, c in enumerate(CELLS) if max(abs(c[0]), abs(c[1]), abs(c[2])) == R]
RIM_SET = set(RIM)


def _own_ring_cells(p):
    cells = [(1, 0, -1), (-1, 0, 1)]
    for _ in range(p):
        cells = [ROT(c) for c in cells]
    return [IDX[c] for c in cells]


def _rim_path_dist_real(board, p):
    """Path distance for p that may only traverse EMPTY cells lying on the rim, so
    the resulting chain is routed along the boundary (an arc). Own cells anywhere
    cost 0; enemies block."""
    plus, minus = SIDES[p]
    minus_set = set(minus)
    dist = [10 ** 9] * N
    pq = []
    for i in plus:
        if board[i] == p:
            dist[i] = 0
            heapq.heappush(pq, (0, i))
        elif board[i] == EMPTY and i in RIM_SET:
            dist[i] = 1
            heapq.heappush(pq, (1, i))
    while pq:
        d, i = heapq.heappop(pq)
        if d > dist[i]:
            continue
        if i in minus_set:
            return d
        for j in NBRS[i]:
            if board[j] == p:
                nd = d
            elif board[j] == EMPTY and j in RIM_SET:
                nd = d + 1
            else:
                continue
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(pq, (nd, j))
    return 10 ** 9


def endpoint_action(env, rng):
    """Claim one cell on each of your own two sides first (a permanent foothold on
    the winning chain — the complete counter to being walled out), then play the
    path-distance heuristic."""
    p = env.to_move
    legal = env.legal_actions()
    need = [set(side) for side in SIDES[p]
            if not any(env.board[i] == p for i in side)]
    if need:
        for a in legal:
            if a < N:
                real = int(PERM[p][a])
                if any(real in s for s in need):
                    return a
    return greedy_action(env, rng)


def rim_arc_action(env, rng):
    """Route a minimum-length winning chain along the rim, covering one side of
    each opponent (a win-and-double-block arc), while still blocking opponents."""
    p = env.to_move
    best_a, best_s = None, None
    for a in env.legal_actions():
        board, beaks = env._simulate(p, a)
        w = winner_after(board, p)
        if w == p:
            return a
        if w is not None:
            s = -1e8
        else:
            my = _rim_path_dist_real(board, p)
            opp = sorted(_path_dist_real(board, q) for q in range(3) if q != p)
            s = -3 * my + opp[0] + 0.3 * opp[1] + 0.01 * rng.random()
        if best_s is None or s > best_s:
            best_s, best_a = s, a
    return best_a


def pump_action(env, rng):
    """Centre pump: pick up both own ring cells so the centre (2 + 2 of the other
    colours) births your colour, then harvest the reborn centre cell every turn.
    A finite beak cap limits the pump to a couple of extra cells (issue #12)."""
    p = env.to_move
    legal = set(env.legal_actions())
    centre = IDX[(0, 0, 0)]
    a_pick_centre = N + int(INV_PERM[p][centre])
    if env.board[centre] == p and a_pick_centre in legal:
        return a_pick_centre
    for cell in _own_ring_cells(p):
        a_pick = N + int(INV_PERM[p][cell])
        if env.board[cell] == p and a_pick in legal:
            return a_pick
    return greedy_action(env, rng)


# Registry of scripted opponents used by training and evaluation.
SCRIPTED = {
    "greedy": greedy_action,       # plain path-distance agent
    "endpoint": endpoint_action,   # endpoint-claimer
    "rim_arc": rim_arc_action,     # rim-arc blocker
    "pump": pump_action,           # centre pumper
}


# ---- network: shared tanh trunk, policy + value heads ----
class Net:
    def __init__(self, nin=FEAT_LEN, h1=192, h2=128, seed=0):
        rs = np.random.RandomState(seed)
        self.nin, self.h1, self.h2 = nin, h1, h2
        self.P = dict(
            W1=rs.randn(h1, nin) * np.sqrt(2.0 / nin), b1=np.zeros(h1),
            W2=rs.randn(h2, h1) * np.sqrt(2.0 / h1), b2=np.zeros(h2),
            Wp=rs.randn(2 * N, h2) * 0.01, bp=np.zeros(2 * N),
            Wv=rs.randn(h2) * 0.01, bv=np.zeros(1),
        )
        self.adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self.P.items()}
        self.ts = 0

    def forward(self, x):
        P = self.P
        z1 = P['W1'] @ x + P['b1']
        h1 = np.tanh(z1)
        z2 = P['W2'] @ h1 + P['b2']
        h2 = np.tanh(z2)
        logits = P['Wp'] @ h2 + P['bp']
        v = float(np.tanh(P['Wv'] @ h2 + P['bv'][0]))
        return logits, v, (x, h1, h2)

    def policy(self, x, mask):
        logits, v, cache = self.forward(x)
        lg = logits.copy()
        lg[~mask] = -1e9
        lg -= lg.max()
        e = np.exp(lg)
        pi = e / e.sum()
        return pi, v, cache, logits

    def update(self, grads, lr):
        self.ts += 1
        c1 = 1 - 0.9 ** self.ts
        c2 = 1 - 0.999 ** self.ts
        for k, g in grads.items():
            m, v = self.adam[k]
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            self.P[k] -= lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)

    def export(self):
        P = self.P
        return dict(
            nin=self.nin, h1=self.h1, h2=self.h2,
            W1=P['W1'].tolist(), b1=P['b1'].tolist(),
            W2=P['W2'].tolist(), b2=P['b2'].tolist(),
            Wp=P['Wp'].tolist(), bp=P['bp'].tolist(),
            Wv=P['Wv'].tolist(), bv=float(P['bv'][0]),
        )

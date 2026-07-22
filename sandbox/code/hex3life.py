"""3-player hexagonal Life board game simulator.

Board: hexagon of radius R (cube coords, max norm <= R).
Players 0,1,2 (R,G,B). Player i owns the side pair {axis_i = +R, axis_i = -R}.
Move: add own stone on empty cell OR remove an own stone; then one synchronous transition.
Transition: live cell survives iff 1<=n<=SURV_MAX (n = live neighbours, any colour);
empty cell with exactly 4 live neighbours is born:
  counts (a>=b>=c sorted): 4-0-0 / 3-1-0 -> majority colour; 2-1-1 -> plurality colour;
  2-2-0 -> the ABSENT third colour.
Win (checked after transition): connect your two sides with a chain of your colour,
or be the only colour left on a nonempty board. Mover wins ties.
Stalemate proxies: position repetition (position + player-to-move) or move cap.
"""
import random, itertools, collections, sys

R = 4
DIRS = [(1,-1,0),(1,0,-1),(0,1,-1),(-1,1,0),(-1,0,1),(0,-1,1)]

CELLS = [(x,y,z) for x in range(-R,R+1) for y in range(-R,R+1)
         for z in [-x-y] if abs(z) <= R]
IDX = {c:i for i,c in enumerate(CELLS)}
N = len(CELLS)
NBRS = [[IDX[(c[0]+d[0],c[1]+d[1],c[2]+d[2])] for d in DIRS
         if (c[0]+d[0],c[1]+d[1],c[2]+d[2]) in IDX] for c in CELLS]

# side membership: SIDES[p] = (set of cell idxs on +R side, set on -R side) for axis p
SIDES = []
for ax in range(3):
    plus  = {i for i,c in enumerate(CELLS) if c[ax] ==  R}
    minus = {i for i,c in enumerate(CELLS) if c[ax] == -R}
    SIDES.append((plus, minus))

EMPTY = -1

def transition(board, surv_max, iso='none', fresh=frozenset()):
    new = board[:]
    for i in range(N):
        cnt = [0,0,0]
        for j in NBRS[i]:
            if board[j] != EMPTY: cnt[board[j]] += 1
        n = sum(cnt)
        if board[i] != EMPTY:
            if iso=='strict': lo = 1
            else: lo = 0 if (iso=='none' or i in fresh) else 1
            if not (lo <= n <= surv_max): new[i] = EMPTY
        else:
            if n == 4:
                s = sorted(range(3), key=lambda p: -cnt[p])
                if cnt[s[0]] >= 3:            new[i] = s[0]        # 4-0 or 3-1
                elif cnt[s[0]]==2 and cnt[s[1]]==2:  new[i] = s[2] # 2-2 -> third colour
                else:                          new[i] = s[0]        # 2-1-1
    return new

def connected(board, p):
    plus, minus = SIDES[p]
    starts = [i for i in plus if board[i] == p]
    if not starts: return False
    seen, stack = set(starts), list(starts)
    while stack:
        i = stack.pop()
        if i in minus: return True
        for j in NBRS[i]:
            if board[j] == p and j not in seen:
                seen.add(j); stack.append(j)
    return False

def path_dist(board, p):
    """Dijkstra: cost 0 through own cells, 1 through empty, inf through enemies,
    from +side to -side. Returns cost (number of stones still needed, roughly)."""
    import heapq
    plus, minus = SIDES[p]
    dist = [10**9]*N
    pq = []
    for i in plus:
        if board[i] == p: dist[i] = 0; heapq.heappush(pq,(0,i))
        elif board[i] == EMPTY: dist[i] = 1; heapq.heappush(pq,(1,i))
    while pq:
        d,i = heapq.heappop(pq)
        if d > dist[i]: continue
        if i in minus: return d
        for j in NBRS[i]:
            if board[j] == p: nd = d
            elif board[j] == EMPTY: nd = d+1
            else: continue
            if nd < dist[j]: dist[j] = nd; heapq.heappush(pq,(nd,j))
    return 10**9

def legal_moves(board, p):
    mv = [('add',i) for i in range(N) if board[i] == EMPTY]
    mv += [('rem',i) for i in range(N) if board[i] == p]
    return mv

def apply_move(board, p, mv, surv_max, iso='none', fresh=frozenset()):
    b = board[:]
    b[mv[1]] = p if mv[0]=='add' else EMPTY
    if iso=='grace1':
        fresh = set(k for k in fresh if b[k]!=EMPTY and b[k]!=p)
        if mv[0]=='add': fresh.add(mv[1])
        fresh = frozenset(fresh)
    return transition(b, surv_max, iso, fresh), fresh

def winner_after(board, mover, t=99):
    conn = [connected(board,p) for p in range(3)]
    if any(conn):
        return mover if conn[mover] else conn.index(True)
    return None

def random_player(board, p, surv_max, rng, iso='none', fresh=frozenset()):
    return rng.choice(legal_moves(board, p))

def greedy_player(board, p, surv_max, rng, iso='none', fresh=frozenset()):
    best, bestscore = [], None
    for mv in legal_moves(board, p):
        b2, _ = apply_move(board, p, mv, surv_max, iso, fresh)
        w = winner_after(b2, p)
        if w == p: return mv
        if w is not None:            # gifting the win to someone else
            score = -10**8
        else:
            my = path_dist(b2, p)
            opp = sorted(path_dist(b2,q) for q in range(3) if q != p)
            # minimise own distance, keep nearest opponent away
            score = -3*my + 1.0*opp[0] + 0.3*opp[1] + 0.01*rng.random()
        if bestscore is None or score > bestscore:
            bestscore, best = score, [mv]
    return best[0]

ROT = lambda c: (c[2],c[0],c[1])   # 120deg: maps axis x->y->z

def domino_init(kind, d):
    board = [EMPTY]*N
    if kind=='radial':   base = [(d,0,-d),(d+1,0,-d-1)]
    elif kind=='tang':   base = [(d,0,-d),(d,1,-d-1)]
    cells = base
    for p in range(3):
        for c in cells: board[IDX[c]] = p
        cells = [ROT(c) for c in cells]
    return board

def play(players, surv_max, rng, cap=400, iso='none', init=None):
    board = init[:] if init else [EMPTY]*N
    fresh = frozenset()
    seen = collections.Counter()
    for t in range(cap):
        p = t % 3
        mv = players[p](board, p, surv_max, rng, iso, fresh)
        board, fresh = apply_move(board, p, mv, surv_max, iso, fresh)
        w = winner_after(board, p, t)
        if w is not None:
            kind = 'conn' if connected(board,w) else 'kill'
            return (w, kind, t+1)
        key = (tuple(board), fresh, (t+1)%3)
        seen[key] += 1
        if seen[key] >= 3:
            return (None, 'cycle', t+1)
    return (None, 'cap', cap)

def run(label, players, surv_max, ngames, seed=0, iso='none', init=None):
    rng = random.Random(seed)
    out = collections.Counter(); lens = []
    for g in range(ngames):
        w, kind, t = play(players, surv_max, rng, iso=iso, init=init)
        out[(w,kind)] += 1; lens.append(t)
    wins = sum(v for (w,k),v in out.items() if w is not None)
    print(f"{label:28s} S0-{surv_max} iso={iso:6s} | decided {wins}/{ngames} "
          f"({100*wins/ngames:.0f}%) | avg len {sum(lens)/len(lens):.0f}")
    for (w,k),v in sorted(out.items(), key=lambda kv:-kv[1]):
        who = 'RGB'[w] if w is not None else '-'
        print(f"    {who} {k:5s}: {v}")

if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv)>1 else 30
    for kind in ('radial','tang'):
        for d in (1,2,3):
            init = domino_init(kind, d)
            # sanity: initial position must be a still life
            b2 = transition(init, 4, 'strict')
            assert b2 == init, f"init {kind} d={d} not stable!"
            run(f"greedy^3 {kind} d={d}", [greedy_player]*3, 4, ng, seed=1,
                iso='strict', init=init)
            run(f"random^3 {kind} d={d}", [random_player]*3, 4, ng, seed=2,
                iso='strict', init=init)

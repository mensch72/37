// Core rules engine for "37" — a faithful port of the beak-economy reference
// implementation in sandbox/code/beak3.py (which builds on hex3life.py).
//
// The module is written as portable ES module code so the exact same engine runs
// in the browser (the web app) and under Node (the unit tests that compare the
// growth step against the Python reference).
//
// Board: hexagon of radius R = 3 = 37 cells in cube coordinates, up to 6
// neighbours per cell. Rules: drop/pick move, then a synchronous growth step
// (birth at exactly 4 neighbours, survival with 1..survMax, death at 0 and at
// >survMax), the beak economy (births come from the box, deaths return to it),
// connection win, superko repetition and elimination.
//
// Two survival variants are selectable (issue #12):
//   "calm"     — S1-5: a cell dies only if isolated (0 living neighbours) or
//                completely surrounded (all six neighbours alive). The default.
//   "volatile" — S1-4: the older rule, death at 5 or more neighbours. Harder and
//                more churny (about twice the cell changes between your turns).
// The birth rule is identical in both: birth at exactly 4 neighbours, majority /
// 2-1-1 plurality / 2-2 -> the absent third colour.

export const R = 3;
export const EMPTY = -1;
export const SURV_MAX_VOLATILE = 4; // survival 1..4, i.e. B4 / S1-4 ("volatile")
export const SURV_MAX_CALM = 5;     // survival 1..5, i.e. B4 / S1-5 ("calm")
export const SURV_MAX = SURV_MAX_VOLATILE; // legacy alias (the reference rule)
export const DEFAULT_SURV_MAX = SURV_MAX_CALM; // calm is the new default

// Named rule variants for the UI / game setup.
export const VARIANTS = {
  calm: SURV_MAX_CALM,
  volatile: SURV_MAX_VOLATILE,
};
export const DEFAULT_VARIANT = 'calm';

// Beak capacity: the largest number of flecks a beak may hold. A finite cap of 6
// (one above the largest holding ever seen in ordinary play) closes the "centre
// pump" and hoarding holes (issue #12) at no cost to normal play. Infinity keeps
// the old "unlimited beak" rule (used by the novice setting).
export const DEFAULT_CAP = 6;

// Six cube-coordinate directions (same order as hex3life.py DIRS).
const DIRS = [
  [1, -1, 0], [1, 0, -1], [0, 1, -1],
  [-1, 1, 0], [-1, 0, 1], [0, -1, 1],
];

// Build the canonical cell list in the SAME order as hex3life.py so that any
// index-based comparison against the Python reference lines up exactly:
//   for x in range(-R,R+1) for y in range(-R,R+1) for z in [-x-y] if abs(z)<=R
export const CELLS = [];
for (let x = -R; x <= R; x++) {
  for (let y = -R; y <= R; y++) {
    const z = -x - y;
    if (Math.abs(z) <= R) CELLS.push([x, y, z]);
  }
}
export const N = CELLS.length; // 37

const keyOf = (x, y, z) => `${x},${y},${z}`;
export const IDX = new Map();
CELLS.forEach((c, i) => IDX.set(keyOf(c[0], c[1], c[2]), i));

// Neighbour index lists.
export const NBRS = CELLS.map(([x, y, z]) => {
  const out = [];
  for (const [dx, dy, dz] of DIRS) {
    const k = keyOf(x + dx, y + dy, z + dz);
    if (IDX.has(k)) out.push(IDX.get(k));
  }
  return out;
});

// 120-degree rotation of a cube coordinate: (x,y,z) -> (z,x,y). Maps axis x->y->z.
const rot = ([x, y, z]) => [z, x, y];

// Side membership for each player axis p: SIDES[p] = {plus:[idx...], minus:[idx...]}
// player p owns the pair of opposite sides on axis p (coordinate == +R / -R).
export const SIDES = [];
for (let ax = 0; ax < 3; ax++) {
  const plus = [];
  const minus = [];
  CELLS.forEach((c, i) => {
    if (c[ax] === R) plus.push(i);
    if (c[ax] === -R) minus.push(i);
  });
  SIDES.push({ plus, minus });
}

// Canonicalisation permutation used by the learned policy (mirrors marl_hexlife.py):
// PERM[p][i] = real cell index of canonical cell i for seat p (rho^p applied).
export const PERM = [];
for (let p = 0; p < 3; p++) PERM.push(new Int32Array(N));
CELLS.forEach((c, i) => {
  let r = c;
  for (let p = 0; p < 3; p++) {
    PERM[p][i] = IDX.get(keyOf(r[0], r[1], r[2]));
    r = rot(r);
  }
});

// The canonical C3 "ring" start: each player has two cells on opposite neighbours
// of the centre (a tricolour ring, a still life). Cells (1,0,-1) and (-1,0,1)
// for player 0, rotated for players 1 and 2 (matches beak3.py board0).
export function initialBoard() {
  const board = new Int8Array(N).fill(EMPTY);
  let cells = [[1, 0, -1], [-1, 0, 1]];
  for (let p = 0; p < 3; p++) {
    for (const c of cells) board[IDX.get(keyOf(c[0], c[1], c[2]))] = p;
    cells = cells.map(rot);
  }
  return board;
}

// Synchronous growth step. Reads from `board`, returns {board, born, died} where
// `born` is a list of {i, color} and `died` a list of {i, color} (the colour the
// cell held just before it died). Faithful to
// beak3.py trans(): survival 1..survMax, birth at exactly 4 neighbours with
// majority / 2-1-1 plurality / 2-2 -> the absent third colour. `survMax` selects
// the rule variant (5 = calm / S1-5 default, 4 = volatile / S1-4).
export function growth(board, survMax = DEFAULT_SURV_MAX) {
  const next = board.slice();
  const born = [];
  const died = [];
  const births = [];
  for (let i = 0; i < N; i++) {
    const cnt = [0, 0, 0];
    for (const j of NBRS[i]) {
      const v = board[j];
      if (v !== EMPTY) cnt[v]++;
    }
    const n = cnt[0] + cnt[1] + cnt[2];
    if (board[i] !== EMPTY) {
      if (!(n >= 1 && n <= survMax)) {
        next[i] = EMPTY;
        died.push({ i, color: board[i] });
      }
    } else if (n === 4) {
      // sort colours by descending count, ties broken by lower colour index
      // (stable, matching Python's sorted(range(3), key=lambda q:-cnt[q])).
      const s = [0, 1, 2].sort((a, b) => cnt[b] - cnt[a]);
      let c;
      if (cnt[s[0]] === 2 && cnt[s[1]] === 2) c = s[2]; // 2-2 -> the absent third colour
      else c = s[0]; // 4-0, 3-1, or 2-1-1 -> the (plurality) colour
      births.push([i, c]);
    }
  }
  for (const [i, c] of births) {
    next[i] = c;
    born.push({ i, color: c });
  }
  return { board: next, born, died };
}

function findConnection(board, p) {
  const { plus, minus } = SIDES[p];
  const minusSet = new Set(minus);
  const stack = [];
  const seen = new Set();
  const parent = new Int32Array(N).fill(-1);
  for (const i of plus) {
    if (board[i] === p) { stack.push(i); seen.add(i); }
  }
  while (stack.length) {
    const i = stack.pop();
    if (minusSet.has(i)) {
      const path = [];
      for (let cur = i; cur !== -1; cur = parent[cur]) path.push(cur);
      path.reverse();
      return path;
    }
    for (const j of NBRS[i]) {
      if (board[j] === p && !seen.has(j)) {
        seen.add(j);
        parent[j] = i;
        stack.push(j);
      }
    }
  }
  return null;
}

// Connection win test for player p: is there an unbroken chain of p's colour
// joining p's two opposite sides?
export function connected(board, p) {
  return findConnection(board, p) !== null;
}

// One concrete winning chain for player p, ordered from p's + side to p's - side.
// Returns null when no such connection exists.
export function winningConnection(board, p) {
  return findConnection(board, p);
}

// Winner test after a growth step (mirrors hex3life.winner_after): if anyone is
// connected, the mover wins ties, otherwise the lowest-index connected player.
export function winnerAfter(board, mover) {
  const conn = [connected(board, 0), connected(board, 1), connected(board, 2)];
  if (conn[0] || conn[1] || conn[2]) {
    if (conn[mover]) return mover;
    return conn.indexOf(true);
  }
  return null;
}

// Dijkstra path distance for player p from its + side to its - side: cost 0
// through own cells, 1 through empty, blocked by enemies (mirrors path_dist()).
// Returns the number of stones still needed to connect (Infinity if impossible).
export function pathDist(board, p) {
  const { plus, minus } = SIDES[p];
  const minusSet = new Set(minus);
  const dist = new Array(N).fill(Infinity);
  // simple binary-bucket Dijkstra is overkill for 37 cells; use a small heap.
  const pq = []; // [d, i]
  const push = (d, i) => {
    pq.push([d, i]);
    let c = pq.length - 1;
    while (c > 0) {
      const par = (c - 1) >> 1;
      if (pq[par][0] <= pq[c][0]) break;
      [pq[par], pq[c]] = [pq[c], pq[par]];
      c = par;
    }
  };
  const pop = () => {
    const top = pq[0];
    const last = pq.pop();
    if (pq.length) {
      pq[0] = last;
      let c = 0;
      for (;;) {
        let l = 2 * c + 1, r = 2 * c + 2, m = c;
        if (l < pq.length && pq[l][0] < pq[m][0]) m = l;
        if (r < pq.length && pq[r][0] < pq[m][0]) m = r;
        if (m === c) break;
        [pq[m], pq[c]] = [pq[c], pq[m]];
        c = m;
      }
    }
    return top;
  };
  for (const i of plus) {
    if (board[i] === p) { dist[i] = 0; push(0, i); }
    else if (board[i] === EMPTY) { dist[i] = 1; push(1, i); }
  }
  while (pq.length) {
    const [d, i] = pop();
    if (d > dist[i]) continue;
    if (minusSet.has(i)) return d;
    for (const j of NBRS[i]) {
      let nd;
      if (board[j] === p) nd = d;
      else if (board[j] === EMPTY) nd = d + 1;
      else continue;
      if (nd < dist[j]) { dist[j] = nd; push(nd, j); }
    }
  }
  return Infinity;
}

// Rim defence for player p: how many of p's own two rim sides are anchored by a
// STABLE own presence. A rim side is counted as held when it contains at least one
// own cell that has an adjacent own neighbour on that same rim side — i.e. an own
// *rim pair*, which cannot be starved out (a lone cell with 0 neighbours dies of
// isolation). Holding such a pair on a side permanently prevents an opponent from
// conquering that whole rim side and using it to block the two other colours (see
// issue #7). Returns 0..2.
export function rimAnchors(board, p) {
  let held = 0;
  for (const side of [SIDES[p].plus, SIDES[p].minus]) {
    let anchored = false;
    for (const i of side) {
      if (board[i] !== p) continue;
      for (const j of NBRS[i]) {
        if (board[j] === p && side.includes(j)) { anchored = true; break; }
      }
      if (anchored) break;
    }
    if (anchored) held++;
  }
  return held;
}

// ---- Full game state machine for the app (beak economy + superko + elimination) ----

// A move is encoded as an integer in [0, 2N): drop@i for i in [0,N), pick@(i-N)
// for i in [N, 2N).
export const MOVE_DROP = 0;
export const MOVE_PICK = 1;
export function decodeMove(a) {
  return a < N ? { type: MOVE_DROP, cell: a } : { type: MOVE_PICK, cell: a - N };
}
export function encodeMove(type, cell) {
  return type === MOVE_DROP ? cell : N + cell;
}

// Position key for the superko rule: board + all beaks + player to move.
export function positionKey(board, beaks, player) {
  let s = player + '|';
  for (let i = 0; i < N; i++) s += board[i] + 1; // 0,1,2,3 single chars
  s += '|' + beaks[0] + ',' + beaks[1] + ',' + beaks[2];
  return s;
}

export class Game {
  // beakStart: array of 3 starting beak counts (the difficulty dial N per seat).
  // cap: beak capacity (default 6; Infinity for the "unlimited beak" rule).
  // survMax: survival rule (5 = calm / S1-5 default, 4 = volatile / S1-4).
  constructor({ beakStart = [4, 4, 4], cap = DEFAULT_CAP, survMax = DEFAULT_SURV_MAX } = {}) {
    this.cap = cap;
    this.survMax = survMax;
    this.board = initialBoard();
    this.beaks = beakStart.slice();
    this.toMove = 0;
    this.alive = [true, true, true];
    this.ply = 0;
    this.winner = null;
    this.over = false;
    this.endReason = null; // 'connection' | 'elimination' | 'stuck'
    this.lastBorn = [];
    this.lastDied = [];
    this.history = new Set();
    this.history.add(positionKey(this.board, this.beaks, this.toMove));
    this._skipDeadPlayers();
  }

  clone() {
    const g = Object.create(Game.prototype);
    g.cap = this.cap;
    g.survMax = this.survMax;
    g.board = this.board.slice();
    g.beaks = this.beaks.slice();
    g.toMove = this.toMove;
    g.alive = this.alive.slice();
    g.ply = this.ply;
    g.winner = this.winner;
    g.over = this.over;
    g.endReason = this.endReason;
    g.lastBorn = this.lastBorn.slice();
    g.lastDied = this.lastDied.slice();
    g.history = new Set(this.history);
    return g;
  }

  cellCount(p) {
    let n = 0;
    for (let i = 0; i < N; i++) if (this.board[i] === p) n++;
    return n;
  }

  totalCells(p) {
    return this.cellCount(p) + this.beaks[p];
  }

  isEliminated(p) {
    return this.beaks[p] === 0 && this.cellCount(p) === 0;
  }

  // Blocked-side elimination (issue #12 section 4): if either of player p's two
  // own sides is entirely occupied by other colours, p can no longer place or
  // grow an own cell on that side and so can never connect. Rim cells almost
  // never vacate under the survival rules, so the position is already decided;
  // declaring p out immediately avoids dozens of plies of play by someone who
  // cannot win. A side counts as blocked only when every cell on it is occupied
  // (no empties) and none of them is p's colour.
  isBlocked(p) {
    for (const side of [SIDES[p].plus, SIDES[p].minus]) {
      let allOthers = true;
      for (const i of side) {
        if (this.board[i] === EMPTY || this.board[i] === p) { allOthers = false; break; }
      }
      if (allOthers) return true;
    }
    return false;
  }

  // Legal moves for a player, honouring the beak economy AND the superko rule
  // (a move that recreates a previously seen position is illegal).
  legalMoves(p = this.toMove) {
    const moves = [];
    const canDrop = this.beaks[p] > 0;
    const canPick = this.beaks[p] < this.cap;
    for (let i = 0; i < N; i++) {
      if (canDrop && this.board[i] === EMPTY) {
        if (this._moveIsLegal(p, MOVE_DROP, i)) moves.push(encodeMove(MOVE_DROP, i));
      } else if (canPick && this.board[i] === p) {
        if (this._moveIsLegal(p, MOVE_PICK, i)) moves.push(encodeMove(MOVE_PICK, i));
      }
    }
    return moves;
  }

  // Simulate a move (drop/pick + growth) without mutating this game, returning the
  // resulting board, beaks, born/died lists and the next live player.
  simulate(p, type, cell) {
    const board = this.board.slice();
    const beaks = this.beaks.slice();
    if (type === MOVE_DROP) { board[cell] = p; beaks[p]--; }
    else { board[cell] = EMPTY; beaks[p]++; }
    const g = growth(board, this.survMax);
    return { board: g.board, beaks, born: g.born, died: g.died };
  }

  _nextLivePlayer(afterBoard, afterBeaks, from) {
    // recompute alive on the resulting position for elimination bookkeeping
    const count = [0, 0, 0];
    for (let i = 0; i < N; i++) if (afterBoard[i] !== EMPTY) count[afterBoard[i]]++;
    const aliveNow = [0, 1, 2].map(q => afterBeaks[q] > 0 || count[q] > 0);
    for (let step = 1; step <= 3; step++) {
      const q = (from + step) % 3;
      if (aliveNow[q]) return q;
    }
    return from; // nobody alive (shouldn't happen); caller handles end
  }

  _moveIsLegal(p, type, cell) {
    const { board, beaks } = this.simulate(p, type, cell);
    const w = winnerAfter(board, p);
    if (w !== null) return true; // a winning/decisive move is always legal
    const next = this._nextLivePlayer(board, beaks, p);
    const key = positionKey(board, beaks, next);
    return !this.history.has(key);
  }

  // Apply a concrete move and advance the game. `a` is an encoded move integer.
  applyMove(a) {
    if (this.over) throw new Error('game is over');
    const { type, cell } = decodeMove(a);
    const p = this.toMove;
    if (type === MOVE_DROP) { this.board[cell] = p; this.beaks[p]--; }
    else { this.board[cell] = EMPTY; this.beaks[p]++; }
    const g = growth(this.board, this.survMax);
    this.board = g.board;
    this.lastBorn = g.born;
    this.lastDied = g.died;
    this.ply++;

    // update elimination status: a player is out if naturally eliminated (empty
    // beak and no cells) or has a fully blocked side (issue #12). Once out, they
    // stay out — their cells remain on the board as inert terrain.
    for (let q = 0; q < 3; q++) {
      if (this.alive[q] && (this.isEliminated(q) || this.isBlocked(q))) this.alive[q] = false;
    }

    const w = winnerAfter(this.board, p);
    if (w !== null) {
      this.winner = w;
      this.over = true;
      this.endReason = 'connection';
      return this;
    }
    const aliveCount = this.alive.filter(Boolean).length;
    if (aliveCount <= 1) {
      this.winner = this.alive.indexOf(true);
      this.over = true;
      this.endReason = 'elimination';
      return this;
    }
    this.toMove = this._nextLivePlayer(this.board, this.beaks, p);
    this.history.add(positionKey(this.board, this.beaks, this.toMove));
    this._skipDeadPlayers();
    return this;
  }

  // If the player to move is out (already marked dead, naturally eliminated, or
  // with no legal move), advance until a player who can act is found or the game
  // ends. The persistent `alive` flag also carries blocked-side elimination
  // (issue #12), so a blocked player who still owns terrain is skipped here.
  _skipDeadPlayers() {
    let guard = 0;
    while (!this.over && guard++ < 6) {
      if (!this.alive[this.toMove]) {
        // already out (blocked-side or a prior elimination) — just advance
      } else if (this.isEliminated(this.toMove)) {
        this.alive[this.toMove] = false;
      } else if (this.legalMoves(this.toMove).length === 0) {
        // no legal move -> eliminated (superko/zugzwang backstop)
        this.alive[this.toMove] = false;
      } else {
        return; // current player can act
      }
      const aliveCount = this.alive.filter(Boolean).length;
      if (aliveCount <= 1) {
        this.winner = this.alive.indexOf(true);
        this.over = true;
        this.endReason = 'elimination';
        return;
      }
      // advance to next player that is still marked alive
      let moved = false;
      for (let step = 1; step <= 3; step++) {
        const q = (this.toMove + step) % 3;
        if (this.alive[q]) { this.toMove = q; moved = true; break; }
      }
      if (!moved) { this.over = true; this.winner = null; this.endReason = 'stuck'; return; }
    }
  }
}

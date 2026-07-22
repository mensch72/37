// Shared feature extraction and neural-policy inference for the learned computer
// player. The feature layout here is mirrored exactly by training/features.py so
// that weights trained in Python run unchanged in the browser.
//
// Everything is computed in the acting player's *canonical* frame (marl_hexlife.py
// trick): rotate the board so the acting player's axis is fixed and relabel colours
// so channel 0 is always "self", 1 "next", 2 "prev". Because the 120 rotation is a
// graph automorphism, the canonical board reuses the same NBRS / SIDES tables.

import { N, EMPTY, NBRS, SIDES, PERM } from './engine.js';

// Feature block sizes.
const OCC = 3 * N;      // self / next / prev occupancy one-hots
const EMPTYF = N;       // empty mask
const PATHF = N;        // self shortest-path membership field
const SCALARS = 6;      // path dist (self/next/prev) + beak sizes (self/next/prev)
export const FEAT_LEN = OCC + EMPTYF + PATHF + SCALARS; // 191

// Finite fallback used when a colour has no path between its two sides (distance is
// infinite): the scalar feature 1/(1+d) then maps to a small positive value. Must
// match the Python reference (training/hexlife37.py) so weights transfer unchanged.
const FAR_PATH_DIST = 20;

// Build the canonical board for acting player p: canonical cell i holds the colour
// of real cell PERM[p][i], relabelled so self=0, next=1, prev=2.
export function canonicalBoard(board, p) {
  const cb = new Int8Array(N);
  for (let i = 0; i < N; i++) {
    const v = board[PERM[p][i]];
    cb[i] = v === EMPTY ? EMPTY : (v - p + 3) % 3;
  }
  return cb;
}

// Multi-source 0/1-weighted Dijkstra distance field for colour `c` on a canonical
// board, starting from the given side cells (cost 0 through c, 1 through empty,
// blocked by other colours). Returns an Int array of distances (Infinity if
// unreachable).
function distField(cb, c, sources) {
  const dist = new Array(N).fill(Infinity);
  const pq = [];
  const push = (d, i) => {
    pq.push([d, i]);
    let k = pq.length - 1;
    while (k > 0) {
      const par = (k - 1) >> 1;
      if (pq[par][0] <= pq[k][0]) break;
      [pq[par], pq[k]] = [pq[k], pq[par]];
      k = par;
    }
  };
  const pop = () => {
    const top = pq[0];
    const last = pq.pop();
    if (pq.length) {
      pq[0] = last;
      let k = 0;
      for (;;) {
        let l = 2 * k + 1, r = 2 * k + 2, m = k;
        if (l < pq.length && pq[l][0] < pq[m][0]) m = l;
        if (r < pq.length && pq[r][0] < pq[m][0]) m = r;
        if (m === k) break;
        [pq[m], pq[k]] = [pq[k], pq[m]];
        k = m;
      }
    }
    return top;
  };
  for (const i of sources) {
    let d;
    if (cb[i] === c) d = 0;
    else if (cb[i] === EMPTY) d = 1;
    else continue;
    if (d < dist[i]) { dist[i] = d; push(d, i); }
  }
  while (pq.length) {
    const [d, i] = pop();
    if (d > dist[i]) continue;
    for (const j of NBRS[i]) {
      let nd;
      if (cb[j] === c) nd = d;
      else if (cb[j] === EMPTY) nd = d + 1;
      else continue;
      if (nd < dist[j]) { dist[j] = nd; push(nd, j); }
    }
  }
  return dist;
}

function pathDistOf(cb, c) {
  const dPlus = distField(cb, c, SIDES[c].plus);
  let best = Infinity;
  for (const i of SIDES[c].minus) if (dPlus[i] < best) best = dPlus[i];
  return { dPlus, best };
}

// Compute the full feature vector for the acting player.
export function computeFeatures(board, beaks, p) {
  const cb = canonicalBoard(board, p);
  const f = new Float32Array(FEAT_LEN);
  // occupancy + empty
  for (let i = 0; i < N; i++) {
    const v = cb[i];
    if (v === EMPTY) f[OCC + i] = 1.0;         // empty mask
    else f[v * N + i] = 1.0;                    // self/next/prev channel
  }
  // self shortest-path membership field: cells lying on some minimal self path.
  const self = pathDistOf(cb, 0);
  const dMinus = distField(cb, 0, SIDES[0].minus);
  const base = OCC + EMPTYF;
  if (Number.isFinite(self.best)) {
    for (let i = 0; i < N; i++) {
      if (cb[i] !== 1 && cb[i] !== 2 &&
          self.dPlus[i] + dMinus[i] === self.best) {
        f[base + i] = 1.0;
      }
    }
  }
  // scalar path distances mapped to (0,1]: 1 when connected (d=0), ->0 when far.
  const sb = base + PATHF;
  const pdNext = pathDistOf(cb, 1).best;
  const pdPrev = pathDistOf(cb, 2).best;
  f[sb + 0] = 1.0 / (1.0 + (Number.isFinite(self.best) ? self.best : FAR_PATH_DIST));
  f[sb + 1] = 1.0 / (1.0 + (Number.isFinite(pdNext) ? pdNext : FAR_PATH_DIST));
  f[sb + 2] = 1.0 / (1.0 + (Number.isFinite(pdPrev) ? pdPrev : FAR_PATH_DIST));
  // beak sizes in canonical order, normalised.
  f[sb + 3] = beaks[p] / 8.0;
  f[sb + 4] = beaks[(p + 1) % 3] / 8.0;
  f[sb + 5] = beaks[(p + 2) % 3] / 8.0;
  return f;
}

// Canonical legal-move mask (length 2N): drop@i for empty canonical cells (if
// beak>0), pick@i for self canonical cells (if beak<cap). Superko is NOT applied
// here (the search / engine applies it); this is the raw economy mask used by the
// policy net, matching the training environment.
export function canonicalMask(board, beaks, p, cap = Infinity) {
  const cb = canonicalBoard(board, p);
  const m = new Uint8Array(2 * N);
  const canDrop = beaks[p] > 0;
  const canPick = beaks[p] < cap;
  for (let i = 0; i < N; i++) {
    if (canDrop && cb[i] === EMPTY) m[i] = 1;
    else if (canPick && cb[i] === 0) m[N + i] = 1;
  }
  return m;
}

// Map a canonical action index (0..2N) to a real encoded engine move.
export function canonicalActionToReal(a, p) {
  const cell = PERM[p][a % N];
  return a < N ? cell : N + cell; // engine encodeMove: drop=cell, pick=N+cell
}

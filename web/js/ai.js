// Client-side inference for the learned policy/value network, plus the shallow
// max-n search over the value head described in FINDINGS section 7 (only ~23 legal
// moves, so a 2-ply search is a few hundred evaluations). The network weights are
// loaded from web/weights/policy.json, exported by training/train_ppo.py.
//
// The network is a small MLP with a shared tanh trunk and two heads: a policy head
// (2N logits over canonical drop/pick actions) and a value head (a scalar in
// (-1,1), the expected outcome for the acting/self player).

import { N, EMPTY, decodeMove, encodeMove, MOVE_DROP, winnerAfter, growth } from './engine.js';
import { computeFeatures, canonicalMask, canonicalActionToReal, FEAT_LEN } from './features.js';

// ---- linear algebra on flat Float32Array weights ----
function matVec(W, rows, cols, x, out) {
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) s += W[base + c] * x[c];
    out[r] = s;
  }
  return out;
}

export class Policy {
  constructor(weights) {
    // weights: { layout, W1,b1,W2,b2,Wp,bp,Wv,bv } with row-major flat arrays.
    this.w = weights;
    this.h1 = new Float32Array(weights.h1);
    this.h2 = new Float32Array(weights.h2);
    this.nin = weights.nin;
    if (this.nin !== FEAT_LEN) {
      console.warn(`policy input size ${this.nin} != feature length ${FEAT_LEN}`);
    }
    this._t1 = new Float32Array(weights.h1);
    this._t2 = new Float32Array(weights.h2);
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`failed to load weights: ${res.status}`);
    const w = await res.json();
    // convert nested arrays to flat Float32Array
    const flat = (a) => Float32Array.from(a.flat ? a.flat() : a);
    return new Policy({
      nin: w.nin, h1: w.h1, h2: w.h2,
      W1: flat(w.W1), b1: Float32Array.from(w.b1),
      W2: flat(w.W2), b2: Float32Array.from(w.b2),
      Wp: flat(w.Wp), bp: Float32Array.from(w.bp),
      Wv: flat(w.Wv), bv: w.bv,
    });
  }

  // Forward pass from a raw feature vector. Returns { logits: Float32Array(2N),
  // value: number }.
  forwardFeatures(x) {
    const w = this.w;
    const t1 = this._t1, t2 = this._t2;
    matVec(w.W1, w.h1, w.nin, x, t1);
    for (let i = 0; i < w.h1; i++) t1[i] = Math.tanh(t1[i] + w.b1[i]);
    matVec(w.W2, w.h2, w.h1, t1, t2);
    for (let i = 0; i < w.h2; i++) t2[i] = Math.tanh(t2[i] + w.b2[i]);
    const logits = new Float32Array(2 * N);
    matVec(w.Wp, 2 * N, w.h2, t2, logits);
    for (let i = 0; i < 2 * N; i++) logits[i] += w.bp[i];
    let v = w.bv;
    for (let i = 0; i < w.h2; i++) v += w.Wv[i] * t2[i];
    return { logits, value: Math.tanh(v) };
  }

  // Convenience: features + forward for a given board/beaks/player.
  evaluate(board, beaks, p) {
    return this.forwardFeatures(computeFeatures(board, beaks, p));
  }

  // Softmax policy over legal canonical moves (superko not applied here).
  policy(board, beaks, p, cap = Infinity) {
    const { logits } = this.evaluate(board, beaks, p);
    const mask = canonicalMask(board, beaks, p, cap);
    let mx = -Infinity;
    for (let i = 0; i < 2 * N; i++) if (mask[i] && logits[i] > mx) mx = logits[i];
    let sum = 0;
    const pi = new Float32Array(2 * N);
    for (let i = 0; i < 2 * N; i++) {
      if (mask[i]) { pi[i] = Math.exp(logits[i] - mx); sum += pi[i]; }
    }
    if (sum > 0) for (let i = 0; i < 2 * N; i++) pi[i] /= sum;
    return pi;
  }
}

// The learned computer player. Given a live Game, returns the encoded engine move
// to play. Uses a shallow value-head search: for each legal (superko-checked) move
// it applies the move + growth, then evaluates the resulting position with the
// value net from the mover's perspective; at depth >= 2 it lets the immediate next
// player pick their value-maximising reply (a max-n backup restricted to the top-K
// policy moves to bound cost). Move ordering / priors come from the policy head.
export class LearnedPlayer {
  constructor(policy, { depth = 1, topK = 8, temperature = 0.0 } = {}) {
    this.policy = policy;
    this.depth = depth;
    this.topK = topK;
    this.temperature = temperature;
  }

  // Value of a position for player `perspective`, from the value net.
  _value(board, beaks, perspective) {
    // decisive positions get exact values
    const w = winnerAfter(board, perspective);
    if (w !== null) return w === perspective ? 1.0 : -0.5;
    return this.policy.evaluate(board, beaks, perspective).value;
  }

  // Recursive max-n value of a position where `player` is to move, `depth` plies
  // left. Returns the value for `rootPlayer`.
  _search(board, beaks, player, depth, rootPlayer, legalFor) {
    // terminal check for the player who just would have moved is handled by caller;
    // here we expand `player`'s moves.
    const priors = this.policy.policy(board, beaks, player);
    const mask = canonicalMask(board, beaks, player);
    // candidate canonical actions, ordered by prior, top-K
    const cand = [];
    for (let a = 0; a < 2 * N; a++) if (mask[a]) cand.push(a);
    if (cand.length === 0) return this._value(board, beaks, rootPlayer);
    cand.sort((a, b) => priors[b] - priors[a]);
    const K = Math.min(this.topK, cand.length);
    let bestForPlayer = -Infinity;
    let bestRootVal = this._value(board, beaks, rootPlayer);
    for (let k = 0; k < K; k++) {
      const a = cand[k];
      const { type, cell } = decodeMove(a); // canonical decode (same encoding)
      // apply canonical move on a canonical-agnostic board: we must map to real.
      // Here board/beaks are in REAL space; convert canonical action to real.
      const realA = canonicalActionToReal(a, player);
      const rm = decodeMove(realA);
      const nb = board.slice();
      const nk = beaks.slice();
      if (rm.type === MOVE_DROP) { nb[rm.cell] = player; nk[player]--; }
      else { nb[rm.cell] = EMPTY; nk[player]++; }
      const g = growth(nb);
      const resBoard = g.board;
      // decisive?
      const win = winnerAfter(resBoard, player);
      let rootVal, playerVal;
      if (win !== null) {
        playerVal = win === player ? 1.0 : -0.5;
        rootVal = win === rootPlayer ? 1.0 : -0.5;
      } else if (depth <= 1) {
        playerVal = this._value(resBoard, nk, player);
        rootVal = player === rootPlayer ? playerVal : this._value(resBoard, nk, rootPlayer);
      } else {
        const nextPlayer = (player + 1) % 3;
        rootVal = this._search(resBoard, nk, nextPlayer, depth - 1, rootPlayer, legalFor);
        playerVal = player === rootPlayer ? rootVal : this._search(resBoard, nk, nextPlayer, depth - 1, player, legalFor);
      }
      if (playerVal > bestForPlayer) { bestForPlayer = playerVal; bestRootVal = rootVal; }
    }
    return bestRootVal;
  }

  chooseMove(game) {
    const p = game.toMove;
    const legal = game.legalMoves(p); // real encoded, superko-filtered
    if (legal.length === 0) return null;
    const board = game.board, beaks = game.beaks;
    const priors = this.policy.policy(board, beaks, p, game.cap);

    // Score each legal move.
    const scored = legal.map((realA) => {
      const rm = decodeMove(realA);
      const nb = board.slice();
      const nk = beaks.slice();
      if (rm.type === MOVE_DROP) { nb[rm.cell] = p; nk[p]--; }
      else { nb[rm.cell] = EMPTY; nk[p]++; }
      const g = growth(nb);
      const resBoard = g.board;
      const win = winnerAfter(resBoard, p);
      let val;
      if (win !== null) {
        val = win === p ? 1e6 : -1e6; // take a win, refuse to hand one away
      } else if (this.depth <= 1) {
        val = this._value(resBoard, nk, p);
      } else {
        const nextPlayer = (p + 1) % 3;
        val = this._search(resBoard, nk, nextPlayer, this.depth - 1, p, null);
      }
      // canonical prior for this real move
      // map real->canonical action index for prior lookup
      const canonCell = _realCellToCanonical(rm.cell, p);
      const canonA = rm.type === MOVE_DROP ? canonCell : N + canonCell;
      const prior = priors[canonA] || 0;
      return { realA, val, prior };
    });

    if (this.temperature > 0) {
      // sample proportionally to exp(val/T) blended with prior
      let mx = -Infinity;
      for (const s of scored) { const z = s.val / this.temperature; if (z > mx) mx = z; }
      let sum = 0;
      for (const s of scored) { s.w = Math.exp(s.val / this.temperature - mx) * (0.1 + s.prior); sum += s.w; }
      let r = Math.random() * sum;
      for (const s of scored) { r -= s.w; if (r <= 0) return s.realA; }
      return scored[0].realA;
    }
    // deterministic: highest value, prior as tiebreak
    scored.sort((a, b) => (b.val - a.val) || (b.prior - a.prior));
    return scored[0].realA;
  }
}

// Inverse of PERM for the current player, computed lazily.
import { PERM } from './engine.js';
const INV_PERM = PERM.map((row) => {
  const inv = new Int32Array(N);
  for (let i = 0; i < N; i++) inv[row[i]] = i;
  return inv;
});
function _realCellToCanonical(cell, p) {
  return INV_PERM[p][cell];
}

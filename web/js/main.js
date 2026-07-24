// App controller for "37": wires the setup form, the human/computer game loop, the
// growth-step highlights, the per-player panel and the move-history scrubber.
import { Game, decodeMove, encodeMove, MOVE_DROP, EMPTY, winningConnection, VARIANTS, DEFAULT_VARIANT } from './engine.js';
import { Renderer, renderPlayers, COLOR_NAME } from './ui.js';
import { Policy, LearnedPlayer } from './ai.js';

const $ = (sel) => document.querySelector(sel);

// Beak dial (issue #1's "N"): the number of stones a seat can hold off-board, i.e.
// how many drops it can make before it must pick one back up. "novice" is the
// game's "unlimited beak" rule; we use a large finite value rather than Infinity
// because the learned AI's features normalise the beak count (dividing by 8), so a
// non-finite value would poison inference. 37 cells make 37 an unreachable cap.
//
// `cap` is the beak capacity. Standard settings use 6 — one above the largest
// holding ever seen in ordinary play — which closes the centre-pump / hoarding
// exploits (issue #12) at no cost. Novice keeps the old unlimited beak.
const UNLIMITED_BEAK = 37;
const DIFFICULTY = {
  novice: { beak: [UNLIMITED_BEAK, UNLIMITED_BEAK, UNLIMITED_BEAK], cap: UNLIMITED_BEAK },
  advanced: { beak: [4, 4, 4], cap: 6 },
  expert: { beak: [2, 2, 2], cap: 6 },
  handicap: { beak: [4, 3, 2], cap: 6 },
};

// The learned policy is trained per rule variant (calm / volatile); load the
// matching weight file and fall back to the shared policy.json if it is absent.
const policyPromises = new Map();
function getPolicy(variant) {
  if (!policyPromises.has(variant)) {
    const url = `weights/policy-${variant}.json`;
    const p = Policy.load(url).catch(() => Policy.load('weights/policy.json'));
    policyPromises.set(variant, p);
  }
  return policyPromises.get(variant);
}

class Controller {
  constructor() {
    this.renderer = new Renderer($('#board'));
    this.renderer.onCellClick = (i) => this.onCellClick(i);
    this.history = [];   // snapshots
    this.viewIdx = 0;
    this.seatTypes = ['human', 'ai', 'ai'];
    this.game = null;
    this.ai = null;
    this.busy = false;
    this.aiTimer = null;
    this.animTimers = [];
    this.aiMoveMs = 2000; // computer move delay so humans can follow the play
    this._bindControls();
  }

  _bindControls() {
    $('#start-btn').addEventListener('click', () => this.start());
    $('#new-game-btn').addEventListener('click', () => this.reset());
    // The AI move-speed knob appears on both the setup page and the play page;
    // keep them in sync and drive both the move delay and the marker fade time.
    this.speedInputs = [$('#ai-speed'), $('#ai-speed-game')].filter(Boolean);
    this.speedLabels = [$('#ai-speed-value'), $('#ai-speed-game-value')].filter(Boolean);
    const applySpeed = (val) => {
      const v = parseFloat(val);
      this.aiMoveMs = Math.round(v * 1000);
      const text = `${v.toFixed(1)} s per move`;
      this.speedInputs.forEach((inp) => { inp.value = String(v); });
      this.speedLabels.forEach((lab) => { lab.textContent = text; });
      if (this.history.length) this.draw();
    };
    this.speedInputs.forEach((inp) => inp.addEventListener('input', (e) => applySpeed(e.target.value)));
    applySpeed(this.speedInputs.length ? this.speedInputs[0].value : 2);
    $('#scrubber').addEventListener('input', (e) => this.viewAt(+e.target.value));
    $('#hist-first').addEventListener('click', () => this.viewAt(0));
    $('#hist-prev').addEventListener('click', () => this.viewAt(this.viewIdx - 1));
    $('#hist-next').addEventListener('click', () => this.viewAt(this.viewIdx + 1));
    $('#hist-last').addEventListener('click', () => this.viewAt(this.history.length - 1));
    $('#undo-btn').addEventListener('click', () => this.undo());
    $('#return-btn').addEventListener('click', () => this.returnToView());
  }

  reset() {
    this.cancelPendingAI();
    this.busy = false;
    $('#game').classList.add('hidden');
    $('#setup').classList.remove('hidden');
  }

  async start() {
    // read seat types
    document.querySelectorAll('.seat-config').forEach((el) => {
      const seat = +el.dataset.seat;
      this.seatTypes[seat] = el.querySelector('[data-role="type"]').value;
    });
    const diff = $('#difficulty').value;
    const { beak, cap } = DIFFICULTY[diff];
    const variantSel = $('#rule-variant');
    const variant = variantSel && VARIANTS[variantSel.value] ? variantSel.value : DEFAULT_VARIANT;
    const survMax = VARIANTS[variant];
    this.cancelPendingAI();
    this.busy = false;
    this.game = new Game({ beakStart: beak.slice(), cap, survMax });

    const depth = +$('#ai-strength').value;
    const anyAI = this.seatTypes.includes('ai');
    if (anyAI) {
      $('#ai-status').textContent = 'Loading learned policy…';
      try {
        const policy = await getPolicy(variant);
        this.ai = new LearnedPlayer(policy, { depth, topK: 8, survMax });
        $('#ai-status').textContent = '';
      } catch (err) {
        this.ai = null;
        $('#ai-status').textContent =
          'Could not load the trained policy (weights/policy.json). Computer seats will be disabled — set them to Human.';
        return;
      }
    } else {
      this.ai = null;
    }

    this.history = [];
    this.snapshot('Game start.');
    $('#setup').classList.add('hidden');
    $('#game').classList.remove('hidden');
    this.viewAt(0);
    this.maybeAIMove();
  }

  snapshot(message) {
    const g = this.game;
    this.history.push({
      game: g.clone(),
      board: g.board.slice(),
      beaks: g.beaks.slice(),
      toMove: g.toMove,
      born: g.lastBorn.slice(),
      died: g.lastDied.slice(),
      ply: g.ply,
      over: g.over,
      winner: g.winner,
      endReason: g.endReason,
      alive: g.alive.slice(),
      message,
    });
    this.viewIdx = this.history.length - 1;
    this.syncScrubber();
  }

  syncScrubber() {
    const s = $('#scrubber');
    s.max = String(this.history.length - 1);
    s.value = String(this.viewIdx);
    $('#hist-label').textContent = `${this.viewIdx} / ${this.history.length - 1}`;
  }

  atLatest() { return this.viewIdx === this.history.length - 1; }

  viewAt(idx) {
    idx = Math.max(0, Math.min(this.history.length - 1, idx));
    this.viewIdx = idx;
    this.syncScrubber();
    this.draw();
  }

  // Cells the current human may click (part of a superko-legal move).
  playableCells() {
    if (!this.game || this.game.over || !this.atLatest()) return null;
    if (this.seatTypes[this.game.toMove] !== 'human') return null;
    const set = new Set();
    for (const a of this.game.legalMoves()) set.add(decodeMove(a).cell);
    return set;
  }

  draw() {
    const snap = this.history[this.viewIdx];
    const playable = this.playableCells();
    this.renderer.render({
      board: snap.board,
      beaks: snap.beaks,
      playable,
      born: snap.born,
      died: snap.died,
      showHighlights: true,
      fadeMs: this.aiMoveMs,
      winningPath: snap.over && snap.endReason === 'connection' && snap.winner !== null
        ? winningConnection(snap.board, snap.winner)
        : null,
      winner: snap.over && snap.endReason === 'connection' ? snap.winner : null,
    });
    this.drawPanels(snap);
    $('#hint').textContent = playable && playable.size
      ? 'Click an empty cell to drop, or one of your cells to pick.'
      : (this.atLatest() ? '' : 'Viewing history — use ⏭ to return to the live game.');
  }

  // The non-board parts of a draw: the per-player panel, the turn banner, the
  // growth log and the history buttons. Shared by draw() and the live-move
  // animation so the resting state is correct without a full static redraw.
  drawPanels(snap) {
    renderPlayers($('#players'), this.gameForView(snap), this.seatTypes);
    this.drawBanner(snap);
    this.drawGrowthLog(snap);
    this.syncHistoryButtons();
  }

  // Build a lightweight game-like object for the panel from a snapshot.
  gameForView(snap) {
    return {
      toMove: snap.toMove,
      over: snap.over,
      beaks: snap.beaks,
      cellCount: (p) => { let n = 0; for (let i = 0; i < snap.board.length; i++) if (snap.board[i] === p) n++; return n; },
      totalCells: (p) => { let n = 0; for (let i = 0; i < snap.board.length; i++) if (snap.board[i] === p) n++; return n + snap.beaks[p]; },
      // A seat is shown "out" when the engine has marked it not alive (natural
      // elimination or a fully blocked side, issue #12); fall back to the
      // beak+board test for snapshots taken before the flag existed.
      isEliminated: (p) => (snap.alive ? !snap.alive[p]
        : snap.beaks[p] === 0 && (() => { for (let i = 0; i < snap.board.length; i++) if (snap.board[i] === p) return false; return true; })()),
    };
  }

  drawBanner(snap) {
    const banner = $('#turn-banner');
    if (snap.over) {
      if (snap.winner === null) {
        banner.textContent = 'Game over — no winner (stalemate).';
      } else {
        const how = snap.endReason === 'elimination' ? 'by elimination' : 'connected their sides';
        banner.textContent = `${COLOR_NAME[snap.winner]} wins — ${how}!`;
      }
      banner.style.color = snap.winner === null ? '' : `var(--${['red', 'yellow', 'blue'][snap.winner]})`;
    } else {
      const who = this.seatTypes[snap.toMove] === 'ai' ? 'Computer' : 'You';
      banner.textContent = `${COLOR_NAME[snap.toMove]} to move (${who}) — ply ${snap.ply}`;
      banner.style.color = `var(--${['red', 'yellow', 'blue'][snap.toMove]})`;
    }
  }

  drawGrowthLog(snap) {
    const el = $('#growth-summary');
    if (snap.ply === 0) { el.textContent = 'The tricolour ring is a still life — nothing has grown yet.'; return; }
    const nb = snap.born.length, nd = snap.died.length;
    const byColor = [0, 0, 0];
    snap.born.forEach((b) => byColor[b.color]++);
    const bornStr = nb
      ? `${nb} born (` + byColor.map((c, i) => c ? `${c} ${COLOR_NAME[i].toLowerCase()}` : null).filter(Boolean).join(', ') + ')'
      : 'no births';
    el.textContent = `${bornStr}; ${nd ? nd + ' died' : 'no deaths'}.`;
  }

  syncHistoryButtons() {
    $('#undo-btn').disabled = !this.game || !this.atLatest() || this.history.length < 2 || this.busy;
    $('#return-btn').disabled = !this.game || this.atLatest() || this.busy;
  }

  cancelPendingAI() {
    if (this.aiTimer !== null) {
      clearTimeout(this.aiTimer);
      this.aiTimer = null;
    }
    this.animTimers.forEach((t) => clearTimeout(t));
    this.animTimers = [];
  }

  restoreHistory(idx) {
    const snap = this.history[idx];
    if (!snap) return;
    this.cancelPendingAI();
    this.busy = false;
    this.game = snap.game.clone();
    this.history = this.history.slice(0, idx + 1);
    this.viewIdx = this.history.length - 1;
    this.syncScrubber();
    this.draw();
    if (!this.game.over) this.maybeAIMove();
  }

  undo() {
    if (!this.game || !this.atLatest() || this.history.length < 2 || this.busy) return;
    this.restoreHistory(this.history.length - 2);
  }

  returnToView() {
    if (!this.game || this.atLatest() || this.busy) return;
    this.restoreHistory(this.viewIdx);
  }

  onCellClick(i) {
    if (this.busy || !this.game || this.game.over || !this.atLatest()) return;
    if (this.seatTypes[this.game.toMove] !== 'human') return;
    const v = this.game.board[i];
    let a;
    if (v === EMPTY) a = encodeMove(MOVE_DROP, i);
    else if (v === this.game.toMove) a = encodeMove(1, i); // pick
    else return;
    if (!this.game.legalMoves().includes(a)) return; // e.g. superko-illegal
    this.applyAndContinue(a);
  }

  applyAndContinue(a) {
    const mover = this.game.toMove;
    const { type, cell } = decodeMove(a);
    this.game.applyMove(a);
    const verb = type === MOVE_DROP ? 'dropped on' : 'picked from';
    this.snapshot(`${COLOR_NAME[mover]} ${verb} cell ${cell}.`);
    // Animate the move as two paced phases — the fleck flying to/from the beak,
    // then the automaton growth — so the growth no longer happens instantly with
    // the placement. When the animation settles, hand off to the next player.
    this.busy = true;
    this.syncHistoryButtons();
    this.animateMove({ mover, type, cell }, () => {
      this.animTimers = [];
      if (this.game.over) {
        this.busy = false;
        this.drawPanels(this.history[this.viewIdx]);
        $('#hint').textContent = '';
        return;
      }
      if (this.seatTypes[this.game.toMove] === 'ai' && this.ai) {
        this.maybeAIMove();
      } else {
        this.busy = false;
        this.draw();
      }
    });
  }

  // Play the two-phase animation for the move that produced the latest snapshot.
  // Phase 1 shows the placement (the dropped/picked fleck flying between the
  // beak and the board); phase 2 shows the boulder growth (born cells grow from
  // their centre, dying cells shrink toward it). Both phases are paced by the
  // move-duration knob; `done` runs once the growth phase has settled.
  animateMove(move, done) {
    const snap = this.history[this.history.length - 1];
    const { fly, grow } = this._phaseDurations();
    // Board right after the placement but before the automaton grew: undo the
    // births (back to empty) and revive the cells that the growth killed.
    const preBoard = snap.board.slice();
    for (const b of snap.born) preBoard[b.i] = EMPTY;
    for (const d of snap.died) preBoard[d.i] = d.color;

    this.renderer.render({
      board: preBoard, beaks: snap.beaks, playable: null,
      born: [], died: [], showHighlights: false,
      anim: { durMs: fly, fly: { cell: move.cell, type: move.type, color: move.mover }, grow: null, shrink: null },
    });
    this.drawPanels(snap);
    $('#hint').textContent = '';

    const growSet = new Set(snap.born.map((b) => b.i));
    const shrinkMap = new Map(snap.died.map((d) => [d.i, d.color]));
    const hasGrowth = growSet.size > 0 || shrinkMap.size > 0;
    const isWin = snap.over && snap.endReason === 'connection' && snap.winner !== null;

    const t1 = setTimeout(() => {
      this.renderer.render({
        board: snap.board, beaks: snap.beaks, playable: null,
        born: [], died: [], showHighlights: false,
        winningPath: isWin ? winningConnection(snap.board, snap.winner) : null,
        winner: isWin ? snap.winner : null,
        anim: hasGrowth ? { durMs: grow, fly: null, grow: growSet, shrink: shrinkMap } : null,
      });
      const t2 = setTimeout(done, grow);
      this.animTimers.push(t2);
    }, fly);
    this.animTimers.push(t1);
  }

  // Split the "seconds per move" knob into a short think pause plus the fleck
  // fly and the growth reveal, so a whole turn takes roughly the set duration.
  _phaseDurations() {
    const d = Math.max(0, this.aiMoveMs);
    return {
      think: Math.max(60, Math.round(d * 0.2)),
      fly: Math.max(140, Math.round(d * 0.4)),
      grow: Math.max(140, Math.round(d * 0.4)),
    };
  }

  maybeAIMove() {
    if (!this.game || this.game.over) return;
    if (this.seatTypes[this.game.toMove] !== 'ai' || !this.ai) return;
    this.busy = true;
    this.syncHistoryButtons();
    $('#hint').textContent = 'Computer is thinking…';
    // let the UI paint, then wait a short think pause before the animated move
    this.aiTimer = setTimeout(() => {
      this.aiTimer = null;
      try {
        const a = this.ai.chooseMove(this.game);
        if (a === null) {
          // no legal move: engine will have eliminated the player on next sync;
          // force a re-evaluation by applying nothing — instead advance via a
          // no-op snapshot. This path is extremely rare (superko zugzwang).
          this.busy = false;
          this.draw();
          return;
        }
        this.applyAndContinue(a);
      } catch (err) {
        console.error(err);
        this.busy = false;
        this.draw();
        $('#hint').textContent = 'Computer error — see console.';
      }
    }, Math.max(50, this._phaseDurations().think));
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.__controller = new Controller();
});

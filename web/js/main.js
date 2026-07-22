// App controller for "37": wires the setup form, the human/computer game loop, the
// growth-step highlights, the per-player panel and the move-history scrubber.
import { Game, decodeMove, encodeMove, MOVE_DROP, EMPTY } from './engine.js';
import { Renderer, renderPlayers, COLOR_NAME } from './ui.js';
import { Policy, LearnedPlayer } from './ai.js';

const $ = (sel) => document.querySelector(sel);

// Beak dial (issue #1's "N"): the number of stones a seat can hold off-board, i.e.
// how many drops it can make before it must pick one back up. "novice" is the
// game's "unlimited beak" rule; we use a large finite value rather than Infinity
// because the learned AI's features normalise the beak count (dividing by 8), so a
// non-finite value would poison inference. 37 cells make 37 an unreachable cap.
const UNLIMITED_BEAK = 37;
const DIFFICULTY = {
  novice: { beak: [UNLIMITED_BEAK, UNLIMITED_BEAK, UNLIMITED_BEAK] },
  advanced: { beak: [4, 4, 4] },
  expert: { beak: [2, 2, 2] },
  handicap: { beak: [4, 3, 2] },
};

let policyPromise = null;
function getPolicy() {
  if (!policyPromise) policyPromise = Policy.load('weights/policy.json');
  return policyPromise;
}

class Controller {
  constructor() {
    this.renderer = new Renderer($('#board'));
    this.renderer.onCellClick = (i) => this.onCellClick(i);
    this.history = [];   // snapshots
    this.viewIdx = 0;
    this.seatTypes = ['ai', 'human', 'ai'];
    this.game = null;
    this.ai = null;
    this.busy = false;
    this._bindControls();
  }

  _bindControls() {
    $('#start-btn').addEventListener('click', () => this.start());
    $('#new-game-btn').addEventListener('click', () => this.reset());
    $('#scrubber').addEventListener('input', (e) => this.viewAt(+e.target.value));
    $('#hist-first').addEventListener('click', () => this.viewAt(0));
    $('#hist-prev').addEventListener('click', () => this.viewAt(this.viewIdx - 1));
    $('#hist-next').addEventListener('click', () => this.viewAt(this.viewIdx + 1));
    $('#hist-last').addEventListener('click', () => this.viewAt(this.history.length - 1));
  }

  reset() {
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
    const beak = DIFFICULTY[diff].beak.slice();
    this.game = new Game({ beakStart: beak, cap: Infinity });

    const depth = +$('#ai-strength').value;
    const anyAI = this.seatTypes.includes('ai');
    if (anyAI) {
      $('#ai-status').textContent = 'Loading learned policy…';
      try {
        const policy = await getPolicy();
        this.ai = new LearnedPlayer(policy, { depth, topK: 8 });
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
      board: g.board.slice(),
      beaks: g.beaks.slice(),
      toMove: g.toMove,
      born: g.lastBorn.slice(),
      died: g.lastDied.slice(),
      ply: g.ply,
      over: g.over,
      winner: g.winner,
      endReason: g.endReason,
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
    });
    renderPlayers($('#players'), this.gameForView(snap), this.seatTypes);
    this.drawBanner(snap);
    this.drawGrowthLog(snap);
    $('#hint').textContent = playable && playable.size
      ? 'Click an empty cell to drop, or one of your cells to pick.'
      : (this.atLatest() ? '' : 'Viewing history — use ⏭ to return to the live game.');
  }

  // Build a lightweight game-like object for the panel from a snapshot.
  gameForView(snap) {
    return {
      toMove: snap.toMove,
      over: snap.over,
      beaks: snap.beaks,
      cellCount: (p) => { let n = 0; for (let i = 0; i < snap.board.length; i++) if (snap.board[i] === p) n++; return n; },
      totalCells: (p) => { let n = 0; for (let i = 0; i < snap.board.length; i++) if (snap.board[i] === p) n++; return n + snap.beaks[p]; },
      isEliminated: (p) => snap.beaks[p] === 0 && (() => { for (let i = 0; i < snap.board.length; i++) if (snap.board[i] === p) return false; return true; })(),
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
    this.viewAt(this.history.length - 1);
    if (!this.game.over) this.maybeAIMove();
  }

  maybeAIMove() {
    if (!this.game || this.game.over) return;
    if (this.seatTypes[this.game.toMove] !== 'ai' || !this.ai) return;
    this.busy = true;
    $('#hint').textContent = 'Computer is thinking…';
    // let the UI paint before the (synchronous) search runs
    setTimeout(() => {
      try {
        const a = this.ai.chooseMove(this.game);
        if (a === null) {
          // no legal move: engine will have eliminated the player on next sync;
          // force a re-evaluation by applying nothing — instead advance via a
          // no-op snapshot. This path is extremely rare (superko zugzwang).
          this.busy = false;
          return;
        }
        this.busy = false;
        this.applyAndContinue(a);
      } catch (err) {
        console.error(err);
        this.busy = false;
        $('#hint').textContent = 'Computer error — see console.';
      }
    }, 220);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.__controller = new Controller();
});

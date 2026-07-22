// Rendering for the "37" board: an SVG hexagon of 37 cells with red/yellow/blue
// lichen shown by hue AND shape (triangle / circle / square) so the board reads
// without relying on the red–green axis. Also renders the birth/death highlights
// for the last growth step, the per-player panel and the turn banner.

import { CELLS, N, EMPTY } from './engine.js';

const SVGNS = 'http://www.w3.org/2000/svg';
const SIZE = 12; // hex radius in SVG units

// Flat-top hex layout: cube (x,y,z) -> axial (q=x, r=z) -> pixel.
function cellCenter([x, , z]) {
  const q = x, r = z;
  const px = SIZE * 1.5 * q;
  const py = SIZE * Math.sqrt(3) * (r + q / 2);
  return [px, py];
}

function hexPoints(cx, cy) {
  return hexPointsR(cx, cy, SIZE);
}

function hexPointsR(cx, cy, size) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 180) * (60 * i);
    pts.push(`${(cx + size * Math.cos(a)).toFixed(2)},${(cy + size * Math.sin(a)).toFixed(2)}`);
  }
  return pts.join(' ');
}

const COLOR_CLASS = ['stone-0', 'stone-1', 'stone-2'];
const COLOR_NAME = ['Red', 'Yellow', 'Blue'];

export class Renderer {
  constructor(svg) {
    this.svg = svg;
    this.cellGroups = [];
    this.onCellClick = null;
    this._build();
  }

  _build() {
    while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);
    this.cellGroups = CELLS.map((c, i) => {
      const [cx, cy] = cellCenter(c);
      const g = document.createElementNS(SVGNS, 'g');
      g.setAttribute('class', 'cell');
      g.dataset.idx = i;

      const bg = document.createElementNS(SVGNS, 'polygon');
      bg.setAttribute('points', hexPoints(cx, cy));
      bg.setAttribute('class', 'cell-bg');
      bg.setAttribute('fill', 'var(--empty)');
      bg.setAttribute('stroke', 'var(--empty-line)');
      bg.setAttribute('stroke-width', '0.7');
      g.appendChild(bg);

      // Coloured, inset outlines showing which player(s) own this edge cell. A
      // corner cell belongs to two players, so outlines stack with a small inset.
      const colorVars = ['var(--red)', 'var(--yellow)', 'var(--blue)'];
      let owners = 0;
      for (let ax = 0; ax < 3; ax++) {
        if (Math.abs(c[ax]) === 3) {
          const inset = SIZE * (0.92 - owners * 0.22);
          const outline = document.createElementNS(SVGNS, 'polygon');
          outline.setAttribute('points', hexPointsR(cx, cy, inset));
          outline.setAttribute('fill', 'none');
          outline.setAttribute('stroke', colorVars[ax]);
          outline.setAttribute('stroke-width', '2');
          outline.setAttribute('opacity', '0.85');
          g.appendChild(outline);
          owners++;
        }
      }

      g.addEventListener('click', () => {
        if (this.onCellClick) this.onCellClick(i);
      });
      this.svg.appendChild(g);
      return { g, bg, cx, cy, stone: null, marks: [] };
    });
  }

  // state: { board, beaks, playable:Set<int>, born:[{i,color}], died:[int],
  //          showHighlights:bool }
  render(state) {
    const { board, playable, born = [], died = [], showHighlights = true } = state;
    const bornSet = new Set(born.map((b) => b.i));
    const diedSet = new Set(died);
    this.cellGroups.forEach((cg, i) => {
      // clear previous stone and marks
      if (cg.stone) { cg.stone.remove(); cg.stone = null; }
      cg.marks.forEach((m) => m.remove());
      cg.marks = [];

      const v = board[i];
      cg.g.classList.toggle('playable', !!playable && playable.has(i));

      if (v !== EMPTY) {
        cg.stone = this._stoneShape(v, cg.cx, cg.cy);
        cg.g.appendChild(cg.stone);
      }
      if (showHighlights && bornSet.has(i)) {
        const ring = document.createElementNS(SVGNS, 'circle');
        ring.setAttribute('cx', cg.cx); ring.setAttribute('cy', cg.cy);
        ring.setAttribute('r', SIZE * 0.85);
        ring.setAttribute('class', 'born-ring');
        cg.g.appendChild(ring); cg.marks.push(ring);
      }
      if (showHighlights && diedSet.has(i)) {
        const x = document.createElementNS(SVGNS, 'path');
        const d = SIZE * 0.45;
        x.setAttribute('d', `M ${cg.cx - d} ${cg.cy - d} L ${cg.cx + d} ${cg.cy + d} M ${cg.cx + d} ${cg.cy - d} L ${cg.cx - d} ${cg.cy + d}`);
        x.setAttribute('class', 'died-mark');
        cg.g.appendChild(x); cg.marks.push(x);
      }
    });
  }

  _stoneShape(color, cx, cy) {
    const r = SIZE * 0.62;
    let el;
    if (color === 0) { // Red triangle
      el = document.createElementNS(SVGNS, 'polygon');
      const p = [
        [cx, cy - r], [cx + r * 0.92, cy + r * 0.7], [cx - r * 0.92, cy + r * 0.7],
      ].map((q) => q.map((n) => n.toFixed(2)).join(',')).join(' ');
      el.setAttribute('points', p);
    } else if (color === 1) { // Yellow circle
      el = document.createElementNS(SVGNS, 'circle');
      el.setAttribute('cx', cx); el.setAttribute('cy', cy); el.setAttribute('r', r * 0.9);
    } else { // Blue square
      el = document.createElementNS(SVGNS, 'rect');
      const s = r * 1.5;
      el.setAttribute('x', cx - s / 2); el.setAttribute('y', cy - s / 2);
      el.setAttribute('width', s); el.setAttribute('height', s);
      el.setAttribute('rx', 2);
    }
    el.setAttribute('class', `stone ${COLOR_CLASS[color]}`);
    return el;
  }
}

export function renderPlayers(container, game, seatTypes) {
  container.innerHTML = '';
  for (let p = 0; p < 3; p++) {
    const row = document.createElement('div');
    row.className = 'player-row';
    if (p === game.toMove && !game.over) row.classList.add('active');
    if (game.isEliminated(p)) row.classList.add('eliminated');

    const chip = document.createElement('span');
    chip.className = `chip chip-${p}`;
    row.appendChild(chip);

    const mid = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'player-name';
    name.textContent = `${COLOR_NAME[p]} (${seatTypes[p] === 'ai' ? 'Computer' : 'Human'})`;
    const stats = document.createElement('div');
    stats.className = 'player-stats';
    stats.textContent = `beak ${game.beaks[p]} · on board ${game.cellCount(p)}`;
    mid.appendChild(name); mid.appendChild(stats);
    row.appendChild(mid);

    const tot = document.createElement('div');
    tot.className = 'player-totals';
    tot.textContent = `${game.totalCells(p)} total`;
    row.appendChild(tot);

    container.appendChild(row);
  }
}

export { COLOR_NAME };

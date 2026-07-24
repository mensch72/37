// Rendering for the "37" board: an SVG hexagon of 37 cells with red/yellow/blue
// lichen shown by hue AND shape (triangle / circle / square) so the board reads
// without relying on the red–green axis. Also renders the birth/death highlights
// for the last growth step, the per-player panel and the turn banner.

import { CELLS, EMPTY, R, MOVE_DROP } from './engine.js';

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
const COLOR_VAR = ['var(--red)', 'var(--yellow)', 'var(--blue)'];
const COLOR_SOFT = ['var(--red-soft)', 'var(--yellow-soft)', 'var(--blue-soft)'];
const COLOR_NAME = ['Red', 'Yellow', 'Blue'];

export class Renderer {
  constructor(svg) {
    this.svg = svg;
    this.cellGroups = [];
    this.sideSegments = Array.from({ length: 3 }, () => ({ plus: null, minus: null }));
    // A point just off the board on each player's home side, used as the
    // visual "beak" that flecks fly to/from when dropped or picked.
    this.beakAnchors = [null, null, null];
    this.winningPath = null;
    this.onCellClick = null;
    this._build();
  }

  _build() {
    while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);
    this._buildSideMarkers();
    this.cellGroups = CELLS.map((c, i) => {
      const [cx, cy] = cellCenter(c);
      const g = document.createElementNS(SVGNS, 'g');
      g.setAttribute('class', 'cell');
      g.dataset.idx = i;

      const bg = document.createElementNS(SVGNS, 'polygon');
      bg.setAttribute('points', hexPoints(cx, cy));
      bg.setAttribute('class', 'cell-bg');
      // Colors come from CSS custom properties, which are only resolved when set
      // as CSS declarations (inline style) — NOT as SVG presentation attributes.
      bg.style.fill = 'var(--empty)';
      bg.style.stroke = 'var(--empty-line)';
      bg.setAttribute('stroke-width', '0.7');
      g.appendChild(bg);

      g.addEventListener('click', () => {
        if (this.onCellClick) this.onCellClick(i);
      });
      this.svg.appendChild(g);
      return { g, bg, cx, cy, stone: null, marks: [] };
    });
  }

  // Coloured line segments drawn just outside the outermost cells to mark which
  // pair of opposite sides belongs to which player. Player p owns the two sides
  // where cube-coordinate axis p equals +R / -R (see SIDES in engine.js); each
  // side is drawn as a straight segment between the outer corners of its two
  // corner cells, offset outward so it sits just off the rock.
  _buildSideMarkers() {
    const colorVars = COLOR_VAR;
    const GAP = SIZE * 0.35;
    for (let ax = 0; ax < 3; ax++) {
      const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
      for (const s of [R, -R]) {
        // outward normal: direction from the board centre to the side's centroid
        let sx = 0, sy = 0, count = 0;
        CELLS.forEach((c) => {
          if (c[ax] === s) { const [px, py] = cellCenter(c); sx += px; sy += py; count++; }
        });
        sx /= count; sy /= count;
        const len = Math.hypot(sx, sy) || 1;
        const nx = sx / len, ny = sy / len;
        // the two corner cells of this side (a second axis is also at ±R)
        const corners = CELLS.filter((c) => c[ax] === s && (Math.abs(c[o1]) === R || Math.abs(c[o2]) === R));
        const ends = corners.map((c) => {
          const [cx, cy] = cellCenter(c);
          // outermost hex vertex along the outward normal
          let best = null, bestDot = -Infinity;
          for (let k = 0; k < 6; k++) {
            const a = (Math.PI / 180) * (60 * k);
            const vx = cx + SIZE * Math.cos(a), vy = cy + SIZE * Math.sin(a);
            const dot = vx * nx + vy * ny;
            if (dot > bestDot) { bestDot = dot; best = [vx, vy]; }
          }
          return [best[0] + nx * GAP, best[1] + ny * GAP];
        });
        if (ends.length < 2) continue;
        const line = document.createElementNS(SVGNS, 'line');
        line.setAttribute('x1', ends[0][0].toFixed(2));
        line.setAttribute('y1', ends[0][1].toFixed(2));
        line.setAttribute('x2', ends[1][0].toFixed(2));
        line.setAttribute('y2', ends[1][1].toFixed(2));
        line.setAttribute('class', 'side-marker');
        line.style.stroke = colorVars[ax];
        this.sideSegments[ax][s === R ? 'plus' : 'minus'] = {
          x1: ends[0][0], y1: ends[0][1], x2: ends[1][0], y2: ends[1][1],
        };
        // Beak anchor: the +R side's midpoint pushed further outward, standing in
        // for the bird's beak that exchanges flecks with the board.
        if (s === R) {
          const mx = (ends[0][0] + ends[1][0]) / 2, my = (ends[0][1] + ends[1][1]) / 2;
          this.beakAnchors[ax] = [mx + nx * SIZE * 2.2, my + ny * SIZE * 2.2];
        }
        this.svg.appendChild(line);
      }
    }
  }

  // state: { board, beaks, playable:Set<int>, born:[{i,color}], died:[int],
  //          showHighlights:bool, anim }
  // `anim` (optional) drives the live-move animations and is one of the phases
  // produced by the controller:
  //   { durMs, fly:{cell,type,color}|null,
  //     grow:Set<int>|null,            // born cells to scale up from centre
  //     shrink:Map<int,color>|null }   // dying cells to scale down (ghost stones)
  render(state) {
    const {
      board, playable, born = [], died = [], showHighlights = true, fadeMs = 0,
      winningPath = null, winner = null, anim = null,
    } = state;
    const bornSet = new Set(born.map((b) => b.i));
    const diedMap = new Map(died.map((d) => [d.i, d.color]));
    if (this.winningPath) {
      this.winningPath.remove();
      this.winningPath = null;
    }
    const flyDur = anim ? `${anim.durMs}ms` : null;
    this.cellGroups.forEach((cg, i) => {
      // clear previous stone and marks
      if (cg.stone) { cg.stone.remove(); cg.stone = null; }
      cg.marks.forEach((m) => m.remove());
      cg.marks = [];

      const v = board[i];
      cg.g.classList.toggle('playable', !!playable && playable.has(i));

      if (v !== EMPTY) {
        cg.stone = this._stoneShape(v, cg.cx, cg.cy);
        if (anim) {
          if (anim.grow && anim.grow.has(i)) {
            // born cell: grow from the cell centre to full size
            cg.stone.classList.add('grow-in');
            cg.stone.style.setProperty('--grow-dur', flyDur);
          } else if (anim.fly && anim.fly.type === MOVE_DROP && anim.fly.cell === i) {
            // dropped fleck: fly in from the mover's beak
            this._applyFly(cg.stone, i, anim.fly.color, 'fly-in', flyDur);
          }
        }
        cg.g.appendChild(cg.stone);
      } else if (anim) {
        // Empty cell that needs a transient ghost stone for its animation.
        if (anim.shrink && anim.shrink.has(i)) {
          // dying cell: shrink a ghost of its former colour toward the centre
          const ghost = this._stoneShape(anim.shrink.get(i), cg.cx, cg.cy);
          ghost.classList.add('shrink-out');
          ghost.style.setProperty('--grow-dur', flyDur);
          cg.g.appendChild(ghost); cg.marks.push(ghost);
        }
        if (anim.fly && anim.fly.type !== MOVE_DROP && anim.fly.cell === i) {
          // picked fleck: fly a ghost of the mover's stone out to the beak
          const ghost = this._stoneShape(anim.fly.color, cg.cx, cg.cy);
          this._applyFly(ghost, i, anim.fly.color, 'fly-out', flyDur);
          cg.g.appendChild(ghost); cg.marks.push(ghost);
        }
      }
      if (showHighlights && bornSet.has(i)) {
        const ring = document.createElementNS(SVGNS, 'circle');
        ring.setAttribute('cx', cg.cx); ring.setAttribute('cy', cg.cy);
        ring.setAttribute('r', SIZE * 0.85);
        ring.setAttribute('class', 'born-ring');
        if (fadeMs > 0) { ring.classList.add('marker-fade'); ring.style.animationDuration = `${fadeMs}ms`; }
        cg.g.appendChild(ring); cg.marks.push(ring);
      }
      if (showHighlights && diedMap.has(i)) {
        const x = document.createElementNS(SVGNS, 'path');
        const d = SIZE * 0.28;
        x.setAttribute('d', `M ${cg.cx - d} ${cg.cy - d} L ${cg.cx + d} ${cg.cy + d} M ${cg.cx + d} ${cg.cy - d} L ${cg.cx - d} ${cg.cy + d}`);
        x.setAttribute('class', 'died-mark');
        x.style.stroke = COLOR_VAR[diedMap.get(i)];
        if (fadeMs > 0) { x.classList.add('marker-fade'); x.style.animationDuration = `${fadeMs}ms`; }
        cg.g.appendChild(x); cg.marks.push(x);
      }
    });
    if (winner !== null && winningPath && winningPath.length) {
      this.winningPath = this._winningPolyline(winner, winningPath);
      this.svg.appendChild(this.winningPath);
    }
  }

  // Give `el` a fleck-fly animation between the mover's beak and cell `i`.
  // `dir` is 'fly-in' (beak -> cell) or 'fly-out' (cell -> beak).
  _applyFly(el, i, color, dir, durMs) {
    const anchor = this.beakAnchors[color] || [this.cellGroups[i].cx, this.cellGroups[i].cy];
    const dx = anchor[0] - this.cellGroups[i].cx;
    const dy = anchor[1] - this.cellGroups[i].cy;
    el.classList.add(dir);
    el.style.setProperty('--fx', `${dx.toFixed(2)}px`);
    el.style.setProperty('--fy', `${dy.toFixed(2)}px`);
    el.style.setProperty('--fly-dur', durMs);
  }

  _winningPolyline(player, path) {
    const project = (point, segment) => {
      const dx = segment.x2 - segment.x1;
      const dy = segment.y2 - segment.y1;
      const denom = dx * dx + dy * dy || 1;
      let t = ((point[0] - segment.x1) * dx + (point[1] - segment.y1) * dy) / denom;
      t = Math.max(0, Math.min(1, t));
      return [segment.x1 + dx * t, segment.y1 + dy * t];
    };
    const points = path.map((i) => cellCenter(CELLS[i]));
    const plus = project(points[0], this.sideSegments[player].plus);
    const minus = project(points[points.length - 1], this.sideSegments[player].minus);
    const poly = document.createElementNS(SVGNS, 'polyline');
    poly.setAttribute(
      'points',
      [plus, ...points, minus].map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' '),
    );
    poly.setAttribute('class', 'winning-connection');
    poly.style.stroke = COLOR_VAR[player];
    // Glow a bit lighter than the stones themselves; CSS fades this in then
    // pulses it (see .winning-connection in style.css).
    poly.style.setProperty('--win-glow', COLOR_SOFT[player]);
    return poly;
  }

  // All stones use 60°-rotationally-symmetric shapes so no orientation is implied:
  // Red is a hexagon, Yellow a six-pointed star, Blue a circle.
  _stoneShape(color, cx, cy) {
    const r = SIZE * 0.62;
    let el;
    if (color === 0) { // Red hexagon (flat-top)
      el = document.createElementNS(SVGNS, 'polygon');
      const pts = [];
      for (let k = 0; k < 6; k++) {
        const a = (Math.PI / 180) * (60 * k);
        pts.push(`${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`);
      }
      el.setAttribute('points', pts.join(' '));
    } else if (color === 1) { // Yellow six-pointed star
      el = document.createElementNS(SVGNS, 'polygon');
      const pts = [];
      for (let k = 0; k < 12; k++) {
        const a = (Math.PI / 180) * (30 * k - 90);
        const rr = (k % 2 === 0) ? r : r * 0.5;
        pts.push(`${(cx + rr * Math.cos(a)).toFixed(2)},${(cy + rr * Math.sin(a)).toFixed(2)}`);
      }
      el.setAttribute('points', pts.join(' '));
    } else { // Blue circle
      el = document.createElementNS(SVGNS, 'circle');
      el.setAttribute('cx', cx); el.setAttribute('cy', cy); el.setAttribute('r', r * 0.92);
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

    const mid = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'player-name';
    name.textContent = `${COLOR_NAME[p]} (${seatTypes[p] === 'ai' ? 'Computer' : 'Human'})`;
    const stats = document.createElement('div');
    stats.className = 'player-stats';
    stats.textContent = `beak ${game.beaks[p]} · on board ${game.cellCount(p)}`;
    mid.appendChild(name); mid.appendChild(stats);
    row.appendChild(mid);

    const right = document.createElement('div');
    right.className = 'player-right';

    const tot = document.createElement('div');
    tot.className = 'player-totals';
    const total = game.totalCells(p);
    tot.textContent = `${total} total`;
    // A player below 7 total cannot form a winning chain until they breed back
    // up — the real content of an attrition attack. Flag it.
    if (total < 7) {
      tot.classList.add('below-seven');
      tot.title = 'Below 7 — cannot connect until this bird breeds back up.';
    }
    right.appendChild(tot);

    // One shape per stone currently held in the beak, in this player's shape.
    const beak = document.createElement('div');
    beak.className = 'player-beak';
    beak.setAttribute('aria-label', `beak ${game.beaks[p]}`);
    for (let k = 0; k < game.beaks[p]; k++) {
      const s = document.createElement('span');
      s.className = `beak-chip chip-${p}`;
      s.setAttribute('aria-hidden', 'true');
      beak.appendChild(s);
    }
    right.appendChild(beak);
    row.appendChild(right);

    container.appendChild(row);
  }
}

export { COLOR_NAME };

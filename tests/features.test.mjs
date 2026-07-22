// Feature-parity test: the JavaScript feature extractor (web/js/features.js) must
// produce exactly the same vectors as the Python trainer (training/hexlife37.py),
// otherwise weights trained in Python would not transfer to the browser. The
// fixture tests/feature_vectors.json holds Python reference features; regenerate it
// with tests/gen_reference or the helper in the repo docs if the feature layout
// ever changes.
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { computeFeatures, FEAT_LEN } from '../web/js/features.js';

const here = dirname(fileURLToPath(import.meta.url));
const { samples } = JSON.parse(readFileSync(join(here, 'feature_vectors.json'), 'utf8'));

let maxDiff = 0;
let mismatched = 0;
for (const s of samples) {
  const f = computeFeatures(Int8Array.from(s.board), s.beaks, s.p);
  assert.strictEqual(f.length, FEAT_LEN);
  assert.strictEqual(f.length, s.feat.length);
  for (let i = 0; i < f.length; i++) {
    const d = Math.abs(f[i] - s.feat[i]);
    if (d > maxDiff) maxDiff = d;
    if (d > 1e-4) mismatched++;
  }
}

if (mismatched === 0) {
  console.log(`feature parity OK: ${samples.length} positions, max abs diff ${maxDiff.toExponential(2)}`);
  process.exit(0);
} else {
  console.error(`feature parity FAILED: ${mismatched} entries differ (max ${maxDiff})`);
  process.exit(1);
}

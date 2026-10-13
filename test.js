'use strict';
// 簡易テスト: ルール(perft)と逆伝播(数値微分)の確認
const assert = require('assert');
const Othello = require('./othello');
const { Net, mulberry32 } = require('./dqn');

// --- perft: 既知の値 4, 12, 56, 244, 1396, 8200, 55092 (パスは手数に数えない標準定義) ---
function perft(board, player, depth) {
  if (depth === 0) return 1;
  const mask = new Uint8Array(64);
  let n = Othello.legalMask(board, player, mask);
  let p = player;
  if (n === 0) {
    p = -player;
    n = Othello.legalMask(board, p, mask);
    if (n === 0) return 1;
  }
  let total = 0;
  for (let c = 0; c < 64; c++) {
    if (!mask[c]) continue;
    const b = board.slice();
    Othello.applyMove(b, p, c);
    total += perft(b, -p, depth - 1);
  }
  return total;
}
const expected = [1, 4, 12, 56, 244, 1396, 8200, 55092];
for (let d = 1; d <= 7; d++) {
  const got = perft(Othello.initialBoard(), 1, d);
  assert.strictEqual(got, expected[d], `perft(${d}) = ${got}, expected ${expected[d]}`);
}
console.log('perft OK');

// --- 勾配チェック ---
const rng = mulberry32(5);
const net = new Net([128, 16, 8, 64], rng);
const B = 4;
const A = net.makeBuffers(B), D = net.sizes.map((n) => new Float32Array(B * n));
for (let i = 0; i < A[0].length; i++) A[0][i] = rng() < 0.3 ? 1 : 0;
const act = Uint8Array.from([3, 10, 40, 63]);
const target = Float32Array.from([0.5, -0.3, 1, -1]);
function loss() {
  net.forward(A, B);
  let l = 0;
  for (let s = 0; s < B; s++) { const e = A[3][s * 64 + act[s]] - target[s]; l += 0.5 * e * e; }
  return l;
}
net.forward(A, B);
const dy = new Float32Array(B);
for (let s = 0; s < B; s++) dy[s] = A[3][s * 64 + act[s]] - target[s];
const G = { W: net.W.map((w) => new Float32Array(w.length)), b: net.b.map((b) => new Float32Array(b.length)) };
net.backward(A, B, act, dy, G, D);
let maxRel = 0, nonzero = 0;
for (let l = 0; l < net.L; l++) {
  for (let k = 0; k < 30; k++) {
    const i = Math.floor(rng() * net.W[l].length);
    const orig = net.W[l][i], h = 1e-2;
    net.W[l][i] = orig + h; const lp = loss();
    net.W[l][i] = orig - h; const lm = loss();
    net.W[l][i] = orig;
    const num = (lp - lm) / (2 * h);
    const ana = G.W[l][i];
    const rel = Math.abs(num - ana) / Math.max(1e-3, Math.abs(num) + Math.abs(ana));
    if (rel > maxRel) maxRel = rel;
    if (Math.abs(num) > 1e-6) nonzero++;
  }
}
console.log("nonzero checks:", nonzero, "/ 90; max rel err =", maxRel.toFixed(4));
assert.ok(maxRel < 0.05, 'gradient mismatch');
console.log('gradient OK');

// --- Web UI 用の推論/対局ロジックが学習側(dqn.js)と一致するか ---
{
  const fs = require('fs');
  const path = require('path');
  const { netFromJSON } = require('./dqn');
  const NN = require('./public/nn');
  const Game = require('./public/game');
  const file = path.join(__dirname, 'output', 'model_final.json');
  const json = JSON.parse(fs.readFileSync(file, 'utf8'));
  const ref = netFromJSON(json), web = NN.fromJSON(json);
  const A = ref.makeBuffers(1);
  const r2 = mulberry32(99);
  let maxDiff = 0;
  for (let t = 0; t < 20; t++) {
    const board = Othello.initialBoard();
    let p = 1;
    for (let k = 0; k < t * 3; k++) { // ランダムに進めて色々な局面を作る
      const m = new Uint8Array(64);
      if (Othello.legalMask(board, p, m) === 0) { p = -p; if (Othello.legalMask(board, p, m) === 0) break; }
      const cs = []; for (let c = 0; c < 64; c++) if (m[c]) cs.push(c);
      Othello.applyMove(board, p, cs[Math.floor(r2() * cs.length)]); p = -p;
    }
    for (const pl of [1, -1]) {
      A[0].fill(0);
      for (let i = 0; i < 64; i++) { const v = board[i] * pl; if (v === 1) A[0][i] = 1; else if (v === -1) A[0][64 + i] = 1; }
      ref.forward(A, 1);
      const q = web.qValues(board, pl);
      for (let i = 0; i < 64; i++) maxDiff = Math.max(maxDiff, Math.abs(q[i] - A[ref.L][i]));
    }
  }
  console.log('web/nn.js vs dqn.js max |Q diff| =', maxDiff);
  assert.ok(maxDiff < 1e-6, 'web inference mismatch');

  // 1局ランダム vs AI を最後まで進め、整合性(石数・パス・終局)を確認
  let st = Game.newGame(web), moves = 0, r3 = mulberry32(3);
  while (!st.over) {
    let cell = null;
    if (moves % 2 === 0) { const l = []; st.legal.forEach((v, i) => v && l.push(i)); cell = l[Math.floor(r3() * l.length)]; }
    st = Game.step(st, cell, web); moves++;
    assert.ok(moves < 200, 'game did not terminate');
  }
  const [b, w] = Othello.countDiscs(Int8Array.from(st.board));
  assert.strictEqual(st.black, b); assert.strictEqual(st.white, w);
  assert.strictEqual(st.winner, b > w ? 1 : b < w ? -1 : 0);
  assert.throws(() => Game.step(Game.newGame(web), 0, web), /illegal move/);
  console.log('Game logic OK (moves=' + moves + ', ' + b + '-' + w + ')');
}

// --- 探索(public/search.js): 深さ1は従来の最大Q、深さ2以上は素朴なminimaxと同じ価値になるか ---
{
  const fs = require('fs');
  const path = require('path');
  const NN = require('./public/nn');
  const { makeSearch } = require('./public/search');
  const Game = require('./public/game');
  const net = NN.fromJSON(JSON.parse(fs.readFileSync(path.join(__dirname, 'output', 'model_final.json'), 'utf8')));
  function plain(board, p, rem) { // alpha-beta なしの参照実装
    const m = new Uint8Array(64);
    if (Othello.legalMask(board, p, m) === 0) {
      const m2 = new Uint8Array(64);
      if (Othello.legalMask(board, -p, m2) === 0) { const [b, w] = Othello.countDiscs(board); const d = (b - w) * p; return d > 0 ? 1 : d < 0 ? -1 : 0; }
      return -plain(board, -p, rem);
    }
    if (rem === 1) { const q = net.qValues(board, p); let mx = -Infinity; for (let i = 0; i < 64; i++) if (m[i] && q[i] > mx) mx = q[i]; return mx; }
    let best = -Infinity;
    for (let i = 0; i < 64; i++) { if (!m[i]) continue; const c = board.slice(); Othello.applyMove(c, p, i); best = Math.max(best, -plain(c, -p, rem - 1)); }
    return best;
  }
  const rs = mulberry32(11);
  let checked = 0;
  for (let t = 0; t < 40; t++) {
    const board = Othello.initialBoard();
    let p = 1;
    for (let k = 0, n = Math.floor(rs() * 58); k < n; k++) {
      const m = new Uint8Array(64);
      if (Othello.legalMask(board, p, m) === 0) { p = -p; if (Othello.legalMask(board, p, m) === 0) break; }
      const cs = []; for (let c = 0; c < 64; c++) if (m[c]) cs.push(c);
      Othello.applyMove(board, p, cs[Math.floor(rs() * cs.length)]); p = -p;
    }
    const mask = new Uint8Array(64);
    if (Othello.legalMask(board, p, mask) === 0) continue;
    checked++;
    const q = net.qValues(board, p);
    let am = -1, aq = -Infinity; for (let i = 0; i < 64; i++) if (mask[i] && q[i] > aq) { aq = q[i]; am = i; }
    const r1 = makeSearch(net, 1)(board, p, mask);
    assert.strictEqual(r1.move, am);
    for (const d of [2, 3, 4]) {
      const r = makeSearch(net, d)(board, p, mask);
      const c = board.slice(); Othello.applyMove(c, p, r.move);
      const vChosen = -plain(c, -p, d - 1);
      assert.ok(Math.abs(vChosen - r.value) < 1e-6, 'search value mismatch at depth ' + d);
      let bv = -Infinity;
      for (let i = 0; i < 64; i++) { if (!mask[i]) continue; const c2 = board.slice(); Othello.applyMove(c2, p, i); bv = Math.max(bv, -plain(c2, -p, d - 1)); }
      assert.ok(Math.abs(bv - r.value) < 1e-6, 'search is not optimal at depth ' + d);
    }
  }
  // Game.step が深さ指定で動く
  const s0 = Game.newGame(net);
  const s1 = Game.step(s0, null, net, 3);
  assert.ok(s1.move !== null && s1.value !== null);
  // 検討用: analyze は step と同じ手・同じ評価値を返し、終局局面では null
  const an = Game.analyze(s0, net, 3);
  assert.strictEqual(an.move, s1.move);
  assert.ok(Math.abs(an.value - s1.value) < 1e-9);
  const bd = new Array(64).fill(1);
  assert.strictEqual(Game.analyze({ board: bd, player: 1 }, net, 3), null);
  console.log('Search OK (' + checked + ' positions, depth 1-4)');
}

// --- 特徴量 v2(着手可能マス + スカラー): 対称変換で正しく同変か、web推論と学習側が一致するか ---
{
  const Features = require('./public/features');
  const NN = require('./public/nn');
  const { Net, SYM, netToJSON } = require('./dqn');
  const rf = mulberry32(2024);
  const nin = Features.SPECS.v2.n;
  let checked = 0;
  for (let t = 0; t < 30; t++) {
    const board = Othello.initialBoard();
    let p = 1;
    for (let k = 0, n = 2 + Math.floor(rf() * 50); k < n; k++) {
      const m = new Uint8Array(64);
      if (Othello.legalMask(board, p, m) === 0) { p = -p; if (Othello.legalMask(board, p, m) === 0) break; }
      const cs = []; for (let c = 0; c < 64; c++) if (m[c]) cs.push(c);
      Othello.applyMove(board, p, cs[Math.floor(rf() * cs.length)]); p = -p;
    }
    const persp = new Int8Array(64), me = new Uint8Array(64), opp = new Uint8Array(64);
    Othello.perspective(board, p, persp);
    Othello.legalMask(board, p, me); Othello.legalMask(board, -p, opp);
    for (const perm of SYM) {
      const moved = new Int8Array(64);
      for (let c = 0; c < 64; c++) moved[perm[c]] = board[c];
      const direct = new Float32Array(nin); Features.encodeFromBoard('v2', moved, p, direct, 0);
      const viaPerm = new Float32Array(nin); Features.encode('v2', persp, me, opp, viaPerm, 0, perm);
      for (let i = 0; i < nin; i++) assert.strictEqual(direct[i], viaPerm[i], 'v2 features not equivariant');
      checked++;
    }
  }
  // ランダム重みの v2 ネットで、学習側(Net.forward)と web(QNet)の出力が一致
  const net = new Net([nin, 32, 16, 64], mulberry32(5), 'v2');
  const web = NN.fromJSON(netToJSON(net));
  assert.strictEqual(web.features, 'v2');
  const A = net.makeBuffers(1);
  let md = 0;
  for (let t = 0; t < 10; t++) {
    const board = Othello.initialBoard(); let p = 1;
    for (let k = 0; k < t * 5; k++) {
      const m = new Uint8Array(64);
      if (Othello.legalMask(board, p, m) === 0) { p = -p; if (Othello.legalMask(board, p, m) === 0) break; }
      const cs = []; for (let c = 0; c < 64; c++) if (m[c]) cs.push(c);
      Othello.applyMove(board, p, cs[Math.floor(rf() * cs.length)]); p = -p;
    }
    A[0].fill(0); Features.encodeFromBoard('v2', board, p, A[0], 0);
    net.forward(A, 1);
    const q = web.qValues(board, p);
    for (let i = 0; i < 64; i++) md = Math.max(md, Math.abs(q[i] - A[net.L][i]));
  }
  assert.ok(md < 1e-6, 'v2 web/train mismatch');
  console.log('Features v2 OK (' + checked + ' symmetry checks, web diff ' + md + ')');
}

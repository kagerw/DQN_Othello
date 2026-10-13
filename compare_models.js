'use strict';
// 2つのモデルを直接対戦させる(先手後手交代、序盤 --open 手はランダム)。
//   node compare_models.js <modelA.json> <modelB.json> [--games 1000] [--depth 1] [--depthB 同じ] [--open 4] [--seed 1]
// A 視点のスコア(勝ち + 0.5×引き分け)と 95% 信頼区間を表示する。
const fs = require('fs');
const path = require('path');
const { playGame } = require('./evaluate');
const { makeSearchPolicy, loadNet } = require('./search');
const { mulberry32 } = require('./dqn');

const argv = process.argv.slice(2);
const files = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--')));
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const games = parseInt(opt('games', '1000'), 10);
const depthA = parseInt(opt('depth', '1'), 10);
const depthB = parseInt(opt('depthB', String(depthA)), 10);
const openRandom = parseInt(opt('open', '4'), 10);
const rng = mulberry32(parseInt(opt('seed', '1'), 10));

const load = (f) => loadNet(JSON.parse(fs.readFileSync(path.resolve(f), 'utf8')));
const A = makeSearchPolicy(load(files[0]), depthA);
const B = makeSearchPolicy(load(files[1]), depthB);
let win = 0, draw = 0, loss = 0;
for (let g = 0; g < games; g++) {
  const aBlack = g % 2 === 0;
  const r = aBlack ? playGame(A, B, rng, openRandom) : -playGame(B, A, rng, openRandom);
  if (r > 0) win++; else if (r < 0) loss++; else draw++;
}
const score = (win + 0.5 * draw) / games;
const ci = 1.96 * Math.sqrt(Math.max(score * (1 - score), 1e-9) / games);
console.log(`A=${files[0]} (深さ${depthA})  vs  B=${files[1]} (深さ${depthB})`);
console.log(`A のスコア ${(score * 100).toFixed(1)}% ±${(ci * 100).toFixed(1)}  (勝${win}/分${draw}/負${loss}, ${games}局)`);

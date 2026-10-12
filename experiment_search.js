'use strict';
// 重みはそのまま、探索の深さだけを変えたときの強さを測る実験。
//   node experiment_search.js [--model output/model_final.json] [--depths 1,2,3] [--games 500]
//                             [--h2h 500] [--open 4] [--seed 2026] [--out results/search_experiment.json]
// 1) 各深さ vs ランダム / 貪欲   2) 各深さ vs 深さ1(同じ重み)の直接対決
const fs = require('fs');
const path = require('path');
const { playGame, randomPolicy, greedyPolicy } = require('./evaluate');
const { makeSearchPolicy, loadNet } = require('./search');
const { mulberry32 } = require('./dqn');

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const modelFile = path.resolve(__dirname, opt('model', 'output/model_final.json'));
const depths = opt('depths', '1,2,3').split(',').map(Number);
const games = parseInt(opt('games', '500'), 10);
const h2hGames = parseInt(opt('h2h', '500'), 10);
const openRandom = parseInt(opt('open', '4'), 10);
const seed = parseInt(opt('seed', '2026'), 10);
const outFile = path.resolve(__dirname, opt('out', 'results/search_experiment.json'));
const skipBase = argv.includes('--skip-base');

const net = loadNet(JSON.parse(fs.readFileSync(modelFile, 'utf8')));

// a 視点の成績。先手後手を交互に入れ替え、序盤 openRandom 手はランダムで局面を散らす
function match(a, b, n, rng) {
  let win = 0, draw = 0, loss = 0;
  for (let g = 0; g < n; g++) {
    const aBlack = g % 2 === 0;
    const r = aBlack ? playGame(a, b, rng, openRandom) : -playGame(b, a, rng, openRandom);
    if (r > 0) win++; else if (r < 0) loss++; else draw++;
  }
  const score = (win + 0.5 * draw) / n;
  const se = Math.sqrt(Math.max(score * (1 - score), 1e-9) / n);
  return { n, win, draw, loss, score, ci95: 1.96 * se };
}

const fmt = (r) => `${(r.score * 100).toFixed(1)}% ±${(r.ci95 * 100).toFixed(1)} (勝${r.win}/分${r.draw}/負${r.loss})`;

const results = fs.existsSync(outFile) ? JSON.parse(fs.readFileSync(outFile, 'utf8')) : { model: path.basename(modelFile), openRandom, vsBase: {}, h2h: {} };
const t00 = Date.now();

if (!skipBase) {
  for (const d of depths) {
    const t0 = Date.now();
    const rng = mulberry32(seed + d);
    const pol = makeSearchPolicy(net, d);
    const vr = match(pol, randomPolicy(rng), games, rng);
    const vg = match(pol, greedyPolicy(rng), games, rng);
    results.vsBase[d] = { vsRandom: vr, vsGreedy: vg, seconds: (Date.now() - t0) / 1000 };
    console.log(`深さ${d}: vs ランダム ${fmt(vr)} | vs 貪欲 ${fmt(vg)}  [${((Date.now() - t0) / 1000).toFixed(0)}s]`);
  }
}
for (const d of depths) {
  if (d === 1) continue;
  const t0 = Date.now();
  const rng = mulberry32(seed * 7 + d);
  const r = match(makeSearchPolicy(net, d), makeSearchPolicy(net, 1), h2hGames, rng);
  results.h2h[`${d}_vs_1`] = { ...r, seconds: (Date.now() - t0) / 1000 };
  console.log(`深さ${d} vs 深さ1: ${fmt(r)}  [${((Date.now() - t0) / 1000).toFixed(0)}s]`);
}
fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, JSON.stringify(results, null, 2));
console.log(`saved ${outFile}  (total ${((Date.now() - t00) / 1000).toFixed(0)}s)`);

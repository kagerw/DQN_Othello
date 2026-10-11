'use strict';
// 学習済みモデルを ランダム / 貪欲(最多ひっくり返し) と対戦させて勝率を測る。
//   node evaluate.js [model.json] [games=200]
const fs = require('fs');
const path = require('path');
const Othello = require('./othello');
const { netFromJSON, makePolicy, mulberry32 } = require('./dqn');

function randomPolicy(rng) {
  const mask = new Uint8Array(64);
  return (board, player, m) => {
    let n = 0;
    for (let i = 0; i < 64; i++) n += m[i];
    let k = Math.floor(rng() * n);
    for (let i = 0; i < 64; i++) if (m[i] && k-- === 0) return i;
    return -1;
  };
}

function greedyPolicy(rng) {
  const tmp = new Int8Array(32);
  return (board, player, m) => {
    let best = -1, bn = -1, ties = 0;
    for (let i = 0; i < 64; i++) {
      if (!m[i]) continue;
      const n = Othello.getFlips(board, player, i, tmp);
      if (n > bn) { bn = n; best = i; ties = 1; }
      else if (n === bn && rng() * ++ties < 1) best = i;
    }
    return best;
  };
}

// 先手(黒)視点で 1:黒勝ち, 0:引き分け, -1:白勝ち。最初の openRandom 手はランダムで局面を散らす。
function playGame(blackPolicy, whitePolicy, rng, openRandom = 4) {
  const board = Othello.initialBoard();
  const mask = new Uint8Array(64);
  const rand = randomPolicy(rng);
  let player = 1, ply = 0;
  for (;;) {
    let n = Othello.legalMask(board, player, mask);
    if (n === 0) {
      player = -player;
      n = Othello.legalMask(board, player, mask);
      if (n === 0) break;
    }
    const pol = ply < openRandom ? rand : (player === 1 ? blackPolicy : whitePolicy);
    const move = pol(board, player, mask);
    Othello.applyMove(board, player, move);
    player = -player;
    ply++;
  }
  const [b, w] = Othello.countDiscs(board);
  return b > w ? 1 : b < w ? -1 : 0;
}

// agent 視点の成績 {win, draw, loss, score(=勝率+0.5*引分率)}
function match(agentPolicy, opponent, games, rng) {
  let win = 0, draw = 0, loss = 0;
  for (let g = 0; g < games; g++) {
    const agentIsBlack = g % 2 === 0;
    const res = agentIsBlack ? playGame(agentPolicy, opponent, rng) : -playGame(opponent, agentPolicy, rng);
    if (res > 0) win++; else if (res < 0) loss++; else draw++;
  }
  return { win, draw, loss, score: (win + 0.5 * draw) / games };
}

module.exports = { randomPolicy, greedyPolicy, playGame, match };

if (require.main === module) {
  const file = process.argv[2] || path.join(__dirname, 'output', 'model_best.json');
  const games = parseInt(process.argv[3] || '200', 10);
  const net = netFromJSON(JSON.parse(fs.readFileSync(file, 'utf8')));
  const policy = makePolicy(net);
  const rng = mulberry32(Date.now() & 0xffffffff);
  const vr = match(policy, randomPolicy(rng), games, rng);
  const vg = match(policy, greedyPolicy(rng), games, rng);
  console.log(`model: ${file}`);
  console.log(`vs ランダム : 勝${vr.win} 分${vr.draw} 負${vr.loss}  (スコア ${(vr.score * 100).toFixed(1)}%)`);
  console.log(`vs 貪欲     : 勝${vg.win} 分${vg.draw} 負${vg.loss}  (スコア ${(vg.score * 100).toFixed(1)}%)`);
}

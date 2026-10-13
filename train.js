'use strict';
// DQN によるオセロの自己対戦学習。報酬は終局時の勝ち(+1)/引き分け(0)/負け(-1)のみ。
//   node train.js [--minutes 60] [--episodes N] [--hidden 256,128] [--batch 64] [--lr 0.0005]
//                 [--train-every 2] [--eps-start 0.6] [--eps-end 0.05] [--eps-decay-episodes 15000]
//                 [--eval-every 500] [--out output] [--resume] [--seed 1] [--features v1|v2]
const fs = require('fs');
const path = require('path');
const Othello = require('./othello');
const { DQN, netToJSON, netFromJSON, optToJSON, optFromJSON, makePolicy, mulberry32 } = require('./dqn');
const { match, randomPolicy, greedyPolicy } = require('./evaluate');

// ---------- 引数 ----------
const args = {
  minutes: Infinity, episodes: Infinity, hidden: '256,128', batch: 64, lr: 0.0005,
  'train-every': 2, 'eps-start': 0.6, 'eps-end': 0.05, 'eps-decay-episodes': 15000,
  'eval-every': 500, 'eval-games': 100, out: 'output', seed: 1, 'replay-size': 200000,
  'target-sync': 1000, warmup: 5000, features: 'v1',
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (!argv[i].startsWith('--')) continue;
  const key = argv[i].slice(2);
  if (key === 'resume') { args.resume = true; continue; }
  args[key] = argv[++i];
}
const num = (k) => Number(args[k]);
const OUT = path.resolve(__dirname, args.out);
fs.mkdirSync(OUT, { recursive: true });

const agent = new DQN({
  hidden: String(args.hidden).split(',').map(Number),
  batch: num('batch'), lr: num('lr'), seed: num('seed'),
  replaySize: num('replay-size'), targetSync: num('target-sync'), features: String(args.features),
});

let episode = 0, elapsedBefore = 0, bestScore = -1;
const ckptPath = path.join(OUT, 'checkpoint.json');
if (args.resume && fs.existsSync(ckptPath)) {
  const c = JSON.parse(fs.readFileSync(ckptPath, 'utf8'));
  agent.online.copyFrom(netFromJSON(c.online));
  agent.target.copyFrom(netFromJSON(c.target));
  optFromJSON(agent.opt, c.opt);
  agent.trainSteps = c.trainSteps; episode = c.episode; elapsedBefore = c.elapsedSec; bestScore = c.bestScore;
  console.log(`resume: episode=${episode} trainSteps=${agent.trainSteps} elapsed=${(elapsedBefore / 60).toFixed(1)}min`);
}

const logPath = path.join(OUT, 'log.csv');
if (!fs.existsSync(logPath)) fs.writeFileSync(logPath, 'episode,minutes,epsilon,loss,train_steps,vs_random,vs_greedy\n');

// ---------- 自己対戦 1 エピソード ----------
const board = new Int8Array(64);
const mask = new Uint8Array(64);
const mask2 = new Uint8Array(64);
const s = new Int8Array(64);
const s2 = new Int8Array(64);
const sOpp = new Uint8Array(64), s2Opp = new Uint8Array(64);
const withLegal = agent.features !== 'v1';
let lossSum = 0, lossN = 0, moveCounter = 0;

function runEpisode(eps) {
  board.set(Othello.initialBoard());
  let player = 1;
  for (;;) {
    if (Othello.legalMask(board, player, mask) === 0) break; // 次の手番は下で必ず合法手持ちを選んでいる
    Othello.perspective(board, player, s);
    if (withLegal) Othello.legalMask(board, -player, sOpp); // 打つ前の局面での相手の着手可能マス(特徴量 v2 用)
    const a = agent.selectAction(s, mask, eps);
    Othello.applyMove(board, player, a);

    const opp = -player;
    let done = 0, r = 0, sign = 0, next = 0;
    if (Othello.legalMask(board, opp, mask2) > 0) { next = opp; sign = -1; }
    else if (Othello.legalMask(board, player, mask2) > 0) { next = player; sign = 1; }
    else {
      done = 1;
      const [b, w] = Othello.countDiscs(board);
      const diff = (b - w) * player;
      r = diff > 0 ? 1 : diff < 0 ? -1 : 0;
    }
    if (!done) {
      Othello.perspective(board, next, s2);
      if (withLegal) Othello.legalMask(board, -next, s2Opp);
    }
    agent.replay.push(s, a, r, done, s2, mask2, sign, mask, sOpp, s2Opp);

    if (++moveCounter % num('train-every') === 0 && agent.replay.size >= num('warmup')) {
      lossSum += agent.trainStep(); lossN++;
    }
    if (done) return;
    player = next;
  }
}

// ---------- 評価 / 保存 ----------
const evalRng = mulberry32(777);
let lastEval = { vr: 0, vg: 0 };
function evaluate() {
  const policy = makePolicy(agent.online);
  const g = num('eval-games');
  const vr = match(policy, randomPolicy(evalRng), g, evalRng);
  const vg = match(policy, greedyPolicy(evalRng), g, evalRng);
  lastEval = { vr, vg };
  return { vr, vg };
}

function save(final) {
  const elapsedSec = elapsedBefore + (Date.now() - t0) / 1000;
  fs.writeFileSync(path.join(OUT, 'model_latest.json'), JSON.stringify(netToJSON(agent.online)));
  fs.writeFileSync(ckptPath + '.tmp', JSON.stringify({
    online: netToJSON(agent.online), target: netToJSON(agent.target), opt: optToJSON(agent.opt),
    trainSteps: agent.trainSteps, episode, elapsedSec, bestScore,
  }));
  fs.renameSync(ckptPath + '.tmp', ckptPath);
  if (final) fs.writeFileSync(path.join(OUT, 'model_final.json'), JSON.stringify(netToJSON(agent.online)));
}

// ---------- メインループ ----------
const t0 = Date.now();
const maxSec = num('minutes') * 60;
let stop = false;
process.on('SIGINT', () => { stop = true; });
process.on('SIGTERM', () => { stop = true; });
console.log(`network: ${agent.sizes.join('-')}  features: ${agent.features}  out: ${OUT}`);

const maxEpisodes = num('episodes');
while (!stop) {
  const elapsed = elapsedBefore + (Date.now() - t0) / 1000;
  if (elapsed >= maxSec || episode >= maxEpisodes) break;
  const frac = Math.min(1, episode / num('eps-decay-episodes'));
  const eps = num('eps-start') + (num('eps-end') - num('eps-start')) * frac;
  runEpisode(eps);
  episode++;

  if (episode % 100 === 0) {
    const el = (elapsed / 60).toFixed(1);
    console.log(`ep ${episode}  ${el}min  eps=${eps.toFixed(3)}  loss=${lossN ? (lossSum / lossN).toFixed(4) : '-'}  steps=${agent.trainSteps}  buf=${agent.replay.size}`);
    lossSum = 0; lossN = 0;
  }
  if (episode % num('eval-every') === 0) {
    const { vr, vg } = evaluate();
    const el = elapsed / 60;
    console.log(`  [eval] vs ランダム ${(vr.score * 100).toFixed(1)}% (勝${vr.win}/分${vr.draw}/負${vr.loss})  vs 貪欲 ${(vg.score * 100).toFixed(1)}% (勝${vg.win}/分${vg.draw}/負${vg.loss})`);
    fs.appendFileSync(logPath, `${episode},${el.toFixed(2)},${eps.toFixed(3)},${lossN ? (lossSum / lossN).toFixed(5) : ''},${agent.trainSteps},${vr.score.toFixed(3)},${vg.score.toFixed(3)}\n`);
    const score = vr.score + vg.score;
    if (score > bestScore) {
      bestScore = score;
      fs.writeFileSync(path.join(OUT, 'model_best.json'), JSON.stringify(netToJSON(agent.online)));
      console.log('  -> best 更新');
    }
    save(false);
  }
}
save(true);
console.log(`finished: episode=${episode} trainSteps=${agent.trainSteps} elapsed=${((elapsedBefore + (Date.now() - t0) / 1000) / 60).toFixed(1)}min`);

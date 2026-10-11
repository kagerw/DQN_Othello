'use strict';
// 8x8 オセロのルール実装。盤面は Int8Array(64): 1=黒, -1=白, 0=空。cell = row*8+col。

const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

// RAYS[cell] = 各方向について、cell から端まで並ぶセルの配列
const RAYS = [];
for (let cell = 0; cell < 64; cell++) {
  const r0 = cell >> 3, c0 = cell & 7;
  const rays = [];
  for (const [dr, dc] of DIRS) {
    const ray = [];
    let r = r0 + dr, c = c0 + dc;
    while (r >= 0 && r < 8 && c >= 0 && c < 8) {
      ray.push(r * 8 + c);
      r += dr; c += dc;
    }
    if (ray.length >= 2) rays.push(Int8Array.from(ray)); // 1マス先で盤端なら挟めない
  }
  RAYS.push(rays);
}

function initialBoard() {
  const b = new Int8Array(64);
  b[27] = -1; b[28] = 1; b[35] = 1; b[36] = -1;
  return b;
}

// player が cell に打ったときにひっくり返せるセルを out に書き、個数を返す(打てないなら0)
function getFlips(board, player, cell, out) {
  if (board[cell] !== 0) return 0;
  const rays = RAYS[cell];
  let n = 0;
  for (let d = 0; d < rays.length; d++) {
    const ray = rays[d];
    let k = 0;
    while (k < ray.length && board[ray[k]] === -player) k++;
    if (k > 0 && k < ray.length && board[ray[k]] === player) {
      for (let j = 0; j < k; j++) out[n++] = ray[j];
    }
  }
  return n;
}

const _tmp = new Int8Array(32);

function isLegal(board, player, cell) {
  return getFlips(board, player, cell, _tmp) > 0;
}

// 合法手マスクを mask(Uint8Array64) に書き、合法手の数を返す
function legalMask(board, player, mask) {
  let n = 0;
  for (let c = 0; c < 64; c++) {
    const ok = board[c] === 0 && getFlips(board, player, c, _tmp) > 0 ? 1 : 0;
    mask[c] = ok;
    n += ok;
  }
  return n;
}

// 実際に打つ。打てない場合は false。
function applyMove(board, player, cell) {
  const n = getFlips(board, player, cell, _tmp);
  if (n === 0) return false;
  board[cell] = player;
  for (let i = 0; i < n; i++) board[_tmp[i]] = player;
  return true;
}

function countDiscs(board) {
  let b = 0, w = 0;
  for (let i = 0; i < 64; i++) {
    if (board[i] === 1) b++;
    else if (board[i] === -1) w++;
  }
  return [b, w];
}

// 盤面を player 視点(自分=1, 相手=-1)に変換して out に書く
function perspective(board, player, out) {
  for (let i = 0; i < 64; i++) out[i] = board[i] * player;
  return out;
}

const api = { initialBoard, getFlips, isLegal, legalMask, applyMove, countDiscs, perspective };
if (typeof module !== 'undefined' && module.exports) module.exports = api; // Node
else globalThis.Othello = api;                                             // ブラウザ(Web UI)

'use strict';
// ネットワークへの入力(特徴量)の作り方。学習(dqn.js)・Web UI(nn.js)・探索で共通。Node とブラウザの両方で動く。
//   v1 (128)  : 自分の石64 + 相手の石64                               ← 従来
//   v2 (259)  : v1 + 自分の着手可能マス64 + 相手の着手可能マス64
//                  + スカラー3つ(空きマス数/60, 自分の着手可能数/20, 相手の着手可能数/20)
// 「自分」= 手番側。盤面の対称変換(perm)を掛けても、石/着手可能マスの面は同じ並べ替えで、スカラーは不変。
(function (root) {
  const isNode = typeof module !== 'undefined' && module.exports;
  const Othello = isNode ? require('../othello') : root.Othello;

  const SPECS = { v1: { n: 128 }, v2: { n: 259 } };
  const _me = new Uint8Array(64), _opp = new Uint8Array(64), _persp = new Int8Array(64);

  // persp: 手番側視点の盤面(自分=1, 相手=-1)。myLegal/oppLegal: 各マスに打てるか(0/1)。
  // out[off ...] に書き込む(呼び出し側で事前に 0 クリアしておく)。perm があれば各マスを perm[c] に移して書く。
  function encode(spec, persp, myLegal, oppLegal, out, off, perm) {
    for (let c = 0; c < 64; c++) {
      const nc = perm ? perm[c] : c;
      const v = persp[c];
      if (v === 1) out[off + nc] = 1; else if (v === -1) out[off + 64 + nc] = 1;
    }
    if (spec === 'v2') {
      let empties = 0, mm = 0, om = 0;
      for (let c = 0; c < 64; c++) {
        const nc = perm ? perm[c] : c;
        if (myLegal[c]) { out[off + 128 + nc] = 1; mm++; }
        if (oppLegal[c]) { out[off + 192 + nc] = 1; om++; }
        if (persp[c] === 0) empties++;
      }
      out[off + 256] = empties / 60;
      out[off + 257] = mm / 20;
      out[off + 258] = om / 20;
    }
  }

  // 絶対座標の盤面 board と手番 player から直接エンコード(推論用)。out[off...] は事前に 0 クリアすること。
  function encodeFromBoard(spec, board, player, out, off) {
    for (let i = 0; i < 64; i++) _persp[i] = board[i] * player;
    if (spec === 'v2') {
      Othello.legalMask(board, player, _me);
      Othello.legalMask(board, -player, _opp);
    }
    encode(spec, _persp, _me, _opp, out, off, null);
  }

  const api = { SPECS, encode, encodeFromBoard };
  if (isNode) module.exports = api;
  else root.Features = api;
})(typeof window !== 'undefined' ? window : globalThis);

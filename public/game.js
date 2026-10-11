'use strict';
// 対局の進行ロジック(パス・終局判定、AIの手選び)。Node とブラウザの両方で動く。
(function (root) {
  const Othello = typeof module !== 'undefined' && module.exports ? require('../othello') : root.Othello;

  function describe(board, player, net) {
    const mask = new Uint8Array(64);
    Othello.legalMask(board, player, mask);
    const [black, white] = Othello.countDiscs(board);
    return {
      board: Array.from(board), player, legal: Array.from(mask), black, white,
      q: net.qValues(board, player),
      over: false, passed: false, move: null, flipped: [], value: null, winner: null,
    };
  }

  function newGame(net) {
    return describe(Othello.initialBoard(), 1, net);
  }

  // state: {board, player}。cell が null/undefined なら net が選ぶ手(最大Q)を打つ。
  // 返り値は次の状態。相手が打てなければ手番は変わらず passed=true、双方打てなければ over=true。
  function step(state, cell, net) {
    const board = Int8Array.from(state.board);
    const player = state.player;
    const mask = new Uint8Array(64);
    if (Othello.legalMask(board, player, mask) === 0) throw new Error('no legal move');

    let value = null;
    if (cell === null || cell === undefined) {
      const q = net.qValues(board, player);
      let bq = -Infinity;
      for (let i = 0; i < 64; i++) if (mask[i] && q[i] > bq) { bq = q[i]; cell = i; }
      value = bq;
    } else if (!Number.isInteger(cell) || cell < 0 || cell > 63 || !mask[cell]) {
      throw new Error('illegal move');
    }

    const before = board.slice();
    Othello.applyMove(board, player, cell);
    const flipped = [];
    for (let i = 0; i < 64; i++) if (i !== cell && before[i] !== board[i]) flipped.push(i);

    let next = -player, passed = false, over = false;
    if (Othello.legalMask(board, next, mask) === 0) {
      if (Othello.legalMask(board, player, mask) > 0) { next = player; passed = true; }
      else over = true;
    }
    const res = describe(board, next, net);
    res.move = cell; res.flipped = flipped; res.passed = passed; res.over = over; res.value = value;
    if (over) {
      res.legal = new Array(64).fill(0);
      res.winner = res.black > res.white ? 1 : res.black < res.white ? -1 : 0;
    }
    return res;
  }

  const api = { newGame, step };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Game = api;
})(typeof window !== 'undefined' ? window : globalThis);

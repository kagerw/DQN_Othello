'use strict';
// 学習済みQ値を評価関数にした minimax(negamax + alpha-beta)。重みは一切変えない。Node とブラウザの両方で動く。
//
//   depth = 1 : 探索なし。その手の Q(s,a) が最大の手を打つ(従来のDQNの打ち方)。
//   depth = n : n手先まで読む。最後の1手(葉)は「その手番側の max_a Q(s,a)」で評価する。
//               終局に達したら実際の勝敗(+1/0/-1)、パスは手数に数えずそのまま読み進める。
//   つまり「3手読み」= 自分の手 → 相手の返し → 自分の手(その価値を Q で見積もる)。
(function (root) {
  const Othello = typeof module !== 'undefined' && module.exports ? require('../othello') : root.Othello;

  // net.qValues(board, player) -> 64個のQ値 を持つオブジェクトを渡す。
  // 返り値 choose(board, player, mask) は { move, value }(value = 手番側から見た評価 -1〜+1)
  function makeSearch(net, depth) {
    const masks = [], boards = [];
    for (let d = 0; d <= depth + 2; d++) { masks.push(new Uint8Array(64)); boards.push(new Int8Array(64)); }
    const tmpMask = new Uint8Array(64);

    // player が手番の局面の価値(player 視点)。remaining = 残り手数(>=1)
    function negamax(board, player, remaining, alpha, beta, level) {
      const mask = masks[level];
      if (Othello.legalMask(board, player, mask) === 0) {
        if (Othello.legalMask(board, -player, tmpMask) === 0) { // 双方打てない = 終局
          const [b, w] = Othello.countDiscs(board);
          const diff = (b - w) * player;
          return diff > 0 ? 1 : diff < 0 ? -1 : 0;
        }
        return -negamax(board, -player, remaining, -beta, -alpha, level); // パス
      }
      const q = net.qValues(board, player);
      if (remaining === 1) {
        let m = -Infinity;
        for (let i = 0; i < 64; i++) if (mask[i] && q[i] > m) m = q[i];
        return m;
      }
      // 有望そうな手(Qが高い順)から読むと alpha-beta がよく効く
      const moves = [];
      for (let i = 0; i < 64; i++) if (mask[i]) moves.push(i);
      moves.sort((a, b) => q[b] - q[a]);
      const child = boards[level + 1];
      let best = -Infinity;
      for (const mv of moves) {
        child.set(board);
        Othello.applyMove(child, player, mv);
        const v = -negamax(child, -player, remaining - 1, -beta, -alpha, level + 1);
        if (v > best) best = v;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
      }
      return best;
    }

    return function choose(board, player, mask) {
      const q = net.qValues(board, player);
      const moves = [];
      for (let i = 0; i < 64; i++) if (mask[i]) moves.push(i);
      moves.sort((a, b) => q[b] - q[a]);
      if (depth <= 1) return { move: moves[0], value: q[moves[0]] };
      const child = boards[0];
      let best = moves[0], bv = -Infinity;
      for (const mv of moves) {
        child.set(board);
        Othello.applyMove(child, player, mv);
        // 現時点の最善 bv を超える手だけが必要なので、その下限を窓に使う(同値以下は打ち切られる)
        const v = -negamax(child, -player, depth - 1, -Infinity, -bv, 1);
        if (v > bv) { bv = v; best = mv; }
      }
      return { move: best, value: bv };
    };
  }

  const api = { makeSearch };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Search = api;
})(typeof window !== 'undefined' ? window : globalThis);

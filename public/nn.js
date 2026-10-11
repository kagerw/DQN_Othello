'use strict';
// 学習済みモデル(output/model_*.json)の推論専用コード。Node とブラウザの両方で動く。
// 学習側の dqn.js と同じ計算(ReLU隠れ層 + tanh出力)。test.js で出力一致を検証している。
(function (root) {
  function b64ToF32(s) {
    const bin = atob(s);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  class QNet {
    constructor(json) {
      this.sizes = json.sizes;
      this.L = this.sizes.length - 1;
      this.W = json.W.map(b64ToF32);
      this.b = json.b.map(b64ToF32);
      this.acts = this.sizes.map((n) => new Float32Array(n));
    }

    // player 視点(自分の石=入力0..63, 相手の石=入力64..127)で全64マスのQ値(-1〜+1)を返す
    qValues(board, player) {
      const A = this.acts, S = this.sizes;
      A[0].fill(0);
      for (let i = 0; i < 64; i++) {
        const v = board[i] * player;
        if (v === 1) A[0][i] = 1; else if (v === -1) A[0][64 + i] = 1;
      }
      for (let l = 1; l <= this.L; l++) {
        const nin = S[l - 1], nout = S[l];
        const prev = A[l - 1], cur = A[l], W = this.W[l - 1], bias = this.b[l - 1];
        for (let o = 0; o < nout; o++) cur[o] = bias[o];
        for (let i = 0; i < nin; i++) {
          const x = prev[i];
          if (x === 0) continue;
          const wo = i * nout;
          for (let o = 0; o < nout; o++) cur[o] += x * W[wo + o];
        }
        if (l < this.L) { for (let o = 0; o < nout; o++) if (cur[o] < 0) cur[o] = 0; }
        else { for (let o = 0; o < nout; o++) cur[o] = Math.tanh(cur[o]); }
      }
      return Array.from(A[this.L]);
    }
  }

  const api = { QNet, fromJSON: (j) => new QNet(j) };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NN = api;
})(typeof window !== 'undefined' ? window : globalThis);

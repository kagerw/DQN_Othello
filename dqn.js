'use strict';
// 依存ライブラリなしの小さなMLP + Adam + DQN(経験再生/ターゲットネット)。
// Q値は「その手を打った側から見た最終的な勝敗(-1〜+1)」の予測。報酬は終局時の勝ち(+1)/引き分け(0)/負け(-1)のみ。

const Othello = require('./othello');

// ---------- 乱数 ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- 8方向の盤面対称(ルールが対称なのでデータ拡張に使う) ----------
const SYM = [];
for (let k = 0; k < 8; k++) {
  const perm = new Int8Array(64);
  for (let cell = 0; cell < 64; cell++) {
    let r = cell >> 3, c = cell & 7;
    if (k & 1) c = 7 - c;           // 左右反転
    if (k & 2) r = 7 - r;           // 上下反転
    if (k & 4) { const t = r; r = c; c = t; } // 転置
    perm[cell] = r * 8 + c;
  }
  SYM.push(perm);
}

// ---------- MLP ----------
// 入力128(自分の石64 + 相手の石64) → 隠れ層(ReLU)... → 出力64(各マスへ打ったときのQ, tanh)
class Net {
  constructor(sizes, rng) {
    this.sizes = sizes;
    this.L = sizes.length - 1;
    this.W = []; this.b = [];
    for (let l = 0; l < this.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const w = new Float32Array(nin * nout);
      const scale = (l === this.L - 1 ? 0.1 : 1) * Math.sqrt(2 / nin);
      if (rng) for (let i = 0; i < w.length; i++) w[i] = randn(rng) * scale;
      this.W.push(w);
      this.b.push(new Float32Array(nout));
    }
  }

  // バッチ用の活性化バッファ(A[0]は入力)
  makeBuffers(B) {
    return this.sizes.map((n) => new Float32Array(B * n));
  }

  forward(A, B) {
    const L = this.L;
    for (let l = 1; l <= L; l++) {
      const nin = this.sizes[l - 1], nout = this.sizes[l];
      const prev = A[l - 1], cur = A[l], W = this.W[l - 1], bias = this.b[l - 1];
      for (let s = 0; s < B; s++) {
        const zo = s * nout, po = s * nin;
        for (let o = 0; o < nout; o++) cur[zo + o] = bias[o];
        for (let i = 0; i < nin; i++) {
          const x = prev[po + i];
          if (x === 0) continue;
          const wo = i * nout;
          for (let o = 0; o < nout; o++) cur[zo + o] += x * W[wo + o];
        }
        if (l < L) {
          for (let o = 0; o < nout; o++) if (cur[zo + o] < 0) cur[zo + o] = 0;
        } else {
          for (let o = 0; o < nout; o++) cur[zo + o] = Math.tanh(cur[zo + o]);
        }
      }
    }
  }

  // 出力層は「各サンプルで選んだ行動1つだけ」に勾配が来る(DQN)。dy[s]=dLoss/d(出力[s, act[s]])
  backward(A, B, act, dy, G, D) {
    const L = this.L;
    // 出力層
    {
      const nin = this.sizes[L - 1], nout = this.sizes[L];
      const prev = A[L - 1], out = A[L], W = this.W[L - 1];
      const gW = G.W[L - 1], gb = G.b[L - 1];
      const dprev = D[L - 1];
      for (let s = 0; s < B; s++) {
        const o = act[s];
        const y = out[s * nout + o];
        const dz = dy[s] * (1 - y * y);
        gb[o] += dz;
        const po = s * nin;
        for (let i = 0; i < nin; i++) {
          const a = prev[po + i];
          if (a !== 0) gW[i * nout + o] += a * dz;
          dprev[po + i] = W[i * nout + o] * dz;
        }
      }
    }
    // 隠れ層
    for (let l = L - 1; l >= 1; l--) {
      const nin = this.sizes[l - 1], nout = this.sizes[l];
      const prev = A[l - 1], cur = A[l], W = this.W[l - 1];
      const gW = G.W[l - 1], gb = G.b[l - 1];
      const dz = D[l];       // dLoss/d(出力側活性) が入っている → ReLU の微分を掛ける
      const dprev = l > 1 ? D[l - 1] : null;
      for (let s = 0; s < B; s++) {
        const zo = s * nout, po = s * nin;
        for (let o = 0; o < nout; o++) if (cur[zo + o] <= 0) dz[zo + o] = 0;
        for (let o = 0; o < nout; o++) gb[o] += dz[zo + o];
        for (let i = 0; i < nin; i++) {
          const a = prev[po + i];
          const wo = i * nout;
          if (a !== 0) {
            for (let o = 0; o < nout; o++) gW[wo + o] += a * dz[zo + o];
          }
          if (dprev) {
            let sum = 0;
            for (let o = 0; o < nout; o++) sum += W[wo + o] * dz[zo + o];
            dprev[po + i] = sum;
          }
        }
      }
    }
  }

  copyFrom(other) {
    for (let l = 0; l < this.L; l++) {
      this.W[l].set(other.W[l]);
      this.b[l].set(other.b[l]);
    }
  }
}

class Grads {
  constructor(net) {
    this.W = net.W.map((w) => new Float32Array(w.length));
    this.b = net.b.map((b) => new Float32Array(b.length));
  }
  zero() {
    for (const w of this.W) w.fill(0);
    for (const b of this.b) b.fill(0);
  }
}

class Adam {
  constructor(net, lr) {
    this.lr = lr; this.t = 0;
    this.mW = net.W.map((w) => new Float32Array(w.length));
    this.vW = net.W.map((w) => new Float32Array(w.length));
    this.mb = net.b.map((b) => new Float32Array(b.length));
    this.vb = net.b.map((b) => new Float32Array(b.length));
  }
  step(net, G) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    this.t++;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    const lr = this.lr * Math.sqrt(c2) / c1;
    for (let l = 0; l < net.L; l++) {
      upd(net.W[l], G.W[l], this.mW[l], this.vW[l]);
      upd(net.b[l], G.b[l], this.mb[l], this.vb[l]);
    }
    function upd(p, g, m, v) {
      for (let i = 0; i < p.length; i++) {
        const gi = g[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= lr * m[i] / (Math.sqrt(v[i]) + eps);
      }
    }
  }
}

// ---------- 経験再生 ----------
// 1遷移 = (s, a, r, done, s2, mask2, sign)
//   s, s2 : 「打つ側視点」の盤面(自分=1,相手=-1)。s2 は次に打つ側の視点。
//   sign  : 次に打つのが相手なら -1、相手がパスで自分がもう一度打つなら +1
//   target = done ? r : sign * max_{a' 合法} Qtarget(s2, a')
class Replay {
  constructor(capacity) {
    this.cap = capacity; this.size = 0; this.pos = 0;
    this.s = new Int8Array(capacity * 64);
    this.s2 = new Int8Array(capacity * 64);
    this.mask2 = new Uint8Array(capacity * 64);
    this.a = new Uint8Array(capacity);
    this.r = new Float32Array(capacity);
    this.done = new Uint8Array(capacity);
    this.sign = new Int8Array(capacity);
  }
  push(s, a, r, done, s2, mask2, sign) {
    const i = this.pos;
    this.s.set(s, i * 64);
    this.a[i] = a; this.r[i] = r; this.done[i] = done; this.sign[i] = sign;
    if (done) {
      this.s2.fill(0, i * 64, i * 64 + 64);
      this.mask2.fill(0, i * 64, i * 64 + 64);
    } else {
      this.s2.set(s2, i * 64);
      this.mask2.set(mask2, i * 64);
    }
    this.pos = (this.pos + 1) % this.cap;
    if (this.size < this.cap) this.size++;
  }
}

// ---------- エージェント ----------
class DQN {
  constructor(opts) {
    this.hidden = opts.hidden;
    this.sizes = [128, ...opts.hidden, 64];
    this.batch = opts.batch;
    this.gamma = 1; // 勝敗のみの報酬なので割引なし
    this.rng = mulberry32(opts.seed ?? 12345);
    this.online = new Net(this.sizes, this.rng);
    this.target = new Net(this.sizes, null);
    this.target.copyFrom(this.online);
    this.opt = new Adam(this.online, opts.lr);
    this.grads = new Grads(this.online);
    this.replay = new Replay(opts.replaySize);
    this.targetSync = opts.targetSync;
    this.trainSteps = 0;

    const B = opts.batch;
    this.Aon = this.online.makeBuffers(B);
    this.Atg = this.target.makeBuffers(B);
    this.D = this.sizes.map((n) => new Float32Array(B * n)); // 逆伝播用
    this.act = new Uint8Array(B);
    this.dy = new Float32Array(B);
    this.tgt = new Float32Array(B);
    this.A1 = this.online.makeBuffers(1); // 行動選択用(1サンプル)
  }

  // persp: 視点変換済み盤面(自分=1), mask: 合法手。Q値配列を返す
  qValues(persp, net = this.online) {
    const x = this.A1[0];
    x.fill(0);
    for (let i = 0; i < 64; i++) {
      const v = persp[i];
      if (v === 1) x[i] = 1; else if (v === -1) x[64 + i] = 1;
    }
    net.forward(this.A1, 1);
    return this.A1[this.online.L];
  }

  selectAction(persp, mask, eps) {
    // ε-greedy
    if (eps > 0 && this.rng() < eps) {
      let n = 0;
      for (let i = 0; i < 64; i++) n += mask[i];
      let k = Math.floor(this.rng() * n);
      for (let i = 0; i < 64; i++) if (mask[i] && k-- === 0) return i;
    }
    const q = this.qValues(persp);
    let best = -1, bq = -Infinity;
    for (let i = 0; i < 64; i++) {
      if (mask[i] && q[i] > bq) { bq = q[i]; best = i; }
    }
    return best;
  }

  trainStep() {
    const R = this.replay, B = this.batch, rng = this.rng;
    const X = this.Aon[0], X2 = this.Atg[0];
    X.fill(0); X2.fill(0);
    const mask2 = new Uint8Array(B * 64); // 対称変換後の合法手
    const idxs = new Int32Array(B);
    for (let s = 0; s < B; s++) {
      const i = Math.floor(rng() * R.size);
      idxs[s] = i;
      const perm = SYM[Math.floor(rng() * 8)];
      const o = s * 128, base = i * 64;
      for (let c = 0; c < 64; c++) {
        const nc = perm[c];
        const v = R.s[base + c];
        if (v === 1) X[o + nc] = 1; else if (v === -1) X[o + 64 + nc] = 1;
        if (!R.done[i]) {
          const v2 = R.s2[base + c];
          if (v2 === 1) X2[o + nc] = 1; else if (v2 === -1) X2[o + 64 + nc] = 1;
          mask2[s * 64 + nc] = R.mask2[base + c];
        }
      }
      this.act[s] = perm[R.a[i]];
    }
    // ターゲット
    this.target.forward(this.Atg, B);
    const Q2 = this.Atg[this.target.L];
    for (let s = 0; s < B; s++) {
      const i = idxs[s];
      if (R.done[i]) { this.tgt[s] = R.r[i]; continue; }
      let m = -Infinity;
      for (let c = 0; c < 64; c++) {
        if (mask2[s * 64 + c]) { const q = Q2[s * 64 + c]; if (q > m) m = q; }
      }
      this.tgt[s] = R.sign[i] * m;
    }
    // オンライン順伝播 → Huber損失の勾配
    this.online.forward(this.Aon, B);
    const Q = this.Aon[this.online.L];
    let loss = 0;
    for (let s = 0; s < B; s++) {
      const err = Q[s * 64 + this.act[s]] - this.tgt[s];
      const ae = Math.abs(err);
      loss += ae < 1 ? 0.5 * err * err : ae - 0.5;
      this.dy[s] = (ae < 1 ? err : Math.sign(err)) / B;
    }
    this.grads.zero();
    this.online.backward(this.Aon, B, this.act, this.dy, this.grads, this.D);
    this.opt.step(this.online, this.grads);
    this.trainSteps++;
    if (this.trainSteps % this.targetSync === 0) this.target.copyFrom(this.online);
    return loss / B;
  }
}

// ---------- 保存/読み込み ----------
const b64 = (f32) => Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString('base64');
const fromB64 = (s, target) => {
  const buf = Buffer.from(s, 'base64');
  target.set(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
};

function netToJSON(net) {
  return { sizes: net.sizes, W: net.W.map(b64), b: net.b.map(b64) };
}
function netFromJSON(j) {
  const net = new Net(j.sizes, null);
  for (let l = 0; l < net.L; l++) { fromB64(j.W[l], net.W[l]); fromB64(j.b[l], net.b[l]); }
  return net;
}
function optToJSON(opt) {
  return { t: opt.t, lr: opt.lr, mW: opt.mW.map(b64), vW: opt.vW.map(b64), mb: opt.mb.map(b64), vb: opt.vb.map(b64) };
}
function optFromJSON(opt, j) {
  opt.t = j.t;
  for (let l = 0; l < opt.mW.length; l++) {
    fromB64(j.mW[l], opt.mW[l]); fromB64(j.vW[l], opt.vW[l]);
    fromB64(j.mb[l], opt.mb[l]); fromB64(j.vb[l], opt.vb[l]);
  }
}

// ---------- 推論用(学習済みネットで打つ) ----------
function makePolicy(net) {
  const A = net.makeBuffers(1);
  const persp = new Int8Array(64);
  return function policy(board, player, mask) {
    Othello.perspective(board, player, persp);
    const x = A[0];
    x.fill(0);
    for (let i = 0; i < 64; i++) {
      if (persp[i] === 1) x[i] = 1; else if (persp[i] === -1) x[64 + i] = 1;
    }
    net.forward(A, 1);
    const q = A[net.L];
    let best = -1, bq = -Infinity;
    for (let i = 0; i < 64; i++) if (mask[i] && q[i] > bq) { bq = q[i]; best = i; }
    return best;
  };
}

module.exports = { DQN, Net, Adam, SYM, mulberry32, netToJSON, netFromJSON, optToJSON, optFromJSON, makePolicy };

'use strict';
// Web UI のローカル確認用サーバ(依存ライブラリなし)。起動時に dist/ を作り、静的配信するだけ。
//   node server.js [--port 8765] [--host 127.0.0.1]
// 対局のロジックとAIの推論はブラウザ側(public/game.js, public/nn.js)で動く。GitHub Pages と同じ構成。
const http = require('http');
const fs = require('fs');
const path = require('path');
const { build, DIST } = require('./scripts/build-site');

const argv = process.argv.slice(2);
const opt = (name, def) => { const i = argv.indexOf('--' + name); return i >= 0 ? argv[i + 1] : def; };
const PORT = parseInt(opt('port', process.env.PORT || '8765'), 10);
const HOST = opt('host', '127.0.0.1');

build();

const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json' };

http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  const rel = url === '/' ? 'index.html' : url.replace(/^\/+/, '');
  const file = path.resolve(DIST, rel);
  if (!file.startsWith(DIST + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('Not Found');
  }
  res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
  fs.createReadStream(file).pipe(res);
}).listen(PORT, HOST, () => {
  console.log(`DQN Othello Web UI: http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}/`);
});

'use strict';
// 公開用の静的サイト dist/ を作る(GitHub Pages / ローカルの server.js で共通)。
//   node scripts/build-site.js
// dist/ = public/ の中身 + othello.js + models/{v2,v1}.json + models/index.json
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
// 公開するモデルは2つ: 特徴量あり(v2)と特徴量なし(v1)。どちらも学習終了時点の最終モデル。先頭が画面の初期選択。
const MODELS = { v2: 'output_v2/model_final.json', v1: 'output/model_final.json' };

function build() {
  fs.rmSync(DIST, { recursive: true, force: true });
  fs.mkdirSync(path.join(DIST, 'models'), { recursive: true });
  for (const f of fs.readdirSync(path.join(ROOT, 'public'))) {
    fs.copyFileSync(path.join(ROOT, 'public', f), path.join(DIST, f));
  }
  fs.copyFileSync(path.join(ROOT, 'othello.js'), path.join(DIST, 'othello.js'));
  const available = [];
  for (const [name, rel] of Object.entries(MODELS)) {
    const src = path.join(ROOT, rel);
    if (!fs.existsSync(src)) continue;
    fs.copyFileSync(src, path.join(DIST, 'models', name + '.json'));
    available.push(name);
  }
  if (available.length === 0) throw new Error('モデルが見つかりません。先に node train.js で学習してください。');
  fs.writeFileSync(path.join(DIST, 'models', 'index.json'), JSON.stringify(available));
  fs.writeFileSync(path.join(DIST, '.nojekyll'), '');
  return { dist: DIST, models: available };
}

module.exports = { build, DIST };

if (require.main === module) {
  const r = build();
  console.log(`built ${r.dist} (models: ${r.models.join(', ')})`);
}

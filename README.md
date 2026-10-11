# DQN_Othello

Node.js(依存ライブラリなし)で書いた、DQN によるオセロの自己対戦学習。

- **報酬は勝敗のみ**: 終局時に 勝ち +1 / 引き分け 0 / 負け −1。途中の石数などは一切使わない。
- ネットワーク: 入力128(自分の石64 + 相手の石64) → 256 → 128 → 出力64(各マスのQ値, tanh)。
- 手法: 単一ネットでの自己対戦、ε-greedy、経験再生(20万遷移)、ターゲットネット(1000step毎に同期)、Adam、Huber損失。
  盤面の8方向対称でデータ拡張。手番視点で盤面を持つので、相手番のQ値は符号反転して使う(パス時は反転しない)。

## ファイル

| ファイル | 内容 |
|---|---|
| `othello.js` | ルール(合法手・反転・終局判定) |
| `dqn.js` | MLP / Adam / 経験再生 / DQN エージェント / モデル入出力 |
| `train.js` | 学習ループ(自己対戦 → 学習 → 定期評価・保存) |
| `evaluate.js` | モデルをランダム・貪欲(最多ひっくり返し)と対戦させる |
| `test.js` | perft によるルール検証 + 逆伝播の数値微分チェック |
| `output/` | `model_best.json` `model_latest.json` `model_final.json` `checkpoint.json` `log.csv` |
| `train.log` | 学習ログ |

## 使い方

```
node test.js                              # ルールと勾配の確認
node train.js --minutes 60                # 学習(60分)
node train.js --resume --minutes 120      # checkpoint から再開(minutes は累積時間)
node evaluate.js output/model_best.json 500   # 対戦評価
```

主なオプション: `--hidden 256,128` `--batch 64` `--lr 0.0005` `--train-every 4` `--eps-start 0.6` `--eps-end 0.05` `--eps-decay-episodes 15000` `--eval-every 1000`

## Web UI(人間 vs DQN)

```
node server.js          # → http://localhost:8765/   (Windows は start.bat をダブルクリックでも可)
```

- 黒/白の選択、AIモデル(最終/ベスト)の切り替え、AIの勝率予想バー、合法手のヒント(Q値を勝率換算で表示)付き。
- 対局ロジックとAI推論はすべてブラウザ内(`public/game.js` `public/nn.js` と `othello.js`)で動く静的サイト。`server.js` は `dist/` を作って配信するだけ。
- `node scripts/build-site.js` で公開用の `dist/` を作成(`public/` + `othello.js` + `models/{final,best}.json`)。
- 既定は `127.0.0.1` のみで待ち受け。LAN内の別端末から使う場合は `node server.js --host 0.0.0.0`。

## GitHub Pages で公開

`.github/workflows/pages.yml` が、`main` への push で `node test.js` → `dist/` のビルド → GitHub Pages へのデプロイを行う。

1. GitHub で空のリポジトリを作る
2. `git remote add origin <URL>` → `git push -u origin main`
3. リポジトリの Settings → Pages → Build and deployment → Source を **GitHub Actions** にする
4. Actions が成功すると `https://<ユーザー名>.github.io/<リポジトリ名>/` で遊べる

公開されるモデルは `output/model_final.json` と `output/model_best.json`(リポジトリにコミットされたもの)。
再学習したらこの2つを更新してコミット/pushすれば、サイトも更新される。

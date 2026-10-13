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

- 黒/白の選択、AIモデル(特徴量あり=v2 / 特徴量なし=v1)の切り替え、**AIの読みの深さ(1〜6手)**、AIの勝率予想バー、合法手のヒント(Q値を勝率換算で表示)付き。
- **検討機能(終局後のみ。新しいゲームを始めるまで有効)**: 「一手戻る / 一手進める」(ボタン・←→キー)、最初/最後へ、棋譜の手をクリックしてその局面へジャンプ。
  戻った局面で別の手を打つとそこから分岐(同じ手なら履歴を残したまま進む)。戻る/進むでAIは勝手に動かない。
  「AIに打たせる」で今の局面から1手だけAIに打たせられ、「AI自動応手」をオフにすると両側を手で打てる。
  「検討表示」をオンにすると、各手のQ値(勝率換算)と推奨手(水色の枠)が出る。常に局面評価(黒の勝率予想)をバーで表示。
- 対局ロジックとAI推論はすべてブラウザ内(`public/game.js` `public/nn.js` と `othello.js`)で動く静的サイト。`server.js` は `dist/` を作って配信するだけ。
- `node scripts/build-site.js` で公開用の `dist/` を作成(`public/` + `othello.js` + `models/{v2,v1}.json`)。
- 既定は `127.0.0.1` のみで待ち受け。LAN内の別端末から使う場合は `node server.js --host 0.0.0.0`。

## 探索(読みの深さ)

重みはそのまま、Q値を葉の評価にした minimax(alpha-beta)で数手先を読める(`public/search.js`)。
深さ1は探索なし(最大Qの手)、深さnは n 手先まで読み、最後の1手は `max_a Q(s,a)` で評価、終局は実際の勝敗を使う。

```
node experiment_search.js --depths 1,2,3,4 --games 1000 --h2h 1000   # 結果: results/search_experiment.json
```

model_final.json での結果(各1000局、先手後手交代、序盤4手ランダム):

| 深さ | vs ランダム | vs 貪欲 | vs 深さ1 |
|---|---|---|---|
| 1 | 96.8% | 95.1% | - |
| 2 | 99.2% | 94.2% | 92.1% |
| 3 | 99.7% | 98.8% | 95.4% |
| 4 | 99.9% | 98.6% | 99.8% |

(スコア = 勝ち + 0.5×引き分け。決定的な打ち手なので各局は独立ではなく、信頼区間は実際より狭め。)
1手あたりの平均時間(Node): 深さ3 約1ms / 4 約5ms / 5 約14ms / 6 約60ms。

## 特徴量(入力)の比較: v1 と v2

| | 入力 | 中身 |
|---|---|---|
| v1 (128) | 自分の石64 + 相手の石64 | 従来。盤面そのままだけ |
| v2 (259) | v1 + 自分の着手可能マス64 + 相手の着手可能マス64 + スカラー3 | 空きマス数/60、自分の着手可能数/20、相手の着手可能数/20 |

報酬は同じく勝敗のみ。ルールから機械的に計算できる量を入力に足しただけで、評価の知識は入れていない。
8方向の対称データ拡張でも着手可能マスの面は同じ並べ替えで正しく変換される(`test.js` で検証)。
`node train.js --features v2 --out output_v2` で学習(`public/features.js` が入力の作り方。モデルJSONに `features` を保存するので古いv1モデルもそのまま動く)。

同じ条件(120分、同じ学習率スケジュール、v1は約3.3万局 / v2は約4.0万局)で学習した結果:

| 対戦(1000局、先手後手交代) | v2 のスコア |
|---|---|
| v2 vs v1(深さ1) | **67.6%** ±2.9 |
| v2 vs v1(深さ1、序盤8手ランダム) | 65.8% ±2.9 |
| v2 vs v1(深さ3) | **81.5%** ±3.4(500局) |
| v2 vs v1(深さ5) | **90.5%** ±3.3(300局) |
| v2 vs v1 ベストモデル(深さ1) | 67.4% ±2.9 |

一方、対「貪欲」(最多ひっくり返し)は深さ1で v1 95.1% に対し v2 78.4% と低かった(深さ3では v2 98.6%)。
直接対戦では v2 が明確に上なので、貪欲法という特定の相手への相性の差と考えられる(原因は未検証)。
v2 の結果: `results/features_v2_experiment.json`、学習ログ: `train_v2.log` `output_v2/log.csv`。比較は `compare_models.js`。

## GitHub Pages で公開

`.github/workflows/pages.yml` が、`main` への push で `node test.js` → `dist/` のビルド → GitHub Pages へのデプロイを行う。

1. GitHub で空のリポジトリを作る
2. `git remote add origin <URL>` → `git push -u origin main`
3. リポジトリの Settings → Pages → Build and deployment → Source を **GitHub Actions** にする
4. Actions が成功すると `https://<ユーザー名>.github.io/<リポジトリ名>/` で遊べる

公開されるモデルは特徴量あり `output_v2/model_final.json`(画面の初期選択)と、特徴量なし `output/model_final.json`(どちらもリポジトリにコミットされた最終モデル)。
再学習したらこの2つを更新してコミット/pushすれば、サイトも更新される。`model_best.json` は途中評価が最良だった時点の保存で、サイトでは使わない(最終モデルとほぼ同じ強さ)。

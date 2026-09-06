# music-analysis — 音楽制作パートナー (作業名)

自分の曲を理解し、音で返してくれる制作相棒。旧Traper Pool構想の再挑戦。

**コンセプト**: Suno的なポン出しではなく、「自分の曲・意図・出自」を条件にした
解析と生成で、詰まった制作を前に進める。成果物は文章ではなく音。

## 構成

```
定規 (信号解析)     analyze_mix.py   BPM/構成/帯域 (librosa) + コード (madmom DeepChroma+CRF)
                    describe_v2.py   コード列からの機能和声キー推定 (KS法より正確)
耳 (感覚の言語化)   ears_gemini.py   Gemini APIに音源を聴かせる
言語化パイプライン  describe_v3.py   定規+耳+truth.yaml(本人訂正)の融合プロファイル
壁打ち              consult.py       全コンテキスト注入のアレンジ相談
生成 (音のスケッチ) gen_sketch.py    ACE-Step (ローカル) audio2audio
統合CLI             sketch.py        ↑全部のフロントエンド
```

## 使い方

```bash
# ACE-Stepサーバー起動 (生成系コマンドの前提)
acestep --port 7865 --bf16 false

# 曲を登録: songs/<name>/audio.wav + truth.yaml(正解データ) を置く
uv run python sketch.py analyze  songs/summer_noise/audio.mp3   # 解析
uv run python sketch.py describe songs/summer_noise/audio.mp3   # 言語化
uv run python sketch.py consult  songs/summer_noise/audio.mp3   # 壁打ち
uv run python sketch.py vibe     songs/summer_noise/audio.mp3   # 全体変奏
uv run python sketch.py texture  songs/summer_noise/audio.mp3   # 背景テクスチャ
uv run python sketch.py part     songs/summer_noise/audio.mp3 --stem drums  # パーツ抽出
```

## データ (使うほど賢くなる仕組み)

- `songs/<name>/truth.yaml` — 本人提供の正解 (キー/BPM/コード/カポ表記/ボーカル属性)
- `songs/<name>/verdicts.md` — AI出力への採点 (これが評価ベンチマーク兼学習データ)
- `profile/artist.yaml` — 影響源プロファイル (音響的類似とは別軸の「出自」)
- `docs/01-learning-notes.md` — 音楽解析の学習ノート
- `docs/02-eval-log.md` — 精度検証の全記録 (v0→v3の言語化、コード検出対決、生成判定)

## 環境メモ

- Python 3.12 (uv)。ACE-StepとDemucsは依存衝突のため `uv tool` で隔離インストール
- ACE-Step: gradio<6 を固定。torchcodec + ffmpeg dylib を uv python の lib/ に symlink 済み
- `GEMINI_API_KEY` は `.env` (gitignore済)。モデルは `GEMINI_MODEL=gemini-3.6-flash`

## ロードマップ

- [ ] ACE-Step 1.5 重みへの更新確認 (現状1.0の可能性)
- [ ] repaint (区間差し替え) / extend (続き生成) の実装
- [ ] 2曲目以降の投入で言語化精度の汎化を検証
- [ ] 品質が納得いったら Web アプリ化 (Vercel+Supabase+Macワーカー案) → メンバー公開

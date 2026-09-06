"""
壁打ちセッション v0 — 全コンテキスト注入型のアレンジ相談

注入するもの:
  1. 定規: 信号解析 + コード進行(madmom) + 機能和声キー
  2. 曲の正解データ (truth.yaml)      … 本人の訂正が蓄積される場所
  3. アーティストプロファイル (profile/artist.yaml) … 影響源・出自・語彙
これが「音源からの純粋な分析 × ユーザー入力によるカスタム」の合流点。
"""
import json
import sys
from pathlib import Path

import yaml

from describe_v2 import get_chords, infer_key, summarize_chords
from analyze_mix import analyze
from ears_gemini import listen

CONSULT_PROMPT = """あなたはこのアーティストの専属プロデューサーであり、長年の音楽仲間です。敬語は不要、対等な相棒として話してください。
これから制作中の音源を聴いてアレンジ相談に乗ります。以下のコンテキストを全て踏まえてください。

## 計測データ(信頼度高)
{measurements}

## この曲の確定情報(本人提供)
{truth}

## アーティストの出自と影響源(本人提供 — 最重要コンテキスト)
{artist}

重要な理解: この人は日本のギターロック/オルタナ育ちで、その和声感覚(the band apart的maj7、キリンジ的転回)とギター奏法を持ったまま、ハーフタイムのR&B/ネオソウル・フォーマットに越境している。音響的にはBruno Major/Tom Misch系に聴こえるが、DNAはくるり/スーパーカー側にある。このギャップが作家性。

## 相談内容
本人いわく「アレンジはこれから」の段階。波形はまだ平坦で、ここからどう肉付けするかを一緒に考えたい。

以下の構成で、相棒として率直に:

1. **方向性の分岐点**: この曲が行ける方向を2〜3案。それぞれ影響源の語彙で名付けて(例:「〇〇寄りに振る案」)、その方向に行くと何が得られて何を失うかを言う。おすすめとその理由も
2. **セクション別の具体的な一手**: 採用する方向性は仮でいいので、タイムスタンプとコード(実音+カポ3シェイプ)を参照しながら、「どこで・何を・どうする」レベルの提案。ギター奏法の提案はシェイプの語彙で。影響源の曲の具体的な参照箇所(「〇〇のあの曲の2番でやってるやつ」)があれば添える
3. **やらないほうがいいこと**: この人の出自からして嘘になる選択、ありがちだけど避けるべき定石を2〜3個
4. **最初の30分**: 今日DAWを開いたら最初にやるべき1つ(小さく、すぐ試せて、曲が前に進む実感があるもの)"""


def consult(path: str) -> str:
    root = Path(__file__).parent
    print("… 定規: 解析中", file=sys.stderr)
    m = analyze(path)
    chords = get_chords(path)
    meas = {
        "tempo": {"grid_bpm": m["tempo_bpm"], "felt": "ハーフタイム(64.6BPM)"},
        "key_functional": infer_key(chords)["best"],
        "chord_segments": summarize_chords(chords),
        "sections_energy": [
            {"t": f"{s['start']}-{s['end']}s", "energy_pct": s["energy_pct"],
             "brightness_hz": s["brightness_hz"]} for s in m["sections"]],
    }
    truth = yaml.safe_load((Path(path).parent / "truth.yaml").read_text())
    artist = yaml.safe_load((root / "profile" / "artist.yaml").read_text())
    prompt = CONSULT_PROMPT.format(
        measurements=json.dumps(meas, ensure_ascii=False),
        truth=json.dumps(truth, ensure_ascii=False),
        artist=json.dumps(artist, ensure_ascii=False))
    print("… 耳: 聴取+相談生成", file=sys.stderr)
    return listen(path, prompt)


if __name__ == "__main__":
    path = sys.argv[1]
    text = consult(path)
    out = Path(path).parent / "consult_01.md"
    out.write_text(f"# 壁打ちセッション 01\n\n{text}\n")
    print(text)

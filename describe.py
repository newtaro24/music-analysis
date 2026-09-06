"""
言語化パイプライン v0 — 「定規」の数値を渡した上でGeminiに聴かせ、
構造化された記述スキーマで曲を言語化する。

北極星: 曲を聴いて、どれだけ解像度の高い自然言語表現ができるか。
数値(定規)と聴感(耳)を融合させることで、感覚的だが検証可能な記述を目指す。
"""
import json
import sys
from pathlib import Path

from analyze_mix import analyze
from ears_gemini import listen

SCHEMA_PROMPT = """あなたは経験豊富な音楽プロデューサーです。この音源を聴いて、以下のスキーマで曲を「言語化」してください。
参考として、信号解析による計測値を渡します。あなたの聴感と計測値が食い違う場合は、その旨を明記した上で聴感を優先してください。

## 計測値(信号解析)
{measurements}

## 記述スキーマ(日本語で、音楽制作者に通じる具体的な語彙で)

1. **一言サマリ**: この曲を知らない人に30字で伝えるなら
2. **スタイル座標**: ジャンル系譜 / 年代感 / シーン(どこで鳴っていそうな音楽か)
3. **サウンドパレット**: 聴き取れるパートを列挙し、それぞれの音色を具体語で(例:「丸くサチュレートしたシンセベース、サブ寄り」)
4. **リズムとグルーヴ**: BPM感、ビートの性格(ジャスト/ヨレ/シャッフル)、手数、どのパートがグルーヴを支配しているか
5. **ハーモニーの色**: キー/モードの感触(計測値のキー候補も検証して)、進行の性格(循環/展開型)、テンションの度合い
6. **空間とミックス**: 奥行き・左右の使い方、質感(ローファイ/ハイファイ、ドライ/ウェット)、帯域の重心
7. **動きの源泉**: この曲の推進力はどこに住んでいるか(音量変化/音色変化/リズム変化/反復の中毒性)。計測値のセクション別データと突き合わせて
8. **類似マップ**: 似たアーティスト・曲を3〜5挙げ、それぞれ「何が似ているか」を要素レベルで特定(音色が/グルーヴが/構成が)
9. **この曲だけの指紋**: 他の類似曲と区別する、この曲固有の特徴を1〜2点"""


def describe(path: str) -> str:
    print("… 定規レイヤー(信号解析)を実行中", file=sys.stderr)
    m = analyze(path)
    # プロンプト用に計測値をコンパクトに整形
    meas = {
        "duration_sec": m["duration_sec"],
        "tempo_bpm": m["tempo_bpm"],
        "key_candidates": m["key"],
        "sections": [
            {"time": f"{s['start']}-{s['end']}s", "bars": s["bars_approx"],
             "energy_pct": s["energy_pct"], "brightness_hz": s["brightness_hz"],
             "onset_density": s["onset_density"]}
            for s in m["sections"]
        ],
        "note": "キー候補はテンプレート相関であり、トニックとモードの区別(例: Cドリアン=B♭メジャーと同一音セット)は苦手",
    }
    prompt = SCHEMA_PROMPT.format(measurements=json.dumps(meas, ensure_ascii=False, indent=1))
    print("… 耳レイヤー(Gemini聴取)を実行中", file=sys.stderr)
    return listen(path, prompt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run python describe.py <audiofile>")
    text = describe(sys.argv[1])
    out = Path(sys.argv[1]).stem + ".profile.md"
    Path(out).write_text(f"# 楽曲言語化プロファイル\n\n{text}\n")
    print(text)
    print(f"\n→ 保存: {out}")

"""
言語化パイプライン v2 — 定規のフル出力(コード進行+機能和声キー推定)を耳に渡す

v0からの強化:
  - コード進行: madmom (DeepChroma+CRF、summer_noiseで実測100%)
  - キー推定: KSテンプレート(音の分布)を捨て、コード列からの機能和声推定に変更
    「検出コードが最も多くダイアトニックに収まるキー」+ V→I カデンツの存在で確定
  - プロンプト: 機能和声(ディグリー)での説明、項目ごとの確信度、断定回避を要求
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from analyze_mix import analyze
from ears_gemini import listen

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def get_chords(path: str) -> list:
    """madmomでコード検出(.labがあれば再利用)"""
    lab = Path(path).parent / "madmom.lab"
    if not lab.exists():
        wav = str(Path(path).with_suffix(".44k.wav"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "44100", wav], check=True)
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        chords = DeepChromaChordRecognitionProcessor()(DeepChromaProcessor()(wav))
        with open(lab, "w") as f:
            for s, e, c in chords:
                f.write(f"{s:.2f}\t{e:.2f}\t{c}\n")
    out = []
    for line in open(lab):
        s, e, c = line.split()
        if c != "N":
            out.append({"start": float(s), "end": float(e), "chord": c})
    return out


def chord_pcs(label: str):
    """'A#:maj' → (ルートpc, 構成音pcの集合)"""
    root_s, qual = label.split(":")
    r = NOTE.index(root_s)
    iv = [0, 4, 7] if qual == "maj" else [0, 3, 7]
    return r, {(r + i) % 12 for i in iv}


def infer_key(chords: list) -> dict:
    """機能和声的キー推定: ダイアトニック度(時間加重) + V→Iカデンツ数"""
    scores = []
    for tonic in range(12):
        for mode, degs in (("major", [0, 2, 4, 5, 7, 9, 11]), ("minor", [0, 2, 3, 5, 7, 8, 10])):
            scale = {(tonic + d) % 12 for d in degs}
            dia = sum(c["end"] - c["start"] for c in chords if chord_pcs(c["chord"])[1] <= scale)
            total = sum(c["end"] - c["start"] for c in chords)
            # V→I カデンツ検出(ルートが完全5度下行してトニックへ)
            cad = 0
            for a, b in zip(chords, chords[1:]):
                ra = chord_pcs(a["chord"])[0]
                rb = chord_pcs(b["chord"])[0]
                if ra == (tonic + 7) % 12 and rb == tonic:
                    cad += 1
            scores.append({"key": f"{NOTE[tonic]} {mode}", "diatonic_pct": round(100 * dia / total),
                           "cadences_V_to_I": cad, "score": dia / total + 0.05 * cad})
    scores.sort(key=lambda x: -x["score"])
    return {"best": scores[0], "runner_up": scores[1]}


def summarize_chords(chords: list) -> list:
    return [{"t": f"{c['start']:.0f}-{c['end']:.0f}s", "chord": c["chord"].replace(":maj", "").replace(":min", "m")}
            for c in chords]


PROMPT = """あなたは経験豊富な音楽プロデューサー兼音楽理論家です。この制作中の音源を聴き、信号解析の計測値と突き合わせて、曲を高解像度で「言語化」してください。

## 計測値(信号解析 — コード進行は高精度モデルによる検出で信頼度高)
{measurements}

## 指示
- コード進行は計測値を信頼し、キーに対する**ディグリー(ローマ数字)による機能和声の説明**を必ず入れること
- 聴感でしか分からないこと(音色の質感、7thやテンションの色、グルーヴのニュアンス、歌唱表現)に耳のリソースを集中すること
- 各項目に確信度(高/中/低)を付け、低いものは断定しないこと(例: ボーカルの属性、楽器の同定)
- 音楽制作者に通じる具体的な語彙で書くこと

## 記述スキーマ
1. **一言サマリ** (30字)
2. **スタイル座標**: ジャンル系譜 / 年代感 / シーン
3. **和声の設計**: 検出された進行をディグリーで解釈し、この進行が生む感情的な効果を説明。三和音表記の裏で実際に鳴っている7th/テンションの色も聴き取る
4. **サウンドパレット**: パートごとの音色を具体語で + 確信度
5. **リズムとグルーヴ**: BPM解釈(倍/半テンポの吟味含む)、ビートの性格、グルーヴの支配者
6. **構成のドラマ**: セクションごとの役割と、エネルギー計測値との対応
7. **動きの源泉**: この曲の推進力はどこに住んでいるか
8. **類似マップ**: 3〜5組。それぞれ「何の要素が似ているか」を特定
9. **この曲だけの指紋**: 固有の特徴1〜2点
10. **プロデューサーの視点**: この曲が次の段階に進むために最も効く一手(具体的に)"""


def describe(path: str) -> str:
    print("… 定規: 信号解析", file=sys.stderr)
    m = analyze(path)
    print("… 定規: コード検出(madmom)", file=sys.stderr)
    chords = get_chords(path)
    key = infer_key(chords)
    meas = {
        "duration_sec": m["duration_sec"],
        "tempo_bpm_grid": m["tempo_bpm"],
        "key_functional": key,
        "chord_segments": summarize_chords(chords),
        "sections_energy": [
            {"t": f"{s['start']}-{s['end']}s", "energy_pct": s["energy_pct"],
             "brightness_hz": s["brightness_hz"], "onset_density": s["onset_density"]}
            for s in m["sections"]],
    }
    print("… 耳: Gemini聴取", file=sys.stderr)
    return listen(path, PROMPT.format(measurements=json.dumps(meas, ensure_ascii=False)))


if __name__ == "__main__":
    path = sys.argv[1]
    text = describe(path)
    out = Path(path).parent / "profile_v2.md"
    Path(out).write_text(f"# 楽曲言語化プロファイル v2\n\n{text}\n")
    print(text)
    print(f"\n→ 保存: {out}", file=sys.stderr)

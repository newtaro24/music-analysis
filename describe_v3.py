"""
言語化パイプライン v3 — パーソナライズ層の追加

v2 (定規フル出力 + 耳) に、本人由来の確定情報(truth.yaml)を注入する。
これが「使うほど本人を理解する」機構の最小実装:
  - 過去の訂正(ボーカル=男性 等)は確定事実としてAIの聴き間違いを上書き
  - 本人の語彙(カポ+シェイプ)で話す
検出と本人情報が矛盾したら、本人情報を優先しつつ矛盾を明示する。
"""
import json
import sys
from pathlib import Path

import yaml

from describe_v2 import get_chords, infer_key, summarize_chords, PROMPT
from analyze_mix import analyze
from ears_gemini import listen

TRUTH_BLOCK = """

## 本人提供の確定情報(最優先。聴感がこれと矛盾する場合は確定情報を採用し、矛盾があったことを明記)
{truth}

## パーソナライズ指示
- ギターに言及するときは、実音に加えて本人の語彙(カポ{capo}のシェイプ)でも表記すること
- 確定情報にある事実(ボーカルの属性等)を聴き直し、その前提で音色や表現の質感を記述すること"""


def describe(path: str, truth_path: str | None = None) -> str:
    print("… 定規: 信号解析 + コード検出", file=sys.stderr)
    m = analyze(path)
    chords = get_chords(path)
    meas = {
        "duration_sec": m["duration_sec"],
        "tempo_bpm_grid": m["tempo_bpm"],
        "key_functional": infer_key(chords),
        "chord_segments": summarize_chords(chords),
        "sections_energy": [
            {"t": f"{s['start']}-{s['end']}s", "energy_pct": s["energy_pct"],
             "brightness_hz": s["brightness_hz"], "onset_density": s["onset_density"]}
            for s in m["sections"]],
    }
    prompt = PROMPT.format(measurements=json.dumps(meas, ensure_ascii=False))
    if truth_path and Path(truth_path).exists():
        truth = yaml.safe_load(Path(truth_path).read_text())
        capo = truth.get("guitar", {}).get("capo", "?")
        prompt += TRUTH_BLOCK.format(truth=json.dumps(truth, ensure_ascii=False), capo=capo)
        print("… パーソナライズ層: truth.yaml注入", file=sys.stderr)
    print("… 耳: Gemini聴取", file=sys.stderr)
    return listen(path, prompt)


if __name__ == "__main__":
    path = sys.argv[1]
    truth = str(Path(path).parent / "truth.yaml")
    text = describe(path, truth)
    out = Path(path).parent / "profile_v3.md"
    Path(out).write_text(f"# 楽曲言語化プロファイル v3 (パーソナライズ済)\n\n{text}\n")
    print(text)

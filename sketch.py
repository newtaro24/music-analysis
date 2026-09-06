"""
sketch — 音楽制作パートナーの統合CLI

使い方:
  uv run python sketch.py analyze  <audio>              # 定規: BPM/キー/コード/構成
  uv run python sketch.py describe <audio>              # 言語化プロファイル(truth.yaml自動注入)
  uv run python sketch.py consult  <audio>              # 壁打ち(全コンテキスト)
  uv run python sketch.py vibe     <audio> [-s 0.5]     # 全体変奏スケッチ
  uv run python sketch.py part     <audio> [--stem drums] [-s 0.5]  # 生成→分離でワンパーツ
  uv run python sketch.py texture  <audio> [-s 0.35] [-p "..."]     # 背景テクスチャ
  uv run python sketch.py similar  <audio>              # 多軸類似マップ

前提: ACE-Stepサーバー起動済み (acestep --port 7865 --bf16 false)
      GEMINI_API_KEY / GEMINI_MODEL は .env or 環境変数
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

def _meas(audio):
    from analyze_mix import analyze
    from describe_v2 import get_chords, infer_key, summarize_chords
    m = analyze(audio)
    chords = get_chords(audio)
    return m, chords, infer_key(chords), summarize_chords(chords)

def cmd_analyze(a):
    m, _, key, chords = _meas(a.audio)
    print(f"BPM(grid): {m['tempo_bpm']}  キー(機能和声): {key['best']['key']} "
          f"(ダイアトニック{key['best']['diatonic_pct']}%, V-I×{key['best']['cadences_V_to_I']})")
    print("コード:", " ".join(f"{c['chord']}" for c in chords[:24]), "…")
    out = Path(a.audio).with_suffix(".analysis.json")
    out.write_text(json.dumps({"measure": m, "key": key, "chords": chords}, ensure_ascii=False, indent=1))
    print(f"→ {out}")

def cmd_describe(a):
    from describe_v3 import describe
    text = describe(a.audio, str(Path(a.audio).parent / "truth.yaml"))
    out = Path(a.audio).parent / "profile.md"
    out.write_text(text); print(text); print(f"\n→ {out}", file=sys.stderr)

def cmd_consult(a):
    from consult import consult
    text = consult(a.audio)
    out = Path(a.audio).parent / "consult.md"
    out.write_text(text); print(text); print(f"\n→ {out}", file=sys.stderr)

def cmd_similar(a):
    from ears_gemini import listen
    prompt = Path(__file__).parent.joinpath("prompts", "similar_axes.txt")
    text = listen(a.audio, prompt.read_text())
    out = Path(a.audio).parent / "similar_map.md"
    out.write_text(text); print(text)

def _generate(audio, prompt, strength, out_name, duration=30):
    from gradio_client import Client, handle_file
    from gen_sketch import DEFAULTS
    c = Client("http://127.0.0.1:7865", verbose=False)
    print(f"生成中… strength={strength}", file=sys.stderr)
    r = c.predict(format="wav", audio_duration=duration, prompt=prompt, lyrics="[instrumental]",
                  audio2audio_enable=True, ref_audio_strength=strength,
                  ref_audio_input=handle_file(audio), **DEFAULTS, api_name="/__call__")
    src = r[0] if isinstance(r, (list, tuple)) else r
    out = Path(audio).parent / "sketches"; out.mkdir(exist_ok=True)
    dst = out / out_name; shutil.copy(src, dst); print(f"→ {dst}")
    return dst

def _base_prompt(audio, extra=""):
    """truth.yaml/解析からプロンプトを構築(v0: gen_sketchの固定文+追記)"""
    from gen_sketch import PROMPT
    return f"{PROMPT}, {extra}" if extra else PROMPT

def cmd_vibe(a):
    _generate(a.audio, _base_prompt(a.audio, a.prompt or ""), a.strength, f"vibe_s{int(a.strength*100)}.wav")

def cmd_texture(a):
    p = a.prompt or "ambient background texture, atmospheric pad, evolving, no drums"
    _generate(a.audio, _base_prompt(a.audio, p), a.strength, f"texture_s{int(a.strength*100)}.wav")

def cmd_part(a):
    dst = _generate(a.audio, _base_prompt(a.audio, a.prompt or ""), a.strength, f"part_src_s{int(a.strength*100)}.wav")
    stems_dir = Path(a.audio).parent / "stems"
    print(f"分離中… (--two-stems={a.stem})", file=sys.stderr)
    subprocess.run(["demucs", f"--two-stems={a.stem}", "-o", str(stems_dir), str(dst)], check=True,
                   capture_output=True)
    stem = stems_dir / "htdemucs" / dst.stem / f"{a.stem}.wav"
    print(f"→ {stem}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="sketch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("analyze", cmd_analyze), ("describe", cmd_describe), ("consult", cmd_consult),
                     ("similar", cmd_similar), ("vibe", cmd_vibe), ("texture", cmd_texture), ("part", cmd_part)]:
        p = sub.add_parser(name); p.add_argument("audio"); p.set_defaults(fn=fn)
        if name in ("vibe", "texture", "part"):
            p.add_argument("-s", "--strength", type=float, default=0.35 if name == "texture" else 0.5)
            p.add_argument("-p", "--prompt", default=None)
        if name == "part":
            p.add_argument("--stem", default="drums", choices=["drums", "bass", "vocals", "other"])
    a = ap.parse_args()
    a.fn(a)

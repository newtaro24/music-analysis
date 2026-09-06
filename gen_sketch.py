"""
音のスケッチ生成 v0 — ACE-Step (ローカル) を解析コンテキストで条件付けて呼ぶ

「解析が生成の制御装置になる」の最小実装:
  プロンプトは人が書かず、truth.yaml(曲の確定情報)とプロファイルから組み立てる。
  audio2audio: 元曲を参照音声として、指定強度で「変奏スケッチ」を生成する。
"""
import shutil
import sys
from pathlib import Path

from gradio_client import Client, handle_file

# ACE-Step推奨デフォルト
DEFAULTS = dict(infer_step=60, guidance_scale=15, scheduler_type="euler", cfg_type="apg",
                omega_scale=10, manual_seeds=None, guidance_interval=0.5,
                guidance_interval_decay=0, min_guidance_scale=3, use_erg_tag=True,
                use_erg_lyric=True, use_erg_diffusion=True, oss_steps=None,
                guidance_scale_text=0, guidance_scale_lyric=0)

# truth.yaml + 解析 + プロファイル由来 (v0は手組み、将来は自動構築)
PROMPT = ("neo soul, lofi r&b, half-time groove, 65 bpm, Bb major, maj7 chords, "
          "clean electric guitar fingerpicking, warm rhodes piano, deep round sub bass, "
          "laidback fat drums, rimshot, mellow, nostalgic summer night, japanese city pop influence, "
          "lo-fi warm mix, intimate")

def generate(ref_audio: str, strength: float = 0.5, duration: float = 30, out_dir: str = "sketches"):
    c = Client("http://127.0.0.1:7865", verbose=False)
    print(f"生成中… (audio2audio, strength={strength}, {duration}s)", file=sys.stderr)
    result = c.predict(
        format="wav", audio_duration=duration, prompt=PROMPT, lyrics="[instrumental]",
        audio2audio_enable=True, ref_audio_strength=strength,
        ref_audio_input=handle_file(ref_audio), **DEFAULTS, api_name="/__call__")
    audio_path = result[0] if isinstance(result, (list, tuple)) else result
    out = Path(ref_audio).parent / out_dir
    out.mkdir(exist_ok=True)
    dst = out / f"a2a_s{int(strength*100)}.wav"
    shutil.copy(audio_path, dst)
    print(f"→ {dst}")
    return dst

if __name__ == "__main__":
    generate("songs/summer_noise/audio.mp3",
             strength=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)

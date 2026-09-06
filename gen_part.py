"""ワンパート生成テスト: パート指定プロンプト + 弱めのa2aアンカー"""
import shutil, sys
from pathlib import Path
from gradio_client import Client, handle_file
from gen_sketch import DEFAULTS

DRUM_PROMPT = ("solo drums only, drum loop, no bass, no melody, no chords, no vocals, "
               "half-time groove, 65 bpm, lofi neo soul drums, fat soft kick, rimshot, "
               "laidback behind the beat, warm analog, minimal")

def gen(ref, strength, out_name):
    c = Client("http://127.0.0.1:7865", verbose=False)
    r = c.predict(format="wav", audio_duration=30, prompt=DRUM_PROMPT, lyrics="[instrumental]",
                  audio2audio_enable=True, ref_audio_strength=strength,
                  ref_audio_input=handle_file(ref), **DEFAULTS, api_name="/__call__")
    audio = r[0] if isinstance(r, (list, tuple)) else r
    dst = Path(ref).parent / "sketches" / out_name
    shutil.copy(audio, dst); print(f"→ {dst}")

if __name__ == "__main__":
    gen("songs/summer_noise/audio.mp3", 0.35, "drums_gen_s35.wav")

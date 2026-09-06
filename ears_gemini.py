"""
耳レイヤー v0 — Gemini APIに音源をまるごと聴かせて、感覚的な記述を得る

librosa(定規)が「数値の事実」を出すのに対して、こちらは
「人が聴いたときにどう感じるか」の記述を担当する。
両方の出力を突き合わせるのがこのプロジェクトの解析コア。

使い方:
  GEMINI_API_KEY=xxx uv run python ears_gemini.py <audiofile>
  (.env に GEMINI_API_KEY=xxx と書いてもOK)
"""
import base64
import json
import os
import sys
from pathlib import Path

import requests

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

LISTENING_PROMPT = """あなたは経験豊富な音楽プロデューサー兼アレンジャーです。
この音源は制作中の楽曲です。じっくり聴いて、以下を日本語で報告してください。

1. **第一印象**: ジャンル/スタイル、ムード、質感。似た傾向のアーティストや曲があれば具体名を挙げる(確信がなければ「〜の系譜」程度でよい)
2. **構成の聴き取り**: タイムスタンプ付きで、セクションごとに「何が鳴っているか」「前のセクションから何が変わったか」を記述
3. **この曲の推進力**: 音量変化に頼らないタイプの曲かどうかも含め、何がリスナーを引っ張っているか(リズム、音色変化、テクスチャ、反復のマジック等)
4. **アレンジ上の課題**: 制作中の曲として、今どこが弱いか。「どこの・何を・どうする」のレベルで具体的に
5. **ミックスの質感**: 帯域バランスや空間処理の印象(簡潔に)

推測で断定せず、聴こえたことを根拠に述べてください。"""


def load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env = Path(__file__).parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("GEMINI_API_KEY が見つからない (.env か環境変数で指定)")
    return key


MIME = {".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/aac",
        ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg", ".aiff": "audio/aiff"}


def listen(path: str, prompt: str = LISTENING_PROMPT) -> str:
    p = Path(path)
    mime = MIME.get(p.suffix.lower())
    if mime is None:
        sys.exit(f"未対応の形式: {p.suffix} (mp3/wav/m4a/flac/ogg/aiff)")
    size_mb = p.stat().st_size / 1e6
    if size_mb > 19:
        sys.exit(f"{size_mb:.1f}MB はインライン上限(20MB)超え。mp3に圧縮するかFiles API対応が必要")

    audio_b64 = base64.b64encode(p.read_bytes()).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": audio_b64}},
            ]
        }]
    }
    r = requests.post(url, params={"key": load_key()}, json=body, timeout=300)
    if r.status_code != 200:
        sys.exit(f"API error {r.status_code}: {r.text[:500]}")
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run python ears_gemini.py <audiofile>")
    text = listen(sys.argv[1])
    out = Path(sys.argv[1]).stem + ".ears.md"
    Path(out).write_text(f"# 耳レイヤー聴取結果 ({MODEL})\n\n{text}\n")
    print(text)
    print(f"\n→ 保存: {out}")

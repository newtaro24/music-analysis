"""
ローカルWebアプリ v0 — 音源を入れてフィードバックを得るまでの一周を回す

構成: FastAPI 1ファイル + static/index.html 1ファイル。
重い処理(解析/生成)はスレッドのジョブキューで非同期実行し、フロントがポーリング。
採点はそのまま songs/<name>/verdicts.md に追記される(=試行ログ)。
"""
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("GEMINI_MODEL", "gemini-3.6-flash")

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
SONGS = ROOT / "songs"
SONGS.mkdir(exist_ok=True)

app = FastAPI(title="music-partner")
_exec = ThreadPoolExecutor(max_workers=1)  # MPS生成は直列
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _safe(name: str) -> str:
    if not re.fullmatch(r"[\w\-ぁ-んァ-ヶ一-龠ー]{1,64}", name):
        raise HTTPException(400, "曲名は英数字/日本語/ハイフンで")
    return name


def _song_dir(name: str) -> Path:
    d = SONGS / _safe(name)
    if not d.exists():
        raise HTTPException(404, f"曲が見つからない: {name}")
    return d


def _audio_of(d: Path) -> Path:
    for ext in (".wav", ".mp3", ".m4a", ".flac"):
        p = d / f"audio{ext}"
        if p.exists():
            return p
    raise HTTPException(404, "audioファイルがない")


def _submit(fn, *args) -> str:
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running"}

    def run():
        try:
            result = fn(*args)
            _jobs[job_id] = {"status": "done", "result": result}
        except Exception as e:
            _jobs[job_id] = {"status": "error", "error": f"{type(e).__name__}: {e}"}

    _exec.submit(run)
    return job_id


# ---- 各アクションの実体 (既存モジュールを呼ぶだけ) ----

def do_analyze(audio: Path) -> dict:
    from analyze_mix import analyze
    from describe_v2 import get_chords, infer_key, summarize_chords
    m = analyze(str(audio))
    chords = get_chords(str(audio))
    key = infer_key(chords)
    return {"type": "analysis", "tempo_bpm": m["tempo_bpm"], "key": key["best"],
            "sections": m["sections"], "chords": summarize_chords(chords)[:60]}


def do_describe(audio: Path) -> dict:
    from describe_v3 import describe
    text = describe(str(audio), str(audio.parent / "truth.yaml"))
    (audio.parent / "profile.md").write_text(text)
    return {"type": "markdown", "name": "言語化プロファイル", "text": text}


def do_consult(audio: Path) -> dict:
    from consult import consult
    text = consult(str(audio))
    (audio.parent / "consult.md").write_text(text)
    return {"type": "markdown", "name": "壁打ち", "text": text}


def do_similar(audio: Path) -> dict:
    from ears_gemini import listen
    text = listen(str(audio), (ROOT / "prompts" / "similar_axes.txt").read_text())
    (audio.parent / "similar_map.md").write_text(text)
    return {"type": "markdown", "name": "類似マップ", "text": text}


def do_generate(audio: Path, mode: str, strength: float, prompt: str, stem: str) -> dict:
    import sketch as sk
    extra = prompt or ("ambient background texture, atmospheric pad, evolving, no drums"
                       if mode == "texture" else "")
    dst = sk._generate(str(audio), sk._base_prompt(str(audio), extra), strength,
                       f"{mode}_s{int(strength*100)}_{uuid.uuid4().hex[:4]}.wav")
    if mode == "part":
        import subprocess
        stems_dir = audio.parent / "stems"
        subprocess.run(["demucs", f"--two-stems={stem}", "-o", str(stems_dir), str(dst)],
                       check=True, capture_output=True)
        dst = stems_dir / "htdemucs" / Path(dst).stem / f"{stem}.wav"
    rel = dst.relative_to(SONGS)
    return {"type": "audio", "name": f"{mode} (strength {strength})", "url": f"/media/{rel}"}


ACTIONS = {"analyze": do_analyze, "describe": do_describe, "consult": do_consult, "similar": do_similar}


# ---- API ----

@app.get("/api/songs")
def list_songs():
    out = []
    for d in sorted(SONGS.iterdir()):
        if d.is_dir():
            try:
                _audio_of(d)
                out.append({"name": d.name, "has_truth": (d / "truth.yaml").exists()})
            except HTTPException:
                pass
    return out


@app.post("/api/songs")
async def upload_song(file: UploadFile, name: str = Form(...)):
    d = SONGS / _safe(name)
    d.mkdir(exist_ok=True)
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    if ext not in (".wav", ".mp3", ".m4a", ".flac"):
        raise HTTPException(400, f"未対応形式: {ext}")
    (d / f"audio{ext}").write_bytes(await file.read())
    t = d / "truth.yaml"
    if not t.exists():
        t.write_text(f"# 正解データ(分かる範囲で埋めると精度が上がる)\ntitle: {name}\n"
                     "vocal:        # male / female / none\nkey:          # 例 Bb major\n"
                     "guitar:\n  capo:       # 例 3\nprogression:  # 例 [BbM7, Gm7, EbM7, 'Cm7 F']\n")
    return {"ok": True, "name": name}


@app.post("/api/songs/{name}/run/{action}")
def run_action(name: str, action: str, strength: float = 0.5, prompt: str = "", stem: str = "drums"):
    audio = _audio_of(_song_dir(name))
    if action in ACTIONS:
        return {"job": _submit(ACTIONS[action], audio)}
    if action in ("vibe", "texture", "part"):
        s = strength if action != "texture" or strength != 0.5 else 0.35
        return {"job": _submit(do_generate, audio, action, s, prompt, stem)}
    raise HTTPException(400, f"不明なアクション: {action}")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _jobs.get(job_id, {"status": "unknown"})


@app.post("/api/songs/{name}/verdict")
def add_verdict(name: str, target: str = Form(...), rating: str = Form(...), comment: str = Form("")):
    d = _song_dir(name)
    from datetime import date
    line = f"- {date.today()} **{target}** 判定:{rating}" + (f" — {comment}" if comment else "") + "\n"
    v = d / "verdicts.md"
    v.write_text((v.read_text() if v.exists() else f"# AI出力への採点 ({name})\n\n") + line)
    return {"ok": True}


@app.get("/media/{path:path}")
def media(path: str):
    p = (SONGS / path).resolve()
    if not p.is_file() or SONGS.resolve() not in p.parents:
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

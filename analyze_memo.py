"""
ボイスメモ解析 v0 — 単旋律（鼻歌・口笛・単音楽器）の音声から「曲の設計図」を抽出する

パイプライン:
  1. 読み込み & 前処理   … m4a等はffmpegでwav化、無音トリム
  2. ピッチ抽出 (pYIN)   … 各時刻の基本周波数f0を推定（確率的YINアルゴリズム）
  3. ノート化            … 連続f0を「音符」に区切る（音楽解析の核心部分）
  4. キー推定            … 音の出現分布をKrumhansl-Schmucklerプロファイルと照合
  5. テンポ推定          … オンセット（音の立ち上がり）の周期性から推定
  6. 設計図JSON出力
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np

SR = 22050  # 解析用サンプルレート（ピッチ解析には十分、計算が軽い）

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler キープロファイル:
# 「あるキーの曲で、12音がどれくらいの重みで使われるか」を心理実験で測った値。
# 曲中の音の分布とこのプロファイルの相関が最も高いキーを採用する。
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_audio(path: str) -> tuple[np.ndarray, float]:
    """音声を読み込む。soundfileが読めない形式(m4a等)はffmpegで変換。"""
    p = Path(path)
    if p.suffix.lower() in {".m4a", ".aac", ".mp4", ".caf"}:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(p), "-ac", "1", "-ar", str(SR), tmp],
            check=True, capture_output=True,
        )
        y, sr = librosa.load(tmp, sr=SR, mono=True)
        Path(tmp).unlink()
    else:
        y, sr = librosa.load(str(p), sr=SR, mono=True)
    # 前後の無音をカット（-40dB以下を無音とみなす）
    y, _ = librosa.effects.trim(y, top_db=40)
    return y, sr


def extract_f0(y: np.ndarray, sr: float):
    """pYINで基本周波数(f0)の時系列を抽出。voiced_flagは「声が鳴っている区間」の判定。"""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # 65Hz  男性の低い声まで
        fmax=librosa.note_to_hz("C6"),   # 1046Hz 口笛の高い音まで
        sr=sr,
        frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr)
    return f0, voiced_flag, times


def segment_notes(f0, voiced_flag, times, min_dur=0.08, split_cents=80):
    """
    連続的なピッチ曲線を離散的な「ノート（音符）」に区切る。
    - 無声区間で切る
    - ピッチが80セント(半音の8割)以上ジャンプしたら別のノートとみなす
    - min_dur秒未満の断片は捨てる（ノイズ・しゃくり対策）
    """
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), 0, f0))
    notes = []
    start_i = None

    def flush(s, e):
        dur = times[e - 1] - times[s]
        if dur < min_dur:
            return
        seg = midi[s:e]
        pitch = float(np.median(seg))  # ノートの音高はf0の中央値（ビブラートに頑健）
        notes.append({
            "start": round(float(times[s]), 3),
            "dur": round(float(dur), 3),
            "midi": round(pitch, 2),
            "note": librosa.midi_to_note(int(round(pitch))),
        })

    for i in range(len(f0)):
        if voiced_flag[i] and not np.isnan(f0[i]):
            if start_i is None:
                start_i = i
            elif abs(midi[i] - np.median(midi[start_i:i])) > split_cents / 100:
                flush(start_i, i)
                start_i = i
        else:
            if start_i is not None:
                flush(start_i, i)
                start_i = None
    if start_i is not None:
        flush(start_i, len(f0))
    return notes


def estimate_key(notes):
    """ノート列からキーを推定。各音の長さで重み付けしたピッチクラス分布を作り、
    24キー(12メジャー+12マイナー)のプロファイルと相関を取る。"""
    if not notes:
        return None
    pc_hist = np.zeros(12)
    for n in notes:
        pc_hist[int(round(n["midi"])) % 12] += n["dur"]
    if pc_hist.sum() == 0:
        return None
    scores = []
    for shift in range(12):
        rolled = np.roll(pc_hist, -shift)
        scores.append((np.corrcoef(rolled, KS_MAJOR)[0, 1], f"{NOTE_NAMES[shift]} major"))
        scores.append((np.corrcoef(rolled, KS_MINOR)[0, 1], f"{NOTE_NAMES[shift]} minor"))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {
        "key": best[1],
        "confidence": round(float(best[0]), 3),
        "runner_up": second[1],
        "runner_up_confidence": round(float(second[0]), 3),
    }


def estimate_tempo(y, sr):
    """オンセット強度の自己相関からテンポ推定。
    鼻歌はアタックが曖昧なので、これは参考値程度（confidenceも返す）。"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    # ビート間隔のばらつきが小さいほど信頼できる
    conf = None
    if len(beats) > 3:
        intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
        conf = round(float(1.0 - min(1.0, np.std(intervals) / np.mean(intervals))), 3)
    return {"bpm": round(tempo, 1), "confidence": conf, "n_beats": int(len(beats))}


def analyze(path: str) -> dict:
    y, sr = load_audio(path)
    f0, voiced, times = extract_f0(y, sr)
    notes = segment_notes(f0, voiced, times)
    result = {
        "source": Path(path).name,
        "duration_sec": round(len(y) / sr, 2),
        "tempo": estimate_tempo(y, sr),
        "key": estimate_key(notes),
        "n_notes": len(notes),
        "pitch_range": (
            {"low": notes and min(n["note"] for n in notes) or None,
             "high": notes and max(n["note"] for n in notes) or None}
            if notes else None
        ),
        "notes": notes,
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python analyze_memo.py <audiofile>")
    result = analyze(sys.argv[1])
    out = Path(sys.argv[1]).with_suffix(".analysis.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # 人間向けサマリ
    print(f"■ {result['source']}  ({result['duration_sec']}秒)")
    k = result["key"]
    if k:
        print(f"  キー   : {k['key']} (確度 {k['confidence']})  次点: {k['runner_up']}")
    t = result["tempo"]
    print(f"  テンポ : {t['bpm']} BPM (確度 {t['confidence']})")
    print(f"  ノート数: {result['n_notes']}  音域: {result['pitch_range']}")
    print(f"  メロディ冒頭: {' '.join(n['note'] for n in result['notes'][:16])}")
    print(f"\n  → 詳細JSON: {out}")

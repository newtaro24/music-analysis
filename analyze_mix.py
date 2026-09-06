"""
2mix解析 v0 — 完成/制作中のミックス音源から「曲の設計図」を抽出する

ボイスメモ解析(単旋律)との違い:
  ポリフォニック音源ではf0追跡が使えないので、
  - キー   → クロマグラム(12音の強度分布)の集計で推定
  - 構成   → 音色・和声の特徴量が「変わる場所」をセクション境界として検出
  - 質感   → 帯域別エネルギー・オンセット密度で各セクションの密度を数値化
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 22050
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# 帯域の定義(Hz): アレンジ相談で使う言葉に対応させる
BANDS = {"low": (20, 120), "low_mid": (120, 500), "mid": (500, 2000), "high_mid": (2000, 6000), "high": (6000, 11000)}


def estimate_key_from_chroma(chroma):
    pc = chroma.mean(axis=1)
    scores = []
    for shift in range(12):
        rolled = np.roll(pc, -shift)
        scores.append((np.corrcoef(rolled, KS_MAJOR)[0, 1], f"{NOTE_NAMES[shift]} major"))
        scores.append((np.corrcoef(rolled, KS_MINOR)[0, 1], f"{NOTE_NAMES[shift]} minor"))
    scores.sort(reverse=True)
    return {"key": scores[0][1], "confidence": round(float(scores[0][0]), 3),
            "runner_up": scores[1][1], "runner_up_confidence": round(float(scores[1][0]), 3)}


def band_energy(S, freqs):
    """パワースペクトログラムから帯域ごとのエネルギー時系列(dB)を出す"""
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = librosa.power_to_db(S[mask].mean(axis=0) + 1e-10)
    return out


def detect_sections(y, sr, beats, k=None):
    """ビート同期したクロマ+MFCCを凝集クラスタリングして構成境界を検出。
    kはセクション数(未指定なら曲長から目安を決める)。"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    csync = librosa.util.sync(chroma, beats)
    msync = librosa.util.sync(mfcc, beats)
    feats = np.vstack([librosa.util.normalize(csync, axis=0), librosa.util.normalize(msync, axis=0)])
    dur = len(y) / sr
    if k is None:
        k = max(4, min(14, int(dur // 14)))  # 1セクション≒14秒を目安
    bounds = librosa.segment.agglomerative(feats, k)
    bound_times = librosa.frames_to_time(beats[np.minimum(bounds, len(beats) - 1)], sr=sr)
    bound_times = np.unique(np.concatenate([[0.0], bound_times, [dur]]))
    return bound_times


def analyze(path: str) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=45)
    dur = len(y) / sr

    # --- テンポ & ビート ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])

    # --- キー ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = estimate_key_from_chroma(chroma)

    # --- スペクトログラム(帯域解析用) ---
    S = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = band_energy(S, freqs)
    frame_times = librosa.times_like(bands["low"], sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # --- 構成検出 ---
    bound_times = detect_sections(y, sr, beats)

    # --- セクションごとの統計 ---
    sections = []
    for i in range(len(bound_times) - 1):
        t0, t1 = bound_times[i], bound_times[i + 1]
        if t1 - t0 < 2.0:
            continue
        fmask = (frame_times >= t0) & (frame_times < t1)
        omask = (librosa.times_like(onset_env, sr=sr) >= t0) & (librosa.times_like(onset_env, sr=sr) < t1)
        sec = {
            "start": round(float(t0), 1),
            "end": round(float(t1), 1),
            "bars_approx": round((t1 - t0) / (60 / tempo * 4), 1),
            "rms_db": round(float(librosa.amplitude_to_db(rms[fmask].mean() + 1e-10)), 1),
            "brightness_hz": int(centroid[fmask].mean()),
            "onset_density": round(float(onset_env[omask].mean()), 2),
            "bands_db": {k2: round(float(v[fmask].mean()), 1) for k2, v in bands.items()},
        }
        sections.append(sec)

    # エネルギーを相対化(曲中最大を100とする)して読みやすく
    if sections:
        max_rms = max(s["rms_db"] for s in sections)
        for s in sections:
            s["energy_pct"] = int(round(100 * 10 ** ((s["rms_db"] - max_rms) / 20)))

    return {
        "source": Path(path).name,
        "duration_sec": round(dur, 1),
        "tempo_bpm": round(tempo, 1),
        "key": key,
        "n_sections_detected": len(sections),
        "sections": sections,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python analyze_mix.py <audiofile> [out.json]")
    r = analyze(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).with_suffix(".analysis.json")
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"■ {r['source']}  {r['duration_sec']}秒  {r['tempo_bpm']} BPM  {r['key']['key']} (確度{r['key']['confidence']})")
    print(f"{'区間':>13} {'小節':>5} {'音量%':>4} {'明るさ':>6} {'密度':>5}  low/lowmid/mid/himid/high (dB)")
    for s in r["sections"]:
        b = s["bands_db"]
        print(f"{s['start']:>6.1f}-{s['end']:>6.1f} {s['bars_approx']:>5} {s['energy_pct']:>4} {s['brightness_hz']:>5}Hz {s['onset_density']:>5} "
              f" {b['low']:.0f}/{b['low_mid']:.0f}/{b['mid']:.0f}/{b['high_mid']:.0f}/{b['high']:.0f}")
    print(f"\n→ 詳細JSON: {out}")

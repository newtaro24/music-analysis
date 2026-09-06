"""
コード検出 v1 — 4和音テンプレート + ベース帯域によるルート推定

v0からの改善(summer_noiseの正解データから学んだこと):
  1. maj7/m7/dom7/m7b5 の4和音テンプレートを追加
     (3和音だけだと B♭M7→F、E♭M7→Gm のような「部分集合誤爆」が起きる)
  2. 低域(〜200Hz)だけのクロマを別に取り、テンプレートのルート音と一致したら加点
     (コードネームの決め手はベースが握っている)
"""
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

SR = 22050
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = {
    "": [0, 4, 7], "m": [0, 3, 7],
    "M7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "m7b5": [0, 3, 6, 10],
}

TEMPLATES, LABELS, ROOTS = [], [], []
for root in range(12):
    for q, iv in QUALITIES.items():
        t = np.zeros(12)
        for i in iv:
            t[(root + i) % 12] = 1.0
        t[root] *= 1.2  # ルートをわずかに強調
        TEMPLATES.append(t / np.linalg.norm(t))
        LABELS.append(NOTE[root] + q)
        ROOTS.append(root)
TEMPLATES = np.array(TEMPLATES)
ROOTS = np.array(ROOTS)

SWITCH_PENALTY = 0.20
BASS_BONUS = 0.35  # ベースクロマがルートと一致したときの加点


def detect(path: str):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=45)
    yh = librosa.effects.harmonic(y, margin=4)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)

    chroma = librosa.feature.chroma_cqt(y=yh, sr=sr)
    # ベース帯域クロマ: C1(32Hz)から2オクターブ分だけ見る
    bass_chroma = librosa.feature.chroma_cqt(y=yh, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=2)

    csync = librosa.util.sync(chroma, beats)
    bsync = librosa.util.sync(bass_chroma, beats)
    csync /= np.linalg.norm(csync, axis=0, keepdims=True) + 1e-9
    bsync /= np.linalg.norm(bsync, axis=0, keepdims=True) + 1e-9

    scores = TEMPLATES @ csync + BASS_BONUS * bsync[ROOTS, :]  # (72, n_beats)

    # ビタビ平滑化
    n = scores.shape[1]
    K = len(LABELS)
    dp = scores[:, 0].copy()
    back = np.zeros((K, n), dtype=int)
    for t in range(1, n):
        best_prev = int(dp.argmax())
        for c in range(K):
            if dp[c] >= dp[best_prev] - SWITCH_PENALTY:
                back[c, t] = c
                base = dp[c]
            else:
                back[c, t] = best_prev
                base = dp[best_prev] - SWITCH_PENALTY
            scores[c, t] += base
        dp = scores[:, t]
    idx = np.zeros(n, dtype=int)
    idx[-1] = int(dp.argmax())
    for t in range(n - 1, 0, -1):
        idx[t - 1] = back[idx[t], t]

    beat_times = librosa.frames_to_time(beats, sr=sr)
    segs = []
    for i, ci in enumerate(idx):
        lab = LABELS[ci]
        if segs and segs[-1]["chord"] == lab:
            segs[-1]["beats"] += 1
        else:
            segs.append({"chord": lab, "start": round(float(beat_times[min(i, len(beat_times)-1)]), 1), "beats": 1})
    return float(np.atleast_1d(tempo)[0]), segs


# ==== 正解データ (本人提供, カポ3→実音に移調済み, Aメロ/サビの主ループ) ====
GROUND_LOOP = ["A#M7", "Gm7", "D#M7", "Cm7", "F"]  # B♭M7/Gm7/E♭M7/Cm7→F


def root_of(label):
    return label[:2] if label[:2] in NOTE else label[:1]


if __name__ == "__main__":
    tempo, segs = detect(sys.argv[1])
    main = [s for s in segs if s["beats"] >= 2]
    print(f"tempo(検出): {tempo:.1f} BPM\n")
    print("コード進行(2拍以上):")
    for s in main:
        print(f"  {s['start']:>6.1f}s  {s['chord']:<6} ({s['beats']}拍)")
    freq = Counter(s["chord"] for s in main)
    print("\n出現頻度: " + "  ".join(f"{c}×{n}" for c, n in freq.most_common(10)))
    # 正解ループとの照合: 検出コードのうち正解ループの構成コードだった拍の割合
    ground_roots = {root_of(g) for g in GROUND_LOOP}
    total_beats = sum(s["beats"] for s in main)
    hit_exact = sum(s["beats"] for s in main if s["chord"] in GROUND_LOOP)
    hit_root = sum(s["beats"] for s in main if root_of(s["chord"]) in ground_roots)
    print(f"\n正解ループ({'/'.join(GROUND_LOOP)})との一致率:")
    print(f"  コード名まで一致: {100*hit_exact/total_beats:.0f}% of beats")
    print(f"  ルート音が一致  : {100*hit_root/total_beats:.0f}% of beats")

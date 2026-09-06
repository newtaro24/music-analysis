"""
コード検出 v0 — ビート同期クロマ × テンプレート照合 + 動的計画法による平滑化

原理:
  1. クロマグラム(12音の強度)をビート単位に集約
  2. 24種のコードテンプレート(メジャー/マイナー各12ルート)と照合
  3. 「コードは頻繁に変わらない」という事前知識を切替ペナルティとして入れ、
     ビタビ的DPで最尤のコード列を求める(これをやらないと1拍ごとにチラつく)

限界(v0): 7th/テンションは区別しない。Cm7はCmに、E♭maj7はE♭に潰れて出る。
"""
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np

SR = 22050
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# コードテンプレート: ルート・3度・5度を1、それ以外0
TEMPLATES, LABELS = [], []
for root in range(12):
    maj = np.zeros(12); maj[[root, (root + 4) % 12, (root + 7) % 12]] = 1
    mnr = np.zeros(12); mnr[[root, (root + 3) % 12, (root + 7) % 12]] = 1
    TEMPLATES += [maj / np.linalg.norm(maj), mnr / np.linalg.norm(mnr)]
    LABELS += [NOTE[root], NOTE[root] + "m"]
TEMPLATES = np.array(TEMPLATES)

SWITCH_PENALTY = 0.15  # コード切替のコスト(大きいほど粘る)


def detect(path: str, harmonic_sep=True):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=45)
    if harmonic_sep:
        # 打楽器成分を除いてクロマの濁りを減らす(HPSS: 調波/打撃音分離)
        y = librosa.effects.harmonic(y, margin=4)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    csync = librosa.util.sync(chroma, beats)             # (12, n_beats)
    csync = csync / (np.linalg.norm(csync, axis=0, keepdims=True) + 1e-9)
    scores = TEMPLATES @ csync                            # (24, n_beats)

    # ビタビ: 各ビートで「そのコードを維持 or 切り替え(ペナルティ)」
    n = scores.shape[1]
    dp = scores[:, 0].copy()
    back = np.zeros((24, n), dtype=int)
    for t in range(1, n):
        stay = dp
        best_prev = stay.max()
        for c in range(24):
            cand_switch = best_prev - SWITCH_PENALTY
            if stay[c] >= cand_switch:
                back[c, t] = c
                newv = stay[c]
            else:
                back[c, t] = int(stay.argmax())
                newv = cand_switch
            dp[c] = newv + scores[c, t] if False else newv  # placeholder
        dp = np.array([dp[c] + scores[c, t] for c in range(24)])
    path_idx = np.zeros(n, dtype=int)
    path_idx[-1] = int(dp.argmax())
    for t in range(n - 1, 0, -1):
        path_idx[t - 1] = back[path_idx[t], t]

    beat_times = librosa.frames_to_time(beats, sr=sr)
    # 連続する同一コードを区間にまとめる
    segs = []
    for i, ci in enumerate(path_idx):
        lab = LABELS[ci]
        t0 = beat_times[i] if i < len(beat_times) else beat_times[-1]
        if segs and segs[-1]["chord"] == lab:
            segs[-1]["beats"] += 1
        else:
            segs.append({"chord": lab, "start": round(float(t0), 1), "beats": 1})
    return float(np.atleast_1d(tempo)[0]), segs


if __name__ == "__main__":
    tempo, segs = detect(sys.argv[1])
    print(f"tempo(検出): {tempo:.1f} BPM   ※聴感はハーフタイムの可能性あり")
    print("\nコード進行(2拍以上の区間のみ):")
    main = [s for s in segs if s["beats"] >= 2]
    for s in main:
        print(f"  {s['start']:>6.1f}s  {s['chord']:<4} ({s['beats']}拍)")
    freq = Counter(s["chord"] for s in main)
    print("\n出現頻度: " + "  ".join(f"{c}×{n}" for c, n in freq.most_common(8)))

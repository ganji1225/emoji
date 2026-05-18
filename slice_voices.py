#!/usr/bin/env python3
"""2キャラ分のスライス処理"""
import os, glob
import numpy as np, soundfile as sf, librosa

TARGETS = [
    ('D:/irodori/emoji/input/mayu_ruru', 'D:/irodori/emoji/data/mayu_ruru/slices'),
    ('D:/irodori/emoji/input/f_ai',      'D:/irodori/emoji/data/f_ai/slices'),
]

MIN_SEC = 2.0
MAX_SEC = 12.0
TOP_DB  = 35

for INPUT_DIR, OUTPUT_DIR in TARGETS:
    name = os.path.basename(INPUT_DIR)
    print(f'\n=== {name} ===')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(f'{INPUT_DIR}/*.wav'))
    total_out = 0

    for f in files:
        data, sr = sf.read(f)
        if data.ndim > 1:
            data = data.mean(axis=1)
        duration = len(data) / sr
        base = os.path.splitext(os.path.basename(f))[0]

        if duration < MIN_SEC:
            continue
        if duration <= MAX_SEC:
            sf.write(f'{OUTPUT_DIR}/{base}_001.wav', data, sr)
            total_out += 1
            continue

        # 12秒超: 無音区間で分割
        intervals = librosa.effects.split(data, top_db=TOP_DB, frame_length=2048, hop_length=512)
        segments = []
        cur_start, cur_end = None, None
        for s, e in intervals:
            if cur_start is None:
                cur_start, cur_end = s, e
            else:
                seg_dur = (e - cur_start) / sr
                gap_dur = (s - cur_end) / sr
                if seg_dur > MAX_SEC:
                    segments.append((cur_start, cur_end))
                    cur_start, cur_end = s, e
                elif gap_dur > 0.6:
                    cur_dur = (cur_end - cur_start) / sr
                    if cur_dur >= MIN_SEC:
                        segments.append((cur_start, cur_end))
                        cur_start, cur_end = s, e
                    else:
                        cur_end = e
                else:
                    cur_end = e
        if cur_start is not None:
            cur_dur = (cur_end - cur_start) / sr
            if cur_dur >= MIN_SEC:
                segments.append((cur_start, cur_end))

        valid_segs = []
        for s, e in segments:
            d = (e - s) / sr
            if d < MIN_SEC:
                continue
            if d <= MAX_SEC:
                valid_segs.append((s, e))
            else:
                n = int(np.ceil(d / MAX_SEC))
                step = (e - s) // n
                for i in range(n):
                    ss = s + i * step
                    ee = s + (i+1) * step if i < n-1 else e
                    if (ee - ss) / sr >= MIN_SEC:
                        valid_segs.append((ss, ee))

        for i, (s, e) in enumerate(valid_segs):
            pad = int(sr * 0.03)
            s_pad = max(0, s - pad)
            e_pad = min(len(data), e + pad)
            sf.write(f'{OUTPUT_DIR}/{base}_{i+1:03d}.wav', data[s_pad:e_pad], sr)
            total_out += 1

        print(f'  {os.path.basename(f)}: {duration:.1f}秒 → {len(valid_segs)}セグメント')

    print(f'スライス出力合計: {total_out}本')

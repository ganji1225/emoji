#!/usr/bin/env python3
"""CFG蛟､豈碑ｼ・ユ繧ｹ繝・- 蜷後§繝・く繧ｹ繝医・seed縺ｧCFG蛟､繧貞､峨∴縺ｦ髻ｳ螢ｰ蜩∬ｳｪ縺ｮ蟾ｮ繧呈､懆ｨｼ"""
import os
import sys
import json
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import subprocess
from pathlib import Path

PYTHON = str(Path("E:/irodori/emoji/.venv/Scripts/python.exe"))
INFER = str(Path("E:/irodori/emoji/infer.py"))
OUTPUT_DIR = Path("E:/irodori/emoji/outputs/cfg_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
SEED = 42

# 繝・せ繝医こ繝ｼ繧ｹ: 3遞ｮ鬘槭・繧ｻ繝ｪ繝・ﾃ・隍・焚CFG蛟､
TEST_CASES = [
    {
        "name": "daily",
        "text": "縺翫・繧医≧縲∽ｻ頑律繧ゅ＞縺・､ｩ豌励□縺ｭ縲・,
        "caption": "闍･縺・･ｳ諤ｧ縺後√Μ繝ｩ繝・け繧ｹ縺励◆閾ｪ辟ｶ縺ｪ蜿｣隱ｿ縺ｧ縲∵・繧九￥隧ｱ縺励※縺・ｋ縲・,
        "desc": "譌･蟶ｸ莨夊ｩｱ・医ル繝･繝ｼ繝医Λ繝ｫ・・,
    },
    {
        "name": "whisper",
        "text": "\U0001f442縺ｭ縺・√ｂ縺｣縺ｨ霑代￥縺ｫ譚･縺ｦ縲・,
        "caption": "闍･縺・･ｳ諤ｧ縺後∬ｳ蜈・〒蝗√￥繧医≧縺ｫ縲∫曝縺乗沐繧峨°縺・｣ｰ縺ｧ隧ｱ縺励※縺・ｋ縲・,
        "desc": "蝗√″・郁ｳ蜈・ｼ・,
    },
    {
        "name": "emotion",
        "text": "\U0001f97a\U0001fae3縺医▲縺ｨ窶ｦ縺ゅ・縺ｭ窶ｦ螂ｽ縺阪↑莠ｺ縺後＞繧九・窶ｦ",
        "caption": "闍･縺・･ｳ諤ｧ縺後∵▼縺壹°縺励◎縺・↓繧ゅ§繧ゅ§縺励↑縺後ｉ縲∝ｰ上＆縺ｪ螢ｰ縺ｧ隧ｱ縺励※縺・ｋ縲・,
        "desc": "諢滓ュ・域▼縺倥ｉ縺・ｼ・,
    },
]

# 繝・せ繝医☆繧気FG蛟､縺ｮ邨・∩蜷医ｏ縺・CFG_PATTERNS = [
    {"label": "low",       "cfg_text": 1.5, "cfg_caption": 1.5, "cfg_speaker": 3.0},
    {"label": "mild",      "cfg_text": 2.0, "cfg_caption": 2.0, "cfg_speaker": 4.0},
    {"label": "standard",  "cfg_text": 3.0, "cfg_caption": 3.0, "cfg_speaker": 5.0},
    {"label": "high",      "cfg_text": 4.5, "cfg_caption": 4.5, "cfg_speaker": 6.0},
    {"label": "very_high", "cfg_text": 6.0, "cfg_caption": 6.0, "cfg_speaker": 7.0},
    {"label": "extreme",   "cfg_text": 8.0, "cfg_caption": 8.0, "cfg_speaker": 8.0},
]

results = []
total = len(TEST_CASES) * len(CFG_PATTERNS)
count = 0

print(f"=== CFG蛟､豈碑ｼ・ユ繧ｹ繝・({total} patterns) ===")
print(f"Model: {HF_CHECKPOINT}")
print(f"Seed: {SEED} (蝗ｺ螳・")
print()

for tc in TEST_CASES:
    print(f"\n--- {tc['name']}: {tc['desc']} ---")

    for cfg in CFG_PATTERNS:
        count += 1
        filename = f"{tc['name']}_{cfg['label']}"
        out_wav = str(OUTPUT_DIR / f"{filename}.wav")

        print(f"[{count}/{total}] {filename} (text={cfg['cfg_text']} cap={cfg['cfg_caption']} spk={cfg['cfg_speaker']})")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            PYTHON, INFER,
            "--hf-checkpoint", HF_CHECKPOINT,
            "--codec-repo", CODEC_REPO,
            "--text", tc["text"],
            "--caption", tc["caption"],
            "--no-ref",
            "--seed", str(SEED),
            "--num-steps", "30",
            "--model-precision", "bf16",
            "--cfg-scale-text", str(cfg["cfg_text"]),
            "--cfg-scale-caption", str(cfg["cfg_caption"]),
            "--cfg-scale-speaker", str(cfg["cfg_speaker"]),
            "--output-wav", out_wav,
            "--no-show-timings",
        ]

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
                env=env, encoding="utf-8", errors="replace"
            )
            elapsed = time.time() - start
            success = proc.returncode == 0 and Path(out_wav).exists()
            size = Path(out_wav).stat().st_size if Path(out_wav).exists() else 0

            results.append({
                "test_case": tc["name"],
                "desc": tc["desc"],
                "cfg_label": cfg["label"],
                "cfg_text": cfg["cfg_text"],
                "cfg_caption": cfg["cfg_caption"],
                "cfg_speaker": cfg["cfg_speaker"],
                "success": success,
                "elapsed_sec": round(elapsed, 1),
                "file_size_kb": round(size / 1024, 1),
                "wav_path": out_wav,
            })

            status = "OK" if success else "FAIL"
            print(f"  -> [{status}] {elapsed:.1f}s, {size/1024:.0f}KB")
            if not success and proc.stderr:
                print(f"  STDERR: {proc.stderr[-200:]}")

        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "test_case": tc["name"],
                "desc": tc["desc"],
                "cfg_label": cfg["label"],
                "cfg_text": cfg["cfg_text"],
                "cfg_caption": cfg["cfg_caption"],
                "cfg_speaker": cfg["cfg_speaker"],
                "success": False,
                "elapsed_sec": round(elapsed, 1),
                "file_size_kb": 0,
                "wav_path": out_wav,
                "error": str(e),
            })
            print(f"  -> [ERROR] {e}")

# Save report
report_path = OUTPUT_DIR / "cfg_comparison_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# RMS analysis
print(f"\n{'='*60}")
print(f"=== 髻ｳ螢ｰ隗｣譫・===")
try:
    import numpy as np
    import soundfile as sf

    print(f"\n{'繝・せ繝・:<12} {'CFG':<12} {'遘呈焚':<6} {'RMS':<10} {'Peak':<8} {'繝輔ぃ繧､繝ｫ'}")
    print("-" * 70)
    for r in results:
        if not r["success"]:
            continue
        data, sr = sf.read(r["wav_path"])
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        dur = len(data) / sr
        rms = float(np.sqrt(np.mean(data ** 2)))
        peak = float(np.max(np.abs(data)))
        print(f"{r['test_case']:<12} {r['cfg_label']:<12} {dur:<6.1f} {rms:<10.5f} {peak:<8.4f} {Path(r['wav_path']).name}")
except ImportError:
    print("[skip] numpy/soundfile 縺瑚ｦ九▽縺九ｊ縺ｾ縺帙ｓ")

ok = sum(1 for r in results if r["success"])
print(f"\nTest Complete! Success: {ok}/{total}")
print(f"Report: {report_path}")
print(f"Audio:  {OUTPUT_DIR}")
print(f"\n閨ｴ縺肴ｯ斐∋謗ｨ螂ｨ: 蜷дest_case縺斐→縺ｫlow竊弾xtreme 縺ｮ鬆・↓閨ｴ縺・※CFG縺ｮ蠖ｱ髻ｿ繧堤｢ｺ隱・)

#!/usr/bin/env python3
"""全38種絵文字テスト - Emoji-TTS (v1モデル) - Windows対応版"""
import os
import sys
import json
import time

# Windows console encoding fix
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import subprocess
from pathlib import Path

PYTHON = str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe")
INFER = str(Path(__file__).parent / "infer.py")
OUTPUT_DIR = Path(__file__).parent / "outputs" / "emoji_test_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_CHECKPOINT = "Aratako/Irodori-TTS-500M"
BASE_TEXT = "あのね、ちょっと聞いてほしいことがあるの。"
SEED = 42

EMOJIS = [
    ("00_baseline", "", "絵文字なし（ベースライン）"),
    ("01_whisper", "\U0001f442", "囁き、耳元の音"),
    ("02_sigh", "\U0001f62e\u200d\U0001f4a8", "吐息、溜息、寝息"),
    ("03_pause", "\u23f8\ufe0f", "間、沈黙"),
    ("04_chuckle", "\U0001f92d", "くすくす笑い"),
    ("05_pant", "\U0001f975", "喘ぎ、うめき声"),
    ("06_echo", "\U0001f4e2", "エコー、リバーブ"),
    ("07_tease", "\U0001f60f", "からかう、甘える"),
    ("08_tremble", "\U0001f97a", "声を震わせて、自信なさげ"),
    ("09_breathless", "\U0001f32c\ufe0f", "息切れ、荒い息遣い"),
    ("10_gasp", "\U0001f62e", "息をのむ"),
    ("11_lick", "\U0001f445", "舐める音、咀嚼音、水音"),
    ("12_lip", "\U0001f48b", "リップノイズ"),
    ("13_gentle", "\U0001faf6", "優しく"),
    ("14_cry", "\U0001f62d", "嗚咽、泣き声"),
    ("15_scream", "\U0001f631", "悲鳴、叫び"),
    ("16_sleepy", "\U0001f62a", "眠そう、気だるげ"),
    ("17_fast", "\u23e9", "早口、まくしたてる"),
    ("18_phone", "\U0001f4de", "電話越しの音"),
    ("19_slow", "\U0001f422", "ゆっくりと"),
    ("20_gulp", "\U0001f964", "唾を飲み込む音"),
    ("21_cough", "\U0001f927", "咳き込み、くしゃみ"),
    ("22_tsk", "\U0001f612", "舌打ち"),
    ("23_fluster", "\U0001f630", "慌てて、動揺、どもり"),
    ("24_joy", "\U0001f606", "喜びながら"),
    ("25_angry", "\U0001f620", "怒り、不満げ"),
    ("26_surprise", "\U0001f632", "驚き、感嘆"),
    ("27_yawn", "\U0001f971", "あくび"),
    ("28_pain", "\U0001f616", "苦しげに"),
    ("29_worry", "\U0001f61f", "心配そうに"),
    ("30_shy", "\U0001fae3", "恥ずかしそうに、照れ"),
    ("31_exasperate", "\U0001f644", "呆れたように"),
    ("32_happy", "\U0001f60a", "楽しげに、嬉しそうに"),
    ("33_nod", "\U0001f44c", "相槌、頷く音"),
    ("34_plead", "\U0001f64f", "懇願するように"),
    ("35_drunk", "\U0001f974", "酔っ払って"),
    ("36_hum", "\U0001f3b5", "鼻歌"),
    ("37_muffled", "\U0001f910", "口を塞がれて"),
    ("38_relief", "\U0001f60c", "安堵、満足げ"),
    ("39_question", "\U0001f914", "疑問の声"),
]

results = []
total = len(EMOJIS)
print(f"=== Emoji-TTS v1 Full Test ({total} patterns) ===")
print(f"Model: {HF_CHECKPOINT}")
print(f"Output: {OUTPUT_DIR}")
print()

for i, (name, emoji, desc) in enumerate(EMOJIS):
    text = f"{emoji}{BASE_TEXT}" if emoji else BASE_TEXT
    out_wav = str(OUTPUT_DIR / f"{name}.wav")

    print(f"[{i+1}/{total}] {name}: {desc}")

    start = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        PYTHON, INFER,
        "--hf-checkpoint", HF_CHECKPOINT,
        "--text", text,
        "--no-ref",
        "--seed", str(SEED),
        "--num-steps", "30",
        "--model-precision", "bf16",
        "--output-wav", out_wav,
        "--no-show-timings",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            env=env, encoding="utf-8", errors="replace"
        )
        elapsed = time.time() - start
        success = proc.returncode == 0 and Path(out_wav).exists()
        size = Path(out_wav).stat().st_size if Path(out_wav).exists() else 0

        results.append({
            "name": name,
            "emoji": emoji,
            "desc": desc,
            "text": text,
            "success": success,
            "elapsed_sec": round(elapsed, 1),
            "file_size_kb": round(size / 1024, 1),
            "wav_path": out_wav,
        })

        status = "OK" if success else "FAIL"
        print(f"  -> [{status}] {elapsed:.1f}s, {size/1024:.0f}KB")
        if not success and proc.stderr:
            print(f"  STDERR: {proc.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        results.append({
            "name": name, "emoji": emoji, "desc": desc, "text": text,
            "success": False, "elapsed_sec": round(elapsed, 1),
            "file_size_kb": 0, "wav_path": out_wav, "error": "TIMEOUT",
        })
        print(f"  -> [TIMEOUT] {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "name": name, "emoji": emoji, "desc": desc, "text": text,
            "success": False, "elapsed_sec": round(elapsed, 1),
            "file_size_kb": 0, "wav_path": out_wav, "error": str(e),
        })
        print(f"  -> [ERROR] {e}")

# Save JSON report
report_path = OUTPUT_DIR / "test_report_v1.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

ok = sum(1 for r in results if r["success"])
print(f"\n{'='*50}")
print(f"Test Complete! Success: {ok}/{total}")
print(f"Report: {report_path}")
print(f"Audio:  {OUTPUT_DIR}")

#!/usr/bin/env python3
"""HFデータセット/torchcodecを経由しない簡易版 prepare_manifest

ローカル CSV (file_name, text, caption) から、
DACVAE 潜在表現を直接計算してJSONLマニフェストを生成する。
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
import torch

from irodori_tts.codec import DACVAECodec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="metadata CSV (file_name, text, caption)")
    parser.add_argument("--audio-dir", required=True, help="WAVファイルのフォルダ")
    parser.add_argument("--output-manifest", required=True, help="出力JSONL")
    parser.add_argument("--latent-dir", required=True, help="潜在.ptの出力先")
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize-db", type=float, default=-16.0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    latent_dir = Path(args.latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV読み込み
    with open(args.csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[load] {len(rows)} 行のメタデータを読み込みました")

    # DACVAE codec ロード
    print(f"[load] DACVAE codec ({args.codec_repo}) ...")
    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=args.device,
        dtype=torch.float32,
        normalize_db=args.normalize_db,
    )
    print(f"[load] codec sample_rate = {codec.sample_rate} Hz")

    written = 0
    skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            fname = row["file_name"]
            text = (row.get("text") or "").strip()
            caption = (row.get("caption") or "").strip()

            wav_path = audio_dir / fname
            if not wav_path.exists():
                print(f"  [skip] file not found: {wav_path}")
                skipped += 1
                continue
            if not text:
                print(f"  [skip] empty text: {fname}")
                skipped += 1
                continue

            try:
                # WAV読み込み
                data, sr = sf.read(str(wav_path))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                # 長さ制限
                max_samples = int(args.max_seconds * sr)
                if len(data) > max_samples:
                    data = data[:max_samples]

                # tensor化
                wav_t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
                # encode (B, T_latent, D_latent)
                latent = codec.encode_waveform(
                    wav_t,
                    sample_rate=sr,
                    normalize_db=args.normalize_db,
                ).squeeze(0).cpu()  # (T_latent, D_latent)

                # 保存
                latent_filename = Path(fname).stem + ".pt"
                latent_path = latent_dir / latent_filename
                torch.save(latent, str(latent_path))

                # マニフェスト1行
                record = {
                    "text": text,
                    "caption": caption,
                    "latent_path": str(latent_path).replace("\\", "/"),
                    "num_frames": int(latent.shape[0]),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if (i + 1) % 10 == 0 or i == len(rows) - 1:
                    print(f"  [{i+1}/{len(rows)}] {fname}: {latent.shape}", flush=True)
            except Exception as e:
                print(f"  [error] {fname}: {e}")
                skipped += 1

    print(f"\n[done] 書き込み: {written}件 / スキップ: {skipped}件")
    print(f"  manifest: {manifest_path}")
    print(f"  latents: {latent_dir}")


if __name__ == "__main__":
    main()

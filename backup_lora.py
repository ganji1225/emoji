#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
LoRAバックアップスクリプト (ganji1225/emoji.git 向け)

使い方:
  python backup_lora.py                          # デフォルトLoRAでバックアップ
  python backup_lora.py --lora maria_v1 sawano_popura_v1  # LoRAを指定

仕組み:
  1. 選択LoRAの safetensors を一時フォルダにコピー（branch切り替えによる削除防止）
  2. orphanブランチ(clean-backup)でコード＋選択LoRAをコミット
  3. backup remote (ganji1225/emoji.git) へ force push
  4. main に戻り、一時フォルダから safetensors を復元
  5. 一時フォルダ削除
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ─── デフォルトのバックアップ対象LoRA ────────────────────────────────────
DEFAULT_LORAS = [
    "voice_aoyama_yukari_v1",
    "voice_ganyu_v1",
    "voice_hokuto_minami_v1",
    "voice_konomi_v1",
    "kanamatsu_yuka_v1",
    "maria_v1",
    "mizuhashi_kaori_v1",
    "sawano_popura_v1",
]

REPO_ROOT = Path(__file__).parent
LORA_DIR  = REPO_ROOT / "lora"
REMOTE    = "backup"
BRANCH    = "clean-backup"


def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check,
                          capture_output=False, text=True)


def main():
    parser = argparse.ArgumentParser(description="LoRAバックアップツール")
    parser.add_argument("--lora", nargs="+", default=DEFAULT_LORAS,
                        metavar="LORA_NAME", help="バックアップするLoRA名")
    parser.add_argument("--dry-run", action="store_true",
                        help="実際には push しない（確認用）")
    args = parser.parse_args()

    selected_loras = args.lora
    print(f"\n=== バックアップ対象LoRA ({len(selected_loras)}個) ===")
    for name in selected_loras:
        sf = LORA_DIR / name / "lora_checkpoint_final_ema" / "adapter_model.safetensors"
        status = "[OK]" if sf.exists() else "[NG] 見つからない"
        print(f"  {status}  {name}")

    missing = [n for n in selected_loras
               if not (LORA_DIR / n / "lora_checkpoint_final_ema" / "adapter_model.safetensors").exists()]
    if missing:
        print(f"\n[エラー] 以下のLoRAが見つかりません: {missing}")
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] ここで終了")
        return

    # ── STEP 1: safetensors を一時フォルダにコピー（branch切り替え前に保護）──
    print("\n=== STEP 1: safetensors を一時フォルダにバックアップ ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="lora_backup_"))
    try:
        for name in selected_loras:
            src = LORA_DIR / name / "lora_checkpoint_final_ema" / "adapter_model.safetensors"
            dst = tmp_dir / name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / "adapter_model.safetensors")
            print(f"  コピー: {name}")

        # ── STEP 2: orphanブランチ作成 ──────────────────────────────────────
        print(f"\n=== STEP 2: orphanブランチ ({BRANCH}) 作成 ===")
        # 既存の clean-backup ブランチがあれば削除
        run(["git", "branch", "-D", BRANCH], check=False)
        run(["git", "checkout", "--orphan", BRANCH])
        run(["git", "rm", "-rf", "--cached", "."], check=False)

        # ── STEP 3: コード＋設定ファイルを add ────────────────────────────
        print("\n=== STEP 3: コード・設定ファイルをステージング ===")
        run(["git", "add", "."])

        # ── STEP 4: 選択LoRAの safetensors を force-add ───────────────────
        print("\n=== STEP 4: 選択LoRAをforce-add ===")
        for name in selected_loras:
            lora_path = str(LORA_DIR / name)
            run(["git", "add", "-f", lora_path])
            print(f"  追加: {name}")

        # ── STEP 5: コミット ────────────────────────────────────────────────
        print("\n=== STEP 5: コミット ===")
        lora_list = "\n".join(f"- {n}" for n in selected_loras)
        msg = f"バックアップ: コード + 選択LoRA {len(selected_loras)}種\n\n{lora_list}"
        run(["git", "commit", "-m", msg])

        # ── STEP 6: force push ─────────────────────────────────────────────
        print(f"\n=== STEP 6: {REMOTE}/main へ force push ===")
        run(["git", "push", REMOTE, f"{BRANCH}:main", "--force"])

        # ── STEP 7: main に戻る ────────────────────────────────────────────
        print("\n=== STEP 7: main ブランチに戻る ===")
        run(["git", "checkout", "main"])
        run(["git", "branch", "-D", BRANCH])

        # ── STEP 8: safetensors を一時フォルダから復元 ────────────────────
        print("\n=== STEP 8: safetensors を復元 ===")
        for name in selected_loras:
            src = tmp_dir / name / "adapter_model.safetensors"
            dst = LORA_DIR / name / "lora_checkpoint_final_ema" / "adapter_model.safetensors"
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"  復元: {name}")
            else:
                print(f"  スキップ（既存）: {name}")

        # ── 後始末 ─────────────────────────────────────────────────────────
        # git インデックスのステージング解除（復元ファイルがステージされている場合）
        run(["git", "restore", "--staged", "lora/"], check=False)

        print("\n[完了] バックアップ完了！")
        print(f"   {REMOTE}/main → {len(selected_loras)}個のLoRA + コード一式")

    except Exception as e:
        print(f"\n[エラー] {e}")
        print("safetensors は一時フォルダに保存済み:", tmp_dir)
        # mainに戻ることだけ試みる
        run(["git", "checkout", "main"], check=False)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

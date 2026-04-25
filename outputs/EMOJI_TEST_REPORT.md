# 絵文字テスト結果レポート

**テスト日**: 2026-04-10
**テスター**: このみお姉さん（自動CLIバッチ）
**テスト環境**: RTX 4090 / CUDA 12.8 / PyTorch 2.11.0+cu128 / bf16

---

## テスト概要

| 項目 | v1モデル | VoiceDesignモデル |
|------|---------|------------------|
| モデル | Aratako/Irodori-TTS-500M | Aratako/Irodori-TTS-500M-v2-VoiceDesign |
| コーデック | facebook/dacvae-watermarked (128dim) | Aratako/Semantic-DACVAE-Japanese-32dim (32dim) |
| Caption | なし | 「若い女性が、耳元で囁くように、甘く柔らかい声で話している。」 |
| テキスト | 「あのね、ちょっと聞いてほしいことがあるの。」 | 同左 |
| Seed | 42 | 42 |
| Steps | 30 | 30 |
| Precision | bf16 | bf16 |
| 成功率 | **40/40 (100%)** | **40/40 (100%)** |
| 平均生成時間 | 約12秒/件 | 約12.5秒/件 |

---

## 音声ファイル一覧

### v1モデル: `D:\irodori\emoji\outputs\emoji_test_v1\`

| # | ファイル名 | 絵文字 | 効果 | サイズ |
|---|-----------|--------|------|--------|
| 0 | 00_baseline.wav | (なし) | ベースライン | 308KB |
| 1 | 01_whisper.wav | 👂 | 囁き、耳元の音 | 480KB |
| 2 | 02_sigh.wav | 😮‍💨 | 吐息、溜息 | 675KB |
| 3 | 03_pause.wav | ⏸️ | 間、沈黙 | 525KB |
| 4 | 04_chuckle.wav | 🤭 | くすくす笑い | 675KB |
| 5 | 05_pant.wav | 🥵 | 喘ぎ、うめき声 | 570KB |
| 6 | 06_echo.wav | 📢 | エコー、リバーブ | 431KB |
| 7 | 07_tease.wav | 😏 | からかう、甘える | 308KB |
| 8 | 08_tremble.wav | 🥺 | 声を震わせて | 390KB |
| 9 | 09_breathless.wav | 🌬️ | 息切れ、荒い息遣い | 668KB |
| 10 | 10_gasp.wav | 😮 | 息をのむ | 675KB |
| 11 | 11_lick.wav | 👅 | 舐める音、水音 | 675KB |
| 12 | 12_lip.wav | 💋 | リップノイズ | 506KB |
| 13 | 13_gentle.wav | 🫶 | 優しく | 308KB |
| 14 | 14_cry.wav | 😭 | 嗚咽、泣き声 | 390KB |
| 15 | 15_scream.wav | 😱 | 悲鳴、叫び | 308KB |
| 16 | 16_sleepy.wav | 😪 | 眠そう | 386KB |
| 17 | 17_fast.wav | ⏩ | 早口 | 308KB |
| 18 | 18_phone.wav | 📞 | 電話越しの音 | 383KB |
| 19 | 19_slow.wav | 🐢 | ゆっくりと | 450KB |
| 20 | 20_gulp.wav | 🥤 | 唾を飲み込む音 | 713KB |
| 21 | 21_cough.wav | 🤧 | 咳き込み | 668KB |
| 22 | 22_tsk.wav | 😒 | 舌打ち | 461KB |
| 23 | 23_fluster.wav | 😰 | 慌てて、どもり | 308KB |
| 24 | 24_joy.wav | 😆 | 喜びながら | 308KB |
| 25 | 25_angry.wav | 😠 | 怒り、不満げ | 386KB |
| 26 | 26_surprise.wav | 😲 | 驚き | 308KB |
| 27 | 27_yawn.wav | 🥱 | あくび | 675KB |
| 28 | 28_pain.wav | 😖 | 苦しげに | 383KB |
| 29 | 29_worry.wav | 😟 | 心配そうに | 450KB |
| 30 | 30_shy.wav | 🫣 | 恥ずかしそうに | 386KB |
| 31 | 31_exasperate.wav | 🙄 | 呆れたように | 308KB |
| 32 | 32_happy.wav | 😊 | 楽しげに | 308KB |
| 33 | 33_nod.wav | 👌 | 相槌 | 431KB |
| 34 | 34_plead.wav | 🙏 | 懇願するように | 386KB |
| 35 | 35_drunk.wav | 🥴 | 酔っ払って | 386KB |
| 36 | 36_hum.wav | 🎵 | 鼻歌 | 308KB |
| 37 | 37_muffled.wav | 🤐 | 口を塞がれて | 713KB |
| 38 | 38_relief.wav | 😌 | 安堵 | 450KB |
| 39 | 39_question.wav | 🤔 | 疑問の声 | 308KB |

### VoiceDesignモデル: `D:\irodori\emoji\outputs\emoji_test_voicedesign\`

同じファイル名構成。キャプション「若い女性が、耳元で囁くように、甘く柔らかい声で話している。」付き。

---

## 重要な発見事項

### 1. コーデックの互換性（要注意）

| モデル系統 | latent_dim | 対応コーデック | --codec-repo オプション |
|-----------|-----------|---------------|----------------------|
| v1系 (500M) | 128 | facebook/dacvae-watermarked | (デフォルト) |
| v2系 (500M-v2, VoiceDesign) | 32 | Aratako/Semantic-DACVAE-Japanese-32dim | 明示指定必要 |

**Emojiフォーク版でv2/VoiceDesignモデルを使う場合、必ず `--codec-repo Aratako/Semantic-DACVAE-Japanese-32dim` を指定すること。**

### 2. ファイルサイズから見る絵文字効果

v1モデルのファイルサイズ（＝音声の長さに比例）から効果の傾向が読み取れる：

**効果音・息遣いが追加される絵文字（ファイルサイズ大）：**
- 😮‍💨 吐息 (675KB), 🤭 笑い (675KB), 😮 息をのむ (675KB)
- 👅 舐め音 (675KB), 🌬️ 荒い息 (668KB), 🥱 あくび (675KB)
- 🤧 咳 (668KB), 🥤 唾 (713KB), 🤐 口塞ぎ (713KB)

**話し方の変化が主体の絵文字（ファイルサイズ小〜中）：**
- 😏 からかう (308KB), 😊 楽しい (308KB), 😆 喜び (308KB)
- 🥺 震え声 (390KB), 😭 泣き声 (390KB)

### 3. Web UIでのVoiceDesign使用に関する注意

- Emoji-TTSフォーク版のWeb UIでVoiceDesignモデルをロードすると `voice_design: enabled` と表示される
- 「🎨 Caption (Voice Design)」アコーディオンが出現するが、UIレイアウトの空白バグで見つけにくい
- **CLIでの使用を推奨**

---

## 聴き比べの手順

1. `D:\irodori\emoji\outputs\emoji_test_v1\` フォルダを開く
2. `00_baseline.wav`（ベースライン）を最初に聴く
3. 各絵文字のwavを聴いてベースラインとの差を確認
4. 同様に `emoji_test_voicedesign\` フォルダも確認
5. v1 vs VoiceDesignの同じ絵文字を比較

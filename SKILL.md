---
name: whisper-kit
description: 要把影片或音檔轉成逐字稿／字幕時觸發——「語音辨識」「Whisper」「自動上字幕」「轉逐字稿」「沒字幕的影片」。本機 faster-whisper，不花錢。看到專案裡自己 new WhisperModel 就該改用這個。不是給即時語音輸入用的（那是聲聲慢 / Sherpa_ASR）。
---

# whisper-kit — 打法

**本機語音辨識：影片/音檔 → 帶時間的逐字稿。顯卡優先、失敗退回 CPU、有進度、可切句、中文轉繁體、鎖語言時有防呆。**

## 何時觸發

- 功能需要「檔案 → 逐字稿」：上字幕、學習平台的逐字稿、影片內容搜尋、餵給 LLM 做摘要。
- 在專案裡看到 `from faster_whisper import WhisperModel`：那是又一份手寫版，改用 kit。

**不適用**：邊講邊出字的語音輸入（聲聲慢）、中文短句的即時辨識（Sherpa_ASR / Paraformer）。Whisper 是整檔批次處理的。

## 接線

**Node 專案**（cutube、plottr 這種：Node 服務、Python 只負責辨識）

```bash
npm i github:Jeffrey0117/whisper-kit
pip install faster-whisper opencc-python-reimplemented
```

```js
const createWhisper = require('whisper-kit')          // ESM: import createWhisper from 'whisper-kit'
const whisper = createWhisper({ python: config.python })
const result = await whisper.transcribe(file, { onProgress })
```

**Python 專案**（ReelScript、AutoReel 這種：FastAPI 長駐）

複製 `whisper_kit.py` 進專案，`from whisper_kit import transcribe`。模型會留在行程裡。

**專案已經有自己的斷句**（AutoReel 這種）：`transcribe(..., include_words=True)`，每段帶逐字時間，照原本的方式切；只把載模型、顯卡、退回 CPU 交給 kit。

**既有腳本要相容舊格式**：`python whisper_kit.py file --json` 印出的就是 plottr `transcribe.py` 一直以來的那個 JSON（`language` / `text` / `segments` / `word_count`）。

## 模式

- **辨識一律丟背景工作**，不要讓 HTTP 請求等。45 分鐘的片要跑好幾分鐘。把 `onProgress` 接到工作狀態上，畫面才有東西可顯示。
- **只需要音訊**。來源是 YouTube 就只下載音軌（`yt-dlp -f ba`），不要為了辨識抓整支影片。
- **模型怎麼選**：英文內容用 `distil-large-v3`（快、準，但**只會英文**）；多語言或不確定就 `medium`；CPU 上用 `small`。中文用 `base` 很爛，至少 `medium`。
- **要不要鎖語言**：內容確定是單一語言才鎖，而且鎖了就開 `guard_language`。不確定就留空讓它偵測。
- **怎麼切段**，三選一：
  - 要**燒進影片的字幕** → `cues: true`。依真實停頓切，一條最多 3.5 秒 / 12 個詞，時間準到字。短影音用這個。
  - 要**給人讀、逐句翻譯** → `maxWords: 30`。只在句尾切，句子完整。學習平台用這個。
  - 只要全文 → 都不設，最快。
  （Node 包裝目前只接了 `maxWords`；`cues` 請用 Python 或指令的 `--cues`。）
- **有 YouTube 字幕時先用字幕**，語音辨識當備案或「字幕太爛」時的替代。作者上傳的字幕通常比辨識好；自動產生的通常比辨識差（沒標點）。

## 地雷

- **`cublas64_12.dll is not found`**：pip 裝的 nvidia 套件把 DLL 放在 `site-packages/nvidia/*/bin`，Windows 不會去那裡找。kit 會自己處理；手寫版要記得加 PATH。而且這個錯**在開始解碼時才出現**，模型載入是成功的——所以「載入失敗才退回 CPU」的寫法擋不住。
- **鎖 `language="en"` 丟非英文的片**：Whisper 不會報錯，會編出通順的假英文。ReelScript 中過。
- **`distil-large-v3` 的語言偵測永遠回 `en`**（它是純英文蒸餾模型），不能拿它判斷語言。kit 的防呆另外用多語言的 `tiny`。
- **RTX 50 系列顯卡用 `int8` 可能丟 `CUBLAS_STATUS_NOT_SUPPORTED`**（AutoReel 踩過），要 `float16`。kit 顯卡預設就是 float16；專案設定檔裡寫死 `compute_type: int8` 又開顯卡的要留意——失敗時 kit 會退回 CPU，不會整個掛掉，但等於白開顯卡。
- **用 `torch.cuda.is_available()` 判斷有沒有顯卡**：專案沒裝 torch 就永遠是 CPU，而 faster-whisper 根本不需要 torch。用 `device: 'auto'`。
- **Windows 上 Python 的 stdout 不是 UTF-8**：中文會變亂碼或直接丟例外。Node 包裝已設 `PYTHONIOENCODING=utf-8`；自己 spawn 要記得設。
- **進度到 99 就停**：kit 故意不報 100，由呼叫端在存好結果後標完成，避免「顯示完成但資料還沒寫入」。
- **第一次用某個模型會下載**（`medium` 約 1.5GB），看起來像卡住。部署到新機器先手動跑一次。
- **伺服器用 `node --watch` 時**，辨識結果寫進專案目錄會觸發重啟、工作消失（cutube 踩過）。

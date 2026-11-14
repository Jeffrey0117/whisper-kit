# whisper-kit

本機語音辨識：影片或音檔 → 帶時間的逐字稿。用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 在自己的機器上跑，不花錢、不上雲。

一個 Python 檔（`whisper_kit.py`，可 import 也可當指令跑）加一個零依賴的 Node 包裝（`index.js`）。

## 為什麼不直接呼叫 faster-whisper

同一段「載模型 → transcribe → 整理成 segments」在 plottr、ReelScript、AutoReel、cutube 各寫了一次，每份都缺一塊。這裡是合起來的版本：

| | 來源 |
|---|---|
| 顯卡優先，失敗退回 CPU——包含「模型載得起來、開始解碼才失敗」的情況 | cutube |
| 自動找到 pip 的 `nvidia-cublas` / `nvidia-cudnn` DLL（Windows 預設找不到，症狀是 `cublas64_12.dll is not found`） | cutube |
| 邊跑邊回報進度 | cutube |
| 依標點與字數上限切成句子大小的段落 | ReelScript / AutoReel |
| 鎖定語言時的防呆：音訊明顯是別的語言就拒絕（否則 Whisper 會編出流利的假英文） | ReelScript |
| 中文轉台灣繁體（Whisper 預設出簡體） | plottr |
| 模型留在記憶體，長駐服務只付一次載入時間 | ReelScript |
| 不靠 torch 判斷有沒有顯卡（沒裝 torch 就永遠跑 CPU） | 新 |

## 安裝

```bash
pip install faster-whisper            # 必要
pip install opencc-python-reimplemented   # 選用：中文轉繁體
# 要用顯卡再加：
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Node 專案：`npm i github:Jeffrey0117/whisper-kit`
Python 專案：把 `whisper_kit.py` 複製進去（單檔，檔頭有 `__version__`）。

## Node

```js
const createWhisper = require('whisper-kit')

const whisper = createWhisper({
  python: process.env.PYTHON_PATH || 'python',   // 有 venv 就注入 venv 的 python
  maxWords: 30,                                   // 要句子大小的段落才設
})

try {
  const result = await whisper.transcribe('lesson.mp4', {
    onProgress: (percent) => job.update({ percent }),
  })
  // { language: 'en', language_probability, duration, device: 'cuda', model: 'medium',
  //   segments: [{ start: 0, end: 3.7, text: 'React, a JavaScript library…' }] }
} catch (error) {
  if (error.code === 'unexpected_language') {
    // error.language / error.probability — 告訴使用者這支片不是預期的語言
  }
  throw error
}
```

錯誤都是 `WhisperError`，用 `error.code` 分辨：`spawn_failed`（找不到 Python）、`failed`（附 stderr 結尾）、`no_result`、`timeout`、`unexpected_language`。

## Python（長駐服務直接 import）

```python
from whisper_kit import transcribe, UnexpectedLanguageError

try:
    result = transcribe(
        video_path,
        model="distil-large-v3",
        language="en", guard_language=True,   # 鎖英文，但不是英文就拒絕
        max_words=30,
        on_progress=lambda percent: update_job(percent),
    )
except UnexpectedLanguageError as error:
    ...  # error.language, error.probability
```

模型會留在行程裡，第二次呼叫不用重新載入。

## 指令

```bash
python whisper_kit.py lesson.mp4                 # 逐行輸出：P <進度> … R <結果 json>
python whisper_kit.py lesson.mp4 --json          # 一個 JSON（plottr 既有的格式）
python whisper_kit.py lesson.mp4 --language en --guard-language --max-words 30
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--model` | 顯卡 `medium`、CPU `small` | 任何 faster-whisper 模型名 |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--compute-type` | 顯卡 `float16`、CPU `int8` | |
| `--language` | 自動偵測 | |
| `--guard-language` | 關 | 搭配 `--language`；被拒絕時 exit code 3，stdout 一行 `E <json>` |
| `--max-words` | 0（不重切） | |
| `--no-vad` | | 不跳過靜音 |
| `--keep-simplified` | | 中文不轉繁體 |

## 測試

```bash
python -m unittest                  # 22 個，不需要模型
node --test "test/*.test.js"        # 6 個，用假腳本代替 Python
node example.js <影片或音檔>         # 真的跑一次
```

## 實測（RTX 5060 8GB）

2 分 08 秒的英文教學片：顯卡 `medium` 約 23 秒，CPU `small` 約 42 秒（都含載入模型）。

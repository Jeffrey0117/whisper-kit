// ============================================================================
// whisper-kit — 本機語音辨識（faster-whisper）的 Node 包裝。
// 影片/音檔 → 帶時間的逐字稿。真正的工作在 whisper_kit.py；這裡負責起 Python、
// 讀進度、把結果與錯誤整理成 JS 好用的樣子。零依賴（node 內建）。
// ============================================================================
const { spawn } = require('node:child_process');
const path = require('node:path');

const SCRIPT = path.join(__dirname, 'whisper_kit.py');
const EXIT_REFUSED = 3;
const STDERR_KEEP = 2400;
const ERROR_TAIL = 500;

class WhisperError extends Error {
  constructor(message, code, details = {}) {
    super(message);
    this.name = 'WhisperError';
    this.code = code; // 'spawn_failed' | 'failed' | 'no_result' | 'unexpected_language' | 'timeout'
    Object.assign(this, details);
  }
}

// opts → command-line flags. Kept separate so it can be tested without running Python.
function buildArgs(file, opts = {}) {
  const args = [opts.script || SCRIPT, file];
  if (opts.model) args.push('--model', String(opts.model));
  if (opts.device) args.push('--device', String(opts.device));
  if (opts.computeType) args.push('--compute-type', String(opts.computeType));
  if (opts.language) args.push('--language', String(opts.language));
  if (opts.guardLanguage) args.push('--guard-language');
  if (opts.maxWords > 0) args.push('--max-words', String(opts.maxWords));
  if (opts.vad === false) args.push('--no-vad');
  if (opts.traditionalChinese === false) args.push('--keep-simplified');
  return args;
}

/**
 * const whisper = createWhisper({ python: 'python', model: 'medium' })
 * const result  = await whisper.transcribe(file, { onProgress: (percent) => {} })
 *
 * Options (set as defaults here, or per call):
 *   python         python executable (default 'python'). Inject the venv's python when there is one.
 *   model, device ('auto' | 'cuda' | 'cpu'), computeType
 *   language       pin a language; empty = detect
 *   guardLanguage  with a pinned language: reject audio that is confidently something else
 *   maxWords       > 0: sentence-sized segments
 *   vad            false = do not skip silence
 *   traditionalChinese  false = leave Chinese as Whisper wrote it
 *   timeoutMs      kill the run after this long (default: no limit)
 *   onProgress     (percent 0-99) => void   — per call
 */
function createWhisper(defaults = {}) {
  function transcribe(file, callOpts = {}) {
    const opts = { ...defaults, ...callOpts };
    const python = opts.python || 'python';

    return new Promise((resolve, reject) => {
      let child;
      try {
        child = spawn(python, buildArgs(file, opts), {
          windowsHide: true,
          env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
        });
      } catch (error) {
        reject(new WhisperError(`could not start ${python}: ${error.message}`, 'spawn_failed'));
        return;
      }

      let pending = '';
      let stderr = '';
      let result = null;
      let refusal = null;
      let timedOut = false;
      const timer = opts.timeoutMs ? setTimeout(() => { timedOut = true; child.kill(); }, opts.timeoutMs) : null;

      const onLine = (line) => {
        if (line.startsWith('P ')) {
          if (opts.onProgress) opts.onProgress(Number(line.slice(2)) || 0);
        } else if (line.startsWith('R ')) {
          result = JSON.parse(line.slice(2));
        } else if (line.startsWith('E ')) {
          refusal = JSON.parse(line.slice(2));
        }
      };

      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => {
        const lines = (pending + chunk).split(/\r?\n/);
        pending = lines.pop();
        try { lines.forEach(onLine); } catch (error) { stderr += `\nbad line from whisper_kit.py: ${error.message}`; }
      });
      child.stderr.on('data', (chunk) => { stderr = (stderr + chunk).slice(-STDERR_KEEP); });

      child.on('error', (error) => {
        clearTimeout(timer);
        const hint = error.code === 'ENOENT' ? `找不到 ${python}（Python 沒裝，或沒指定對的路徑）` : error.message;
        reject(new WhisperError(hint, 'spawn_failed'));
      });

      child.on('close', (code) => {
        clearTimeout(timer);
        if (pending) { try { onLine(pending); } catch { /* reported below as no_result */ } }
        if (timedOut) return reject(new WhisperError(`語音辨識超過 ${opts.timeoutMs}ms 被中止`, 'timeout'));
        if (code === EXIT_REFUSED && refusal) return reject(new WhisperError(refusal.message, refusal.code, refusal));
        if (code !== 0) return reject(new WhisperError(`語音辨識失敗 (exit ${code}): ${stderr.trim().slice(-ERROR_TAIL)}`, 'failed'));
        if (!result) return reject(new WhisperError('語音辨識沒有回傳結果', 'no_result'));
        return resolve(result);
      });
    });
  }

  return { transcribe };
}

module.exports = createWhisper;
module.exports.createWhisper = createWhisper;
module.exports.buildArgs = buildArgs;
module.exports.WhisperError = WhisperError;
module.exports.SCRIPT = SCRIPT;

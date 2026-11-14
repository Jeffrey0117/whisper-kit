// 跑法：node example.js <影片或音檔>
// 不給檔案就只印說明（不需要 Python）。
const createWhisper = require('./index');

async function main() {
  const file = process.argv[2];
  if (!file) {
    process.stdout.write('用法：node example.js <影片或音檔>\n需要 Python 與 pip install faster-whisper\n');
    return;
  }

  const whisper = createWhisper({ python: process.env.PYTHON_PATH || 'python', maxWords: 30 });
  const result = await whisper.transcribe(file, {
    onProgress: (percent) => process.stdout.write(`\r辨識中 ${percent}%`),
  });

  process.stdout.write(`\r${result.language}，${result.model}，${result.device}，${result.segments.length} 段\n`);
  for (const segment of result.segments.slice(0, 5)) {
    process.stdout.write(`[${segment.start.toFixed(1)}–${segment.end.toFixed(1)}] ${segment.text}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.code || 'error'}: ${error.message}\n`);
  process.exitCode = 1;
});

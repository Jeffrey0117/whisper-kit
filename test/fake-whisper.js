// Stands in for whisper_kit.py in tests: node runs this instead of python, so no model is needed.
// The "file" argument picks the behaviour.
const mode = process.argv[2];
const out = (line) => process.stdout.write(`${line}\n`);

if (mode === 'ok') {
  out('P 10');
  process.stdout.write('P 5'); // a progress line split across two chunks
  setTimeout(() => {
    out('5');
    out(`R ${JSON.stringify({ language: 'zh', device: 'cpu', model: 'small', segments: [{ start: 0, end: 1, text: '你好' }], args: process.argv.slice(3) })}`);
  }, 20);
} else if (mode === 'refuse') {
  out(`E ${JSON.stringify({ code: 'unexpected_language', expected: 'en', language: 'ja', probability: 0.93, message: 'expected en audio, detected ja (93%)' })}`);
  process.exitCode = 3;
} else if (mode === 'crash') {
  process.stderr.write('Transcription failed: boom\n');
  process.exitCode = 1;
} else if (mode === 'silent') {
  out('P 50');
} else if (mode === 'hang') {
  setTimeout(() => {}, 60000);
}

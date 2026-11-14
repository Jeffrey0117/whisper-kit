const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const createWhisper = require('../index');
const { buildArgs, WhisperError, SCRIPT } = createWhisper;

// node plays the part of python, running a stub instead of whisper_kit.py
const fake = { python: process.execPath, script: path.join(__dirname, 'fake-whisper.js') };

test('buildArgs: defaults add nothing; every option maps to its flag', () => {
  assert.deepEqual(buildArgs('a.mp4'), [SCRIPT, 'a.mp4']);
  assert.deepEqual(
    buildArgs('a.mp4', { script: 's.py', model: 'medium', device: 'cpu', computeType: 'int8', language: 'en', guardLanguage: true, maxWords: 30, vad: false, traditionalChinese: false }),
    ['s.py', 'a.mp4', '--model', 'medium', '--device', 'cpu', '--compute-type', 'int8', '--language', 'en', '--guard-language', '--max-words', '30', '--no-vad', '--keep-simplified'],
  );
  assert.deepEqual(buildArgs('a.mp4', { maxWords: 0, vad: true, guardLanguage: false }), [SCRIPT, 'a.mp4']);
});

test('result, progress (even when a line arrives in two chunks) and UTF-8 text', async () => {
  const seen = [];
  const result = await createWhisper(fake).transcribe('ok', { onProgress: (p) => seen.push(p) });
  assert.deepEqual(seen, [10, 55]);
  assert.equal(result.segments[0].text, '你好');
});

test('per-call options override the defaults and reach the script', async () => {
  const whisper = createWhisper({ ...fake, model: 'small', language: 'en' });
  const result = await whisper.transcribe('ok', { model: 'medium' });
  assert.deepEqual(result.args, ['--model', 'medium', '--language', 'en']);
});

test('a refusal becomes an error the caller can branch on', async () => {
  await assert.rejects(createWhisper(fake).transcribe('refuse'), (error) => {
    assert.ok(error instanceof WhisperError);
    assert.equal(error.code, 'unexpected_language');
    assert.equal(error.language, 'ja');
    return true;
  });
});

test('a crash carries the stderr tail; finishing without a result is its own error', async () => {
  await assert.rejects(createWhisper(fake).transcribe('crash'), (error) => error.code === 'failed' && /boom/.test(error.message));
  await assert.rejects(createWhisper(fake).transcribe('silent'), (error) => error.code === 'no_result');
});

test('missing python and timeouts are reported, not hung', async () => {
  await assert.rejects(createWhisper({ python: 'definitely-not-a-real-python' }).transcribe('x'), (error) => error.code === 'spawn_failed');
  await assert.rejects(createWhisper({ ...fake, timeoutMs: 200 }).transcribe('hang'), (error) => error.code === 'timeout');
});

"""Tests for everything that does not need a model. Run: python -m unittest"""
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import whisper_kit as kit


class PlanAttemptsTest(unittest.TestCase):
    def test_auto_tries_gpu_then_cpu_with_their_own_defaults(self):
        self.assertEqual(kit.plan_attempts(), [("cuda", "float16", "medium"), ("cpu", "int8", "small")])

    def test_explicit_model_is_used_on_both(self):
        self.assertEqual(kit.plan_attempts("large-v3"), [("cuda", "float16", "large-v3"), ("cpu", "int8", "large-v3")])

    def test_single_device(self):
        self.assertEqual(kit.plan_attempts(device="cpu"), [("cpu", "int8", "small")])
        self.assertEqual(kit.plan_attempts(device="cuda"), [("cuda", "float16", "medium")])

    def test_gpu_only_compute_type_is_not_passed_to_cpu(self):
        self.assertEqual(kit.plan_attempts(compute_type="float16")[1], ("cpu", "int8", "small"))
        self.assertEqual(kit.plan_attempts(compute_type="int8_float16")[1], ("cpu", "int8_float16", "small"))
        self.assertEqual(kit.plan_attempts(compute_type="float32")[1], ("cpu", "float32", "small"))

    def test_unknown_device_is_an_error(self):
        with self.assertRaises(ValueError):
            kit.plan_attempts(device="gpu")


class SplitWordsTest(unittest.TestCase):
    def test_cuts_at_sentence_end(self):
        words = [(0.0, 0.4, " React"), (0.4, 0.8, " is"), (0.8, 1.2, " great."), (1.5, 2.0, " Next"), (2.0, 2.6, " one?")]
        self.assertEqual(kit.split_words(words, 30), [
            {"start": 0.0, "end": 1.2, "text": "React is great."},
            {"start": 1.5, "end": 2.6, "text": "Next one?"},
        ])

    def test_cuts_long_sentences_at_the_word_limit_and_keeps_the_tail(self):
        words = [(i, i + 1, f" w{i}") for i in range(5)]
        self.assertEqual([s["text"] for s in kit.split_words(words, 2)], ["w0 w1", "w2 w3", "w4"])

    def test_chinese_is_joined_without_spaces(self):
        words = [(0, 1, "今天"), (1, 2, "天氣"), (2, 3, "很好。"), (3, 4, "走吧")]
        self.assertEqual([s["text"] for s in kit.split_words(words, 30)], ["今天天氣很好。", "走吧"])

    def test_blank_pieces_are_dropped_and_input_is_not_changed(self):
        words = [(0, 1, " "), (1, 2, " .")]
        before = list(words)
        self.assertEqual(kit.split_words(words, 1), [{"start": 1, "end": 2, "text": "."}])
        self.assertEqual(words, before)


class PlainFloatTest(unittest.TestCase):
    def test_timings_come_back_as_plain_floats_even_from_float_subclasses(self):
        class NumpyLike(float):
            pass

        piece = kit.split_words([(NumpyLike(0.0), NumpyLike(1.234), " hi.")], 30)[0]
        self.assertIs(type(piece["start"]), float)
        self.assertIs(type(piece["end"]), float)
        self.assertEqual(piece["end"], 1.23)


class HelpersTest(unittest.TestCase):
    def test_progress_never_reaches_100_and_survives_bad_durations(self):
        self.assertEqual(kit.progress_percent(30, 120), 25)
        self.assertEqual(kit.progress_percent(500, 120), 99)
        self.assertEqual(kit.progress_percent(5, 0), 0)
        self.assertEqual(kit.progress_percent(5, None), 0)

    def test_converter_only_touches_chinese(self):
        self.assertEqual(kit.traditional_converter("en")("软件"), "软件")
        self.assertEqual(kit.traditional_converter("zh", enabled=False)("软件"), "软件")

    def test_plottr_json_shape(self):
        result = {"language": "en", "device": "cpu", "model": "small",
                  "segments": [{"start": 0, "end": 1, "text": "hello there"}, {"start": 1, "end": 2, "text": "friend"}]}
        out = kit.to_plottr_json(result)
        self.assertEqual(out["text"], "hello there friend")
        self.assertEqual(out["word_count"], 3)
        self.assertEqual(out["segments"], result["segments"])
        self.assertEqual(kit.to_plottr_json({**result, "segments": []})["word_count"], 0)


class TranscribeControlFlowTest(unittest.TestCase):
    """transcribe() with the model work mocked out: fallback and the language guard."""

    def setUp(self):
        patcher = mock.patch("os.path.exists", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_falls_back_to_cpu_when_the_gpu_fails_mid_run(self):
        calls = []

        def fake_run(path, device, *rest):
            calls.append(device)
            if device == "cuda":
                raise RuntimeError("cublas64_12.dll is not found")
            return {"device": device}

        with mock.patch.object(kit, "_run", fake_run), redirect_stderr(io.StringIO()):
            self.assertEqual(kit.transcribe("x.mp4"), {"device": "cpu"})
        self.assertEqual(calls, ["cuda", "cpu"])

    def test_raises_when_every_device_fails(self):
        with mock.patch.object(kit, "_run", side_effect=RuntimeError("boom")), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "every device.*boom"):
                kit.transcribe("x.mp4")

    def test_guard_refuses_confident_other_language_only(self):
        ok = {"device": "cpu"}
        with mock.patch.object(kit, "_run", return_value=ok):
            with mock.patch.object(kit, "detect_language", return_value=("ja", 0.93)):
                with self.assertRaises(kit.UnexpectedLanguageError) as caught:
                    kit.transcribe("x.mp4", language="en", guard_language=True)
                self.assertEqual((caught.exception.language, caught.exception.expected), ("ja", "en"))
                # not pinned, or guard off → never checked
                self.assertEqual(kit.transcribe("x.mp4", guard_language=True), ok)
                self.assertEqual(kit.transcribe("x.mp4", language="en"), ok)
            with mock.patch.object(kit, "detect_language", return_value=("ja", 0.31)):
                self.assertEqual(kit.transcribe("x.mp4", language="en", guard_language=True), ok)
            with mock.patch.object(kit, "detect_language", return_value=("en", 0.99)):
                self.assertEqual(kit.transcribe("x.mp4", language="en", guard_language=True), ok)

    def test_a_broken_guard_does_not_block_transcription(self):
        with mock.patch.object(kit, "_run", return_value={"device": "cpu"}), redirect_stderr(io.StringIO()):
            with mock.patch.object(kit, "detect_language", side_effect=RuntimeError("no model")):
                self.assertEqual(kit.transcribe("x.mp4", language="en", guard_language=True), {"device": "cpu"})

    def test_missing_file(self):
        with mock.patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                kit.transcribe("nope.mp4")


class CliTest(unittest.TestCase):
    RESULT = {"language": "en", "language_probability": 0.99, "duration": 2.0, "device": "cpu", "model": "small",
              "segments": [{"start": 0, "end": 2, "text": "hi"}]}

    def run_cli(self, argv, **patch):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(kit, "transcribe", **patch), redirect_stdout(out), redirect_stderr(err):
            code = kit.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_line_protocol_reports_progress_then_result(self):
        def fake(path, on_progress=None, **options):
            on_progress(40)
            return self.RESULT

        code, out, _ = self.run_cli(["a.mp4"], side_effect=fake)
        self.assertEqual(code, 0)
        lines = out.strip().split("\n")
        self.assertEqual(lines[0], "P 40")
        self.assertEqual(json.loads(lines[1][2:]), self.RESULT)

    def test_json_mode_prints_one_object_and_no_progress(self):
        code, out, _ = self.run_cli(["a.mp4", "--json"], return_value=self.RESULT)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["word_count"], 1)

    def test_options_reach_transcribe(self):
        with mock.patch.object(kit, "transcribe", return_value=self.RESULT) as fake, redirect_stdout(io.StringIO()):
            kit.main(["a.mp4", "--model", "medium", "--device", "cpu", "--language", "en",
                      "--guard-language", "--max-words", "30", "--no-vad", "--keep-simplified", "--json"])
        options = fake.call_args.kwargs
        self.assertEqual((options["model"], options["device"], options["language"]), ("medium", "cpu", "en"))
        self.assertEqual((options["guard_language"], options["max_words"]), (True, 30))
        self.assertEqual((options["vad"], options["traditional_chinese"]), (False, False))

    def test_refusal_is_a_structured_line_and_its_own_exit_code(self):
        error = kit.UnexpectedLanguageError("en", "ja", 0.93)
        code, out, _ = self.run_cli(["a.mp4"], side_effect=error)
        self.assertEqual(code, kit.EXIT_REFUSED)
        self.assertEqual(json.loads(out.strip()[2:])["code"], "unexpected_language")

    def test_other_failures_go_to_stderr_with_exit_1(self):
        code, out, err = self.run_cli(["a.mp4"], side_effect=RuntimeError("boom"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("boom", err)


if __name__ == "__main__":
    unittest.main()

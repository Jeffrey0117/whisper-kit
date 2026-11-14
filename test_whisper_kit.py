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


class RunTest(unittest.TestCase):
    """_run() against a fake model: what each option does to the output."""

    class Word:
        def __init__(self, start, end, word, probability=0.9):
            self.start, self.end, self.word, self.probability = start, end, word, probability

    class Seg:
        def __init__(self, start, end, text, words=None):
            self.start, self.end, self.text, self.words = start, end, text, words

    class Info:
        language, language_probability, duration = "en", 0.98, 10.0

    def run_kit(self, segments, **options):
        seen = {}

        class FakeModel:
            def transcribe(_, path, **kwargs):
                seen.update(kwargs)
                return iter(segments), RunTest.Info()

        defaults = dict(language="", max_words=0, vad=True, vad_parameters=None,
                        traditional_chinese=True, on_progress=None, include_words=False)
        with mock.patch.object(kit, "load_model", return_value=FakeModel()):
            result = kit._run("a.mp4", "cpu", "int8", "small", **{**defaults, **options})
        return result, seen

    def segments(self):
        W = self.Word
        return [self.Seg(0.0, 2.0, " Hello there.", [W(0.0, 1.0, " Hello"), W(1.0, 2.0, " there.", 0.5)]),
                self.Seg(5.0, 6.0, " Bye.", None)]

    def test_plain_run_reports_progress_and_asks_for_no_word_timings(self):
        seen_progress = []
        result, asked = self.run_kit(self.segments(), on_progress=seen_progress.append)
        self.assertEqual(result["segments"], [{"start": 0.0, "end": 2.0, "text": "Hello there."},
                                              {"start": 5.0, "end": 6.0, "text": "Bye."}])
        self.assertEqual(seen_progress, [20, 60])
        self.assertFalse(asked["word_timestamps"])
        self.assertEqual((result["device"], result["model"], result["language"]), ("cpu", "small", "en"))

    def test_include_words_keeps_segments_whole_and_attaches_word_timings(self):
        result, asked = self.run_kit(self.segments(), include_words=True)
        self.assertTrue(asked["word_timestamps"])
        first, second = result["segments"]
        self.assertEqual(first["text"], "Hello there.")
        self.assertEqual(first["words"], [
            {"word": " Hello", "start": 0.0, "end": 1.0, "probability": 0.9},
            {"word": " there.", "start": 1.0, "end": 2.0, "probability": 0.5},
        ])
        self.assertEqual(second["words"], [])

    def test_max_words_splits_and_can_be_combined_with_vad_parameters(self):
        result, asked = self.run_kit(self.segments(), max_words=1, vad_parameters={"min_silence_duration_ms": 500})
        self.assertEqual([s["text"] for s in result["segments"]], ["Hello", "there.", "Bye."])
        self.assertEqual(asked["vad_parameters"], {"min_silence_duration_ms": 500})
        self.assertNotIn("vad_parameters", self.run_kit(self.segments(), vad=False, vad_parameters={"x": 1})[1])


class PlainFloatTest(unittest.TestCase):
    def test_timings_come_back_as_plain_floats_even_from_float_subclasses(self):
        class NumpyLike(float):
            pass

        piece = kit.split_words([(NumpyLike(0.0), NumpyLike(1.234), " hi.")], 30)[0]
        self.assertIs(type(piece["start"]), float)
        self.assertIs(type(piece["end"]), float)
        self.assertEqual(piece["end"], 1.234)  # millisecond precision


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

        def fake_run(path, device, *rest, **options):
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


class CueTest(unittest.TestCase):
    """build_cues: subtitle cues cut where speech really pauses (ported from AutoReel's recognizer)."""

    class W:
        def __init__(self, start, end, word):
            self.start, self.end, self.word = start, end, word

    class Seg:
        def __init__(self, words=None, start=0.0, end=0.0, text=""):
            self.words, self.start, self.end, self.text = words, start, end, text

    def words(self, *triples):
        return [self.W(*t) for t in triples]

    def cues(self, *segments, **options):
        return list(kit.build_cues(iter(segments), **options))

    def test_cuts_at_sentence_end_and_at_a_real_pause(self):
        seg = self.Seg(self.words((0.0, 0.5, " Hello"), (0.5, 1.0, " there."), (1.1, 1.6, " How"), (1.6, 2.2, " are"),
                                  (3.0, 3.6, " you"), (3.6, 4.2, " today")))
        self.assertEqual(self.cues(seg), [(0.0, 1.0, "Hello there."), (1.1, 2.2, "How are"), (3.0, 4.2, "you today")])

    def test_cues_run_across_segment_boundaries(self):
        first = self.Seg(self.words((0.0, 0.5, " one"), (0.5, 1.0, " two")))
        second = self.Seg(self.words((1.0, 1.5, " three."), (1.6, 2.4, " four")))
        self.assertEqual(self.cues(first, second), [(0.0, 1.5, "one two three."), (1.6, 2.4, "four")])

    def test_a_forced_cut_hands_trailing_connectors_to_the_next_cue(self):
        triples = [(i * 0.3, i * 0.3 + 0.3, f" w{i}") for i in range(11)] + [(3.3, 3.6, " and"), (3.6, 3.9, " the"),
                                                                              (3.9, 4.2, " end")]
        result = self.cues(self.Seg(self.words(*triples)))
        self.assertEqual(result[0][2], " ".join(f"w{i}" for i in range(11)))
        self.assertEqual(result[1][2], "and the end")
        self.assertAlmostEqual(result[1][0], 3.3)

    def test_long_unpunctuated_speech_is_capped_by_word_count_and_duration(self):
        many = self.Seg(self.words(*[(i * 0.1, i * 0.1 + 0.1, f" x{i}") for i in range(30)]))
        self.assertTrue(all(len(text.split()) <= 12 for _, _, text in self.cues(many)))
        slow = self.Seg(self.words(*[(i * 1.0, i * 1.0 + 0.9, f" y{i}") for i in range(8)]))
        self.assertTrue(all(end - start <= 3.5 + 0.9 for start, end, _ in self.cues(slow)))
        self.assertEqual(len(self.cues(many, max_words=5)), 6)

    def test_a_very_short_cue_right_after_another_is_merged_back(self):
        seg = self.Seg(self.words((0.0, 0.5, " Wait."), (0.55, 0.8, " No.")))
        self.assertEqual(self.cues(seg), [(0.0, 0.8, "Wait. No.")])
        apart = self.Seg(self.words((0.0, 0.5, " Wait."), (2.0, 2.2, " No.")))
        self.assertEqual(len(self.cues(apart)), 2)

    def test_a_segment_without_word_timings_becomes_one_cue_and_nothing_is_lost(self):
        result = self.cues(self.Seg(self.words((0.0, 1.0, " Before"))), self.Seg(None, 2.0, 4.0, " whole segment "),
                           self.Seg(self.words((5.0, 6.0, " After"))))
        self.assertEqual([text for _, _, text in result], ["Before", "whole segment", "After"])

    def test_empty_input(self):
        self.assertEqual(self.cues(), [])
        self.assertEqual(self.cues(self.Seg(None, 0, 1, "   ")), [])


class RunOptionsTest(RunTest):
    def test_cues_mode_uses_word_timings_and_reports_each_cue_as_it_appears(self):
        seen = []
        result, asked = self.run_kit(self.segments(), cues=True, on_segment=seen.append)
        self.assertTrue(asked["word_timestamps"])
        self.assertEqual([s["text"] for s in result["segments"]], ["Hello there.", "Bye."])
        self.assertEqual(seen, result["segments"])
        self.assertIs(type(result["segments"][0]["start"]), float)

    def test_cue_options_and_decode_options_are_passed_through(self):
        result, asked = self.run_kit(self.segments(), cues=True, cue_options={"max_words": 1},
                                     decode_options={"beam_size": 5, "condition_on_previous_text": True})
        self.assertEqual([s["text"] for s in result["segments"]], ["Hello", "there.", "Bye."])
        self.assertEqual((asked["beam_size"], asked["condition_on_previous_text"]), (5, True))

    def test_cli_flags(self):
        with mock.patch.object(kit, "transcribe", return_value=CliTest.RESULT) as fake, redirect_stdout(io.StringIO()):
            kit.main(["a.mp4", "--cues", "--beam-size", "5", "--json"])
            self.assertEqual((fake.call_args.kwargs["cues"], fake.call_args.kwargs["decode_options"]), (True, {"beam_size": 5}))
            kit.main(["a.mp4", "--json"])
            self.assertEqual((fake.call_args.kwargs["cues"], fake.call_args.kwargs["decode_options"]), (False, None))


if __name__ == "__main__":
    unittest.main()

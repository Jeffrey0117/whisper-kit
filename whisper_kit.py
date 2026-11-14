#!/usr/bin/env python3
"""whisper-kit — local speech-to-text with faster-whisper. One file: import it, or run it.

    from whisper_kit import transcribe
    result = transcribe("lesson.mp4", on_progress=print)
    result["segments"]  # [{"start": 0.0, "end": 4.28, "text": "..."}]

    python whisper_kit.py lesson.mp4            # line protocol on stdout (see below)
    python whisper_kit.py lesson.mp4 --json     # one JSON object on stdout

What it adds over calling faster-whisper directly:
  - GPU first, CPU fallback — also when the GPU only fails once decoding starts
  - finds the CUDA DLLs that pip's nvidia-* wheels install where Windows does not look
  - progress while it runs
  - optional sentence-sized segments (split at punctuation / a word limit)
  - optional guard against a pinned language that does not match the audio
  - Chinese output converted to Taiwan traditional characters
  - models stay loaded between calls (for long-lived services)

Line protocol (default CLI output), one line each:
    P <0-100>     progress
    R <json>      result
    E <json>      a refusal the caller should handle, e.g. {"code": "unexpected_language", ...}
Other failures: message on stderr, exit code 1.

Only dependency: faster-whisper. opencc is optional (without it Chinese stays as Whisper wrote it).
"""
import argparse
import json
import os
import sys

__version__ = "0.3.0"

GPU_MODEL = "medium"
CPU_MODEL = "small"
DETECT_MODEL = "tiny"  # multilingual; English-only distilled models always report "en"
SENTENCE_END = (".", "?", "!", "。", "？", "！")
EXIT_REFUSED = 3

_models = {}
_dll_dirs_exposed = False


class UnexpectedLanguageError(Exception):
    """The audio is confidently in a different language than the one that was pinned."""

    code = "unexpected_language"

    def __init__(self, expected: str, language: str, probability: float):
        self.expected = expected
        self.language = language
        self.probability = probability
        super().__init__(f"expected {expected} audio, detected {language} ({probability:.0%})")


# ── model loading ──────────────────────────────────────────────────────────────────────────

def expose_nvidia_libraries() -> None:
    """Make the DLLs from pip's nvidia-cublas / nvidia-cudnn wheels loadable on Windows."""
    global _dll_dirs_exposed
    if _dll_dirs_exposed or sys.platform != "win32":
        return
    import glob
    import site

    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    folders = [d for root in roots for d in glob.glob(os.path.join(root, "nvidia", "*", "bin"))]
    for folder in folders:
        os.add_dll_directory(folder)
    if folders:
        os.environ["PATH"] = os.pathsep.join(folders + [os.environ.get("PATH", "")])
    _dll_dirs_exposed = True


def plan_attempts(model: str = "", device: str = "auto", compute_type: str = "") -> list:
    """(device, compute type, model) to try, in order. 'auto' tries the GPU first."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, not {device!r}")
    attempts = []
    if device in ("auto", "cuda"):
        attempts.append(("cuda", compute_type or "float16", model or GPU_MODEL))
    if device in ("auto", "cpu"):
        # a compute type chosen for the GPU (float16) does not exist on CPU
        cpu_compute = compute_type if compute_type.startswith("int8") or compute_type == "float32" else "int8"
        attempts.append(("cpu", cpu_compute, model or CPU_MODEL))
    return attempts


def load_model(name: str, device: str, compute_type: str):
    """Loaded models are kept, so a service pays the load time once."""
    key = (name, device, compute_type)
    if key not in _models:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:  # optional: lets model downloads use the Windows certificate store
            import truststore
            truststore.inject_into_ssl()
        except Exception:
            pass
        from faster_whisper import WhisperModel
        _models[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _models[key]


def forget_model(name: str, device: str, compute_type: str) -> None:
    _models.pop((name, device, compute_type), None)


# ── pure helpers (no model needed) ─────────────────────────────────────────────────────────

def split_words(words: list, max_words: int) -> list:
    """Word timings → sentence-sized segments: cut at sentence-ending punctuation, or at
    max_words when a sentence runs long. `words` are (start, end, text) with Whisper's own
    spacing (leading space in English, none in Chinese/Japanese), so text is joined as-is."""
    segments = []
    current = []
    for start, end, text in words:
        current.append((start, end, text))
        ended = text.rstrip().endswith(SENTENCE_END)
        if ended or len(current) >= max_words:
            segments.append(_join(current))
            current = []
    if current:
        segments.append(_join(current))
    return [s for s in segments if s["text"]]


def _join(words: list) -> dict:
    text = "".join(w[2] for w in words).strip()
    # float(): word timings arrive as numpy floats, which leak into reprs and some serializers
    return {"start": round(float(words[0][0]), 3), "end": round(float(words[-1][1]), 3), "text": text}


CUE_DEFAULTS = {"max_dur": 3.5, "max_words": 12, "gap_cut": 0.6, "min_dur": 0.5}
CUE_CONNECTORS = frozenset({
    "the", "a", "an", "to", "of", "and", "or", "but", "for", "with", "that",
    "this", "your", "you're", "is", "are", "was", "were", "be", "not", "my",
    "in", "on", "at", "as", "so", "who", "what", "when", "how", "more", "less",
})


def build_cues(segments, *, max_dur=3.5, max_words=12, gap_cut=0.6, min_dur=0.5):
    """Whisper segments (with .words) → subtitle cues, yielded as (start, end, text).

    Tuned on AutoReel's short videos, where one cue per Whisper segment gave 8-second blobs.
    Cuts where speech really pauses (gap >= gap_cut), where a sentence ends, or at a comma once
    the cue is long enough; forces a cut at max_dur seconds / max_words words, and then hands
    trailing connector words ("and", "the", ...) to the next cue so the break reads naturally.
    A cue shorter than min_dur that follows immediately is merged into the previous one.
    It is a generator over a lazy iterator, so cues come out while decoding is still running.
    """
    current = []    # words of the cue being built
    pending = None  # one cue held back, in case the next one is too short to stand alone

    def make(words):
        text = "".join(w.word for w in words).strip()
        return (words[0].start, words[-1].end, text) if text else None

    def emit(cue):
        nonlocal pending
        if cue is None:
            return None
        if pending is None:
            pending = cue
            return None
        p_start, p_end, p_text = pending
        start, end, text = cue
        if (end - start) < min_dur and (start - p_end) < 0.3:
            pending = (p_start, end, (p_text + " " + text).strip())
            return None
        ready = pending
        pending = cue
        return ready

    def cut(forced):
        carry = []
        if forced:
            while (len(current) > 2 and len(carry) < 3
                   and current[-1].word.strip().lower().strip(".,!?") in CUE_CONNECTORS):
                carry.append(current.pop())
        cue = make(current)
        current.clear()
        current.extend(reversed(carry))
        return emit(cue)

    for segment in segments:
        words = getattr(segment, "words", None) or []
        if not words:  # rare: no word timings → the whole segment is one cue, nothing is dropped
            if current:
                ready = cut(False)
                if ready:
                    yield ready
            text = (segment.text or "").strip()
            ready = emit((segment.start, segment.end, text)) if text else None
            if ready:
                yield ready
            continue
        for word in words:
            if current:
                gap = word.start - current[-1].end
                duration = word.end - current[0].start
                previous = current[-1].word.strip()
                natural = (
                    gap >= gap_cut
                    or previous.endswith(SENTENCE_END)
                    or (previous.endswith((",", "，")) and duration > 2.0)
                )
                forced = duration > max_dur or len(current) >= max_words
                if natural or forced:
                    ready = cut(forced and not natural)
                    if ready:
                        yield ready
            current.append(word)
    if current:
        ready = cut(False)
        if ready:
            yield ready
    if pending is not None:
        yield pending


def progress_percent(position: float, duration: float) -> int:
    """Never reports 100 while still decoding; the caller decides when it is done."""
    if not duration or duration <= 0:
        return 0
    return int(max(0, min(99, position / duration * 100)))


def traditional_converter(language: str, enabled: bool = True):
    """Whisper writes Chinese in simplified characters. Returns a text → text function."""
    if not enabled or language != "zh":
        return lambda text: text
    try:
        from opencc import OpenCC
    except ImportError:
        return lambda text: text
    return OpenCC("s2twp").convert


def to_plottr_json(result: dict) -> dict:
    """The single-JSON shape plottr's python/transcribe.py has always printed."""
    text = " ".join(segment["text"] for segment in result["segments"])
    return {
        "language": result["language"],
        "text": text,
        "segments": result["segments"],
        "word_count": len(text.split()) if text else 0,
        "device": result["device"],
        "model": result["model"],
    }


# ── transcription ──────────────────────────────────────────────────────────────────────────

def detect_language(path: str, device: str = "cpu") -> tuple:
    """(language, probability) from the first 30 seconds. Cheap: nothing is decoded."""
    model = load_model(DETECT_MODEL, device, "int8")
    generator, info = model.transcribe(path, language=None, vad_filter=False, condition_on_previous_text=False)
    del generator  # detection only — never consume it
    return info.language, info.language_probability


def transcribe(
    path: str,
    *,
    model: str = "",
    device: str = "auto",
    compute_type: str = "",
    language: str = "",
    guard_language: bool = False,
    guard_threshold: float = 0.5,
    max_words: int = 0,
    vad: bool = True,
    vad_parameters: dict = None,
    traditional_chinese: bool = True,
    include_words: bool = False,
    cues: bool = False,
    cue_options: dict = None,
    decode_options: dict = None,
    on_progress=None,
    on_segment=None,
) -> dict:
    """Transcribe an audio or video file.

    language        pin the language ("en"); empty = detect
    guard_language  with a pinned language: refuse (UnexpectedLanguageError) when the audio is
                    confidently something else. Pinning "en" on non-English audio makes Whisper
                    invent fluent fake English, so services that pin should guard.
    max_words       > 0: return sentence-sized segments (needs word timestamps, a little slower)
    include_words   attach each segment's word timings ("words": [{word, start, end, probability}])
                    for callers that do their own splitting; segments are then left whole
    cues            subtitle-sized segments cut at real pauses (see build_cues); cue_options
                    overrides its limits. Use for burned-in subtitles; max_words is for reading.
    decode_options  extra arguments passed straight to faster-whisper (beam_size, ...)
    on_segment      called with each output segment as soon as it exists (if a device fails
                    mid-run, the next device starts over and segments are reported again)
    on_progress     called with an int 0-99 as decoding advances
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    expose_nvidia_libraries()

    if language and guard_language:
        try:
            detected, probability = detect_language(path)
        except Exception as error:  # a broken guard must not block real work
            print(f"[whisper-kit] language detection failed, continuing: {error}", file=sys.stderr)
        else:
            if detected != language and probability >= guard_threshold:
                raise UnexpectedLanguageError(language, detected, probability)

    last_error = None
    for dev, compute, model_name in plan_attempts(model, device, compute_type):
        try:
            return _run(path, dev, compute, model_name, language=language, max_words=max_words, vad=vad,
                        vad_parameters=vad_parameters, traditional_chinese=traditional_chinese,
                        on_progress=on_progress, include_words=include_words, cues=cues,
                        cue_options=cue_options, decode_options=decode_options, on_segment=on_segment)
        except Exception as error:
            # A GPU can fail at load or only once decoding starts (missing CUDA library, out of
            # memory), so the whole attempt is retried on the next device, not just the load.
            last_error = error
            forget_model(model_name, dev, compute)
            print(f"[whisper-kit] {dev} failed, trying the next option: {error}", file=sys.stderr)
    raise RuntimeError(f"transcription failed on every device: {last_error}")


def _word_dicts(words) -> list:
    return [{"word": w.word, "start": _seconds(w.start), "end": _seconds(w.end),
             "probability": round(float(getattr(w, "probability", 1.0)), 3)} for w in (words or [])]


def _seconds(value) -> float:
    """Plain float, millisecond precision (word timings arrive as numpy floats)."""
    return round(float(value), 3)


def _pieces(segments_iter, max_words: int, include_words: bool, cues: bool, cue_options):
    """Whisper's segments → the output segments, lazily, in whichever shape was asked for."""
    if cues:
        for start, end, text in build_cues(segments_iter, **{**CUE_DEFAULTS, **(cue_options or {})}):
            yield {"start": _seconds(start), "end": _seconds(end), "text": text}
        return
    for segment in segments_iter:
        whole = {"start": _seconds(segment.start), "end": _seconds(segment.end), "text": segment.text.strip()}
        if include_words:
            yield {**whole, "words": _word_dicts(segment.words)}
        elif max_words > 0 and segment.words:
            yield from split_words([(w.start, w.end, w.word) for w in segment.words], max_words)
        else:
            yield whole


def _run(path, device, compute, model_name, *, language, max_words, vad, vad_parameters,
         traditional_chinese, on_progress, include_words, cues=False, cue_options=None,
         decode_options=None, on_segment=None) -> dict:
    model = load_model(model_name, device, compute)
    options = {"language": language or None, "vad_filter": vad,
               "word_timestamps": max_words > 0 or include_words or cues}
    if vad and vad_parameters:
        options["vad_parameters"] = vad_parameters
    options.update(decode_options or {})
    segments_iter, info = model.transcribe(path, **options)
    convert = traditional_converter(info.language, traditional_chinese)

    segments = []
    last_percent = -1
    for piece in _pieces(segments_iter, max_words, include_words, cues, cue_options):  # decoding happens here
        if not piece["text"]:
            continue
        piece = {**piece, "text": convert(piece["text"])}
        segments.append(piece)
        if on_segment:
            on_segment(piece)
        percent = progress_percent(piece["end"], info.duration)
        if on_progress and percent != last_percent:
            on_progress(percent)
            last_percent = percent

    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "duration": round(float(info.duration or 0), 2),
        "device": device,
        "model": model_name,
        "segments": segments,
    }


# ── command line ───────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio/video locally with faster-whisper")
    parser.add_argument("path", help="audio or video file")
    parser.add_argument("--model", default="", help=f"model (default: {GPU_MODEL} on GPU, {CPU_MODEL} on CPU)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-type", default="", help="default: float16 on GPU, int8 on CPU")
    parser.add_argument("--language", default="", help="pin a language code; empty = detect")
    parser.add_argument("--guard-language", action="store_true", help="refuse audio that is not --language")
    parser.add_argument("--max-words", type=int, default=0, help="split into sentences of at most N words")
    parser.add_argument("--cues", action="store_true", help="subtitle-sized segments cut at real pauses")
    parser.add_argument("--beam-size", type=int, default=0, help="faster-whisper beam size (default: its own)")
    parser.add_argument("--no-vad", action="store_true", help="do not skip silence")
    parser.add_argument("--keep-simplified", action="store_true", help="do not convert Chinese to traditional")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of the line protocol")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    def emit(prefix: str, payload: str) -> None:
        sys.stdout.write(f"{prefix} {payload}\n")
        sys.stdout.flush()

    try:
        result = transcribe(
            args.path,
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            guard_language=args.guard_language,
            max_words=args.max_words,
            cues=args.cues,
            decode_options={"beam_size": args.beam_size} if args.beam_size > 0 else None,
            vad=not args.no_vad,
            traditional_chinese=not args.keep_simplified,
            on_progress=None if args.json else (lambda percent: emit("P", str(percent))),
        )
    except UnexpectedLanguageError as error:
        refusal = {"code": error.code, "expected": error.expected, "language": error.language,
                   "probability": round(error.probability, 3), "message": str(error)}
        if args.json:
            print(f"Transcription refused: {error}", file=sys.stderr)
        else:
            emit("E", json.dumps(refusal, ensure_ascii=False))
        return EXIT_REFUSED
    except Exception as error:
        print(f"Transcription failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(to_plottr_json(result), sys.stdout, ensure_ascii=False)
    else:
        emit("R", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

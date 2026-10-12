"""ASR stage: faster-whisper with word-level timestamps.

Runs in its own process (see `mmpquant.stages`) because faster-whisper pulls in
PyAV and RapidOCR pulls in OpenCV, and on macOS both ship a copy of
`libavdevice` — loading them together makes the ObjC runtime warn about
duplicate class registration. Process isolation removes that failure mode and
frees the model's memory before OCR starts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..types import AsrSegment, AsrWord, Transcript

log = logging.getLogger(__name__)


class AsrError(RuntimeError):
    pass


def transcribe(
    audio_path: Path,
    *,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    vad_filter: bool = True,
    language: str | None = None,
    word_timestamps: bool = True,
    duration_hint: float | None = None,
) -> Transcript:
    """Transcribe `audio_path` and return a `Transcript`.

    An empty transcript is a legitimate result (silent video), not an error.
    The caller decides what that means for scoring — here it just means
    `segments == ()`.

    `vad_filter=True` matters more than it looks: Whisper hallucinates fluent
    text over silence and music, and those hallucinations would show up as
    narration in windows that have none, inflating redundancy. VAD gating is the
    cheapest defence.
    """
    from faster_whisper import WhisperModel  # heavy; imported inside the stage

    if not audio_path.exists():
        raise AsrError(f"audio not found: {audio_path}")

    log.info("loading whisper model=%s device=%s compute_type=%s",
             model_name, device, compute_type)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    # When a decoding language is set, the language is still DETECTED first and
    # that detection is what the transcript records -- so a genuinely
    # non-English video still raises the language warning. Only the decoding
    # is fixed. Added 05 Oct 2026: on Sir Ken Robinson's English lecture
    # (Test_24-27) auto-detection chose Welsh at p=0.70-0.84 and Whisper then
    # wrote 14-27% of each transcript in Welsh ("cwelwch", "dweud").
    detected: tuple[str | None, float | None] = (None, None)
    if language is not None:
        from faster_whisper.audio import decode_audio

        lang, prob, _ = model.detect_language(decode_audio(str(audio_path)), vad_filter=vad_filter)
        detected = (lang, float(prob))

    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        vad_filter=vad_filter,
        language=language,
        word_timestamps=word_timestamps,
    )

    segments: list[AsrSegment] = []
    for seg in segments_iter:  # generator — transcription happens here
        words: tuple[AsrWord, ...] = ()
        if word_timestamps and getattr(seg, "words", None):
            words = tuple(
                AsrWord(start=float(w.start), end=float(w.end), text=w.word.strip())
                for w in seg.words
                if w.start is not None and w.end is not None and w.word.strip()
            )
        else:
            # No word timings: fall back to the segment as one pseudo-word so
            # alignment still has something to place. Recorded as a warning by
            # the caller because window assignment gets coarser.
            text = seg.text.strip()
            if text:
                words = (AsrWord(start=float(seg.start), end=float(seg.end), text=text),)

        segments.append(
            AsrSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                words=words,
            )
        )

    if language is not None:
        detected_language, detected_probability = detected
    else:
        detected_language = getattr(info, "language", None)
        detected_probability = (
            float(info.language_probability)
            if getattr(info, "language_probability", None) is not None
            else None
        )
    return Transcript(
        segments=tuple(segments),
        language=detected_language,
        language_probability=detected_probability,
        duration=float(
            duration_hint
            if duration_hint is not None
            else getattr(info, "duration", 0.0) or 0.0
        ),
    )


def language_warnings(
    transcript: Transcript,
    expected: str | None,
    min_probability: float,
    decoded_as: str | None = None,
) -> list[str]:
    """Non-fatal checks on the detected language.

    Non-English audio is a documented failure mode: the sentence-embedding model
    is English-centric and ROUGE-L over a mixed-script token stream is not
    meaningful. We warn rather than abort so the run still produces inspectable
    artifacts.
    """
    warnings: list[str] = []
    if transcript.is_silent:
        return ["no speech detected — redundancy is N/A for this video"]
    if decoded_as and transcript.language and transcript.language != decoded_as:
        warnings.append(
            f"language detection said '{transcript.language}' "
            f"(p={transcript.language_probability:.2f}) but the audio was transcribed "
            f"as '{decoded_as}' (asr.language). Right for an accent the detector "
            "misjudges; wrong if the speaker really is not speaking that language -- "
            "check the transcript."
        )
    elif expected and transcript.language and transcript.language != expected:
        warnings.append(
            f"detected language '{transcript.language}' != expected '{expected}'. "
            "Semantic overlap uses an English-centric embedding model, so the "
            "semantic component of this score is not trustworthy."
        )
    if (
        transcript.language_probability is not None
        and transcript.language_probability < min_probability
    ):
        warnings.append(
            f"low language-detection confidence "
            f"({transcript.language_probability:.2f} < {min_probability}) — "
            "audio may be multilingual, noisy, or mostly music"
        )
    return warnings

"""Where the lesson itself starts and ends: the intro and outro are not scored.

User's ruling, 13 Nov 2026 (replaces the fixed 10-second coherence intro of
12 Nov): an editor's or channel's opening and closing are not part of the
teaching, so nothing in them counts -- neither for coherence (music, logos,
pictures) nor for redundancy (a closing caption read aloud). Measured per
video, with no tuned number:

  intro = from the start of the video to the narrator's first word
  outro = from the start of the closing sign-off to the end of the video,
          where the sign-off is the run of FINAL sentences that only say
          goodbye or promote something (subscribe, like, link in the
          description, visit / check out our site, Patreon, enrol today,
          see you next time, thanks for watching). It stops at the first
          sentence (counting back from the end) that does not. With no
          sign-off, the outro starts after the last spoken word.

Known limit, accepted by the user: a spoken greeting at the start ("Hi, I'm
Mike, welcome to ...") is not recognised as intro -- the user wants the
narrated course title of GDPR 0:00-0:10 scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .types import AsrSegment, Transcript

# A sentence that only signs off or promotes. Matched anywhere in the sentence,
# but only the trailing run of sentences is ever tested, so a lesson that says
# "see you" mid-way is unaffected.
_CLOSING = re.compile(
    r"\b(?:subscrib\w*|hit the (?:like|bell)|like (?:and|&) (?:share|subscribe)"
    r"|link (?:is )?in the description|click (?:on )?the link|description below"
    r"|check out (?:our|my|the)\b|visit (?:our|my|us)\b|patreon|our (?:website|blog|channel)"
    r"|follow (?:us|me)\b|thanks? (?:you )?(?:so much )?for watching"
    r"|see you (?:guys )?(?:next time|in the next|inside|soon|there|later)"
    r"|until next time|enrol+ (?:today|now)|sign up (?:today|now)|register (?:today|now))",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?]['\")\]]*$")


@dataclass(frozen=True)
class LessonSpan:
    start: float                 # first spoken word
    end: float                   # start of the sign-off, else end of the last word
    duration: float
    closing: tuple[str, ...] = field(default_factory=tuple)  # sign-off sentences

    @property
    def intro_s(self) -> float:
        return round(self.start, 3)

    @property
    def outro_s(self) -> float:
        return round(max(0.0, self.duration - self.end), 3)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "intro_s": self.intro_s, "outro_s": self.outro_s,
                "closing_sentences": list(self.closing),
                "rule": "intro = before the first word; outro = closing sign-off "
                        "(or after the last word); neither is scored (13 Nov 2026)"}


def _sentences(transcript: Transcript) -> list[tuple[float, float, str]]:
    """The narration as sentences (start, end, text), split at . ! ? on words."""
    out, cur = [], []
    for w in transcript.words:
        cur.append(w)
        if _SENTENCE_END.search(w.text.strip()):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [(s[0].start, s[-1].end, " ".join(x.text.strip() for x in s)) for s in out]


def lesson_span(transcript: Transcript, duration: float) -> LessonSpan | None:
    """The lesson's span, or None when nobody speaks (the video is N/A anyway)."""
    sents = _sentences(transcript)
    if not sents:
        return None
    closing = []
    for s in reversed(sents):
        if not _CLOSING.search(s[2]):
            break
        closing.append(s)
    if len(closing) == len(sents):   # a video that is all sign-off: score it whole
        closing = []
    end = closing[-1][0] if closing else sents[-1][1]
    return LessonSpan(start=sents[0][0], end=end, duration=duration,
                      closing=tuple(s[2] for s in reversed(closing)))


def within(transcript: Transcript, span: LessonSpan | None) -> Transcript:
    """The transcript with only the words spoken inside the lesson span."""
    if span is None:
        return transcript
    segs = []
    for s in transcript.segments:
        words = tuple(w for w in s.words if span.start <= w.start < span.end)
        if words:
            segs.append(AsrSegment(words[0].start, words[-1].end,
                                   " ".join(w.text.strip() for w in words), words))
    return Transcript(tuple(segs), transcript.language, transcript.language_probability,
                      transcript.duration)

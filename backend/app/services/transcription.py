"""Multilingual, timestamp-preserving faster-whisper transcription."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any

from app.config import SUPPORTED_LANGUAGES, get_settings
from app.services.language_detection import LanguageDetection, detect_language, detect_segment_languages
from app.utils.audio import get_audio_duration

logger = logging.getLogger(__name__)
_whisper_model: Any | None = None
_whisper_model_name: str | None = None
_whisper_error: str | None = None
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")
_KANNADA_REPEATED_RE = re.compile(r"([\u0C80-\u0CFF]{1,3})\1{2,}")
_REPEATED_PHRASE_RE = re.compile(r"(\b\S+\b)(?:\s+\1){2,}")
LANGUAGE_PROMPTS = {
    "te": "నమస్కారం, ఇది తెలుగు సంభాషణ.",
    "kn": "ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡ ಸಂಭಾಷಣೆ.",
    "hi": "नमस्ते, यह हिंदी बातचीत है।",
    "en": "Hello, this is an English conversation.",
}


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    language: str = "en"
    language_name: str = "English"
    words: list[TranscriptWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "language": self.language,
            "language_name": self.language_name,
            "text": self.text,
            "timestamp": _fmt_timestamp(self.start),
            "words": [word.to_dict() for word in self.words],
        }

    def with_speaker(self, speaker: str | None) -> "TranscriptSegment":
        return replace(self, speaker=speaker)

    @classmethod
    def from_words(
        cls,
        words: list[TranscriptWord],
        speaker: str | None,
        language: str,
    ) -> "TranscriptSegment":
        raw_joined = "".join(word.text for word in words)
        if " " in raw_joined:
            text = re.sub(r"\s+", " ", raw_joined).strip()
        else:
            text = " ".join(word.text.strip() for word in words if word.text.strip())
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        return cls(
            start=round(words[0].start, 3),
            end=round(words[-1].end, 3),
            text=_clean_unicode_text(text),
            speaker=speaker,
            language=language,
            language_name=lang_name,
            words=words,
        )

    def merged_with(self, other: "TranscriptSegment") -> "TranscriptSegment":
        combined_text = f"{self.text} {other.text}".strip()
        return TranscriptSegment(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            text=combined_text,
            speaker=self.speaker,
            language=self.language,
            language_name=self.language_name,
            words=[*self.words, *other.words],
        )


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_name: str
    language_probability: float
    language_detection: LanguageDetection
    segments: list[TranscriptSegment] = field(default_factory=list)
    detected_confident: bool = True


def _fmt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def whisper_status() -> str:
    try:
        import faster_whisper  # noqa: F401
        return "available"
    except Exception as exc:  # noqa: BLE001
        global _whisper_error
        _whisper_error = str(exc)
        return "unavailable"


def get_whisper_model() -> Any:
    global _whisper_model, _whisper_model_name, _whisper_error
    settings = get_settings()
    wanted = (settings.whisper_model or "medium").strip()
    if _whisper_model is not None and _whisper_model_name == wanted:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel

        device = settings.device.lower() if settings.device.lower() in {"cpu", "cuda"} else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info("[WHISPER] Loading model=%s device=%s compute_type=%s", wanted, device, compute_type)
        _whisper_model = WhisperModel(wanted, device=device, compute_type=compute_type)
        _whisper_model_name = wanted
        return _whisper_model
    except Exception as exc:  # noqa: BLE001
        _whisper_error = str(exc)
        logger.exception("[WHISPER] Model load failure")
        raise RuntimeError(f"Speech-to-text model '{wanted}' could not be loaded: {exc}") from exc


def _clean_unicode_text(text: str) -> str:
    """Sanitize transcript text while strictly preserving Indic and Latin Unicode characters."""
    text = text.replace("\ufffd", "").replace("\u200b", "").replace("\ufeff", "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = _REPEATED_CHAR_RE.sub(r"\1\1\1", text)
    text = _KANNADA_REPEATED_RE.sub(r"\1\1\1", text)
    text = _REPEATED_PHRASE_RE.sub(r"\1", text)
    return text.strip()


_DEVANAGARI_TO_TELUGU_EXPLICIT: dict[str, str] = {
    "\u0901": "\u0C01",  # candrabindu
    "\u0902": "\u0C02",  # anusvara
    "\u0903": "\u0C03",  # visarga
    "\u094D": "\u0C4D",  # virama / halant
    "\u093C": "",        # nukta
    "\u0950": "\u0C50",  # om
    "\u0944": "\u0C44",  # vocalic rr matra
    "\u0943": "\u0C43",  # vocalic r matra
    "\u090B": "\u0C0B",  # vocalic r
    "\u0960": "\u0C60",  # vocalic rr
    "\u090C": "\u0C0C",  # vocalic l
    "\u0961": "\u0C61",  # vocalic ll
    "\u0949": "\u0C4A",  # candra o -> short o
    "\u0945": "\u0C46",  # candra e -> short e
    "\u0911": "\u0C12",  # candra o -> short o
    "\u090F": "\u0C0F",  # e
    "\u0947": "\u0C47",  # e matra
    "\u0910": "\u0C10",  # ai
    "\u0948": "\u0C48",  # ai matra
    "\u0913": "\u0C13",  # o
    "\u094B": "\u0C4B",  # o matra
    "\u0914": "\u0C14",  # au
    "\u094C": "\u0C4C",  # au matra
    "\u0964": ".",       # danda -> period
    "\u0965": ".",       # double danda -> period
}


def devanagari_to_telugu(text: str) -> str:
    """Accurately convert Devanagari script representation of Telugu speech into native Telugu script."""
    import unicodedata

    res: list[str] = []
    for ch in text:
        if ch in _DEVANAGARI_TO_TELUGU_EXPLICIT:
            res.append(_DEVANAGARI_TO_TELUGU_EXPLICIT[ch])
        elif 0x0900 <= ord(ch) <= 0x097F:
            tel_ch = chr(ord(ch) + 0x0300)
            if unicodedata.name(tel_ch, None):
                res.append(tel_ch)
            else:
                res.append(ch)
        else:
            res.append(ch)
    return "".join(res)


def _extract_words(raw_segment: Any, is_telugu: bool = False) -> list[TranscriptWord]:
    result: list[TranscriptWord] = []
    raw_words = getattr(raw_segment, "words", None) or []
    for word in raw_words:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        text = getattr(word, "word", "")
        if start is not None and end is not None and float(end) > float(start):
            cleaned = _clean_unicode_text(str(text))
            if is_telugu and any(0x0900 <= ord(c) <= 0x097F for c in cleaned):
                cleaned = devanagari_to_telugu(cleaned)
            if cleaned:
                result.append(TranscriptWord(start=float(start), end=float(end), text=cleaned))
    return result


def resolve_segment_language(text: str, default_lang: str = "en") -> tuple[str, str]:
    """Resolve language code and display name based on Unicode script."""
    kn = sum(1 for ch in text if 0x0C80 <= ord(ch) <= 0x0CFF)
    te = sum(1 for ch in text if 0x0C00 <= ord(ch) <= 0x0C7F)
    hi = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
    en = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))

    scores = {"kn": kn, "te": te, "hi": hi}
    best_indic = max(scores, key=scores.get)
    if scores[best_indic] > 0:
        code = best_indic
    elif en > 0:
        code = "en"
    else:
        code = default_lang if default_lang in SUPPORTED_LANGUAGES else "en"
    return code, SUPPORTED_LANGUAGES.get(code, "English")


def transcribe_audio(audio_path: str) -> TranscriptionResult:
    """Transcribe audio with accurate multi-language support, preserving native script and timestamps.

    Task is always 'transcribe' (never translation).
    """
    model = get_whisper_model()
    audio_duration = get_audio_duration(audio_path)

    # Detect language using multi-region audio sampling
    detection = detect_language(model, audio_path)
    # If single language is confident and not code-switched, pin target language; otherwise multilingual auto-detect
    lang_to_use = detection.code if (detection.confidence >= 0.70 and not detection.code_switched) else None
    is_telugu_audio = (detection.code == "te")

    try:
        segments_iter, info = model.transcribe(
            audio_path,
            language=lang_to_use,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            without_timestamps=False,
            vad_filter=True,
            compression_ratio_threshold=2.2,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.60,
            repetition_penalty=1.2,
            initial_prompt=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[TRANSCRIPTION] faster-whisper transcribe failed")
        raise RuntimeError("Transcription failed. The audio file may be corrupt or unreadable.") from exc

    default_lang = detection.code if detection.code in SUPPORTED_LANGUAGES else "en"
    segments: list[TranscriptSegment] = []

    for raw in segments_iter:
        start = max(0.0, float(raw.start or 0.0))
        end = min(audio_duration, float(raw.end or 0.0))
        if end - start < 0.15:
            continue

        text = _clean_unicode_text(raw.text or "")
        if not text:
            continue

        no_speech = float(getattr(raw, "no_speech_prob", 0.0) or 0.0)
        if no_speech > 0.85 and len(text) < 6:
            continue

        if is_telugu_audio and any(0x0900 <= ord(c) <= 0x097F for c in text):
            text = devanagari_to_telugu(text)

        words = _extract_words(raw, is_telugu=is_telugu_audio)
        if words:
            word_start = max(0.0, words[0].start)
            word_end = min(audio_duration, words[-1].end)
            if word_end > word_start:
                start = max(start, word_start)
                end = min(end, word_end)

        seg_code, seg_name = resolve_segment_language(text, default_lang)
        segments.append(
            TranscriptSegment(
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                speaker=None,
                language=seg_code,
                language_name=seg_name,
                words=words,
            )
        )

    if not segments:
        raise RuntimeError("Transcription produced an empty result. The audio may be silent or unclear.")

    lang_durations = {c: 0.0 for c in SUPPORTED_LANGUAGES}
    for seg in segments:
        lang_durations[seg.language] += seg.duration

    total_dur = sum(lang_durations.values()) or 1.0
    primary_code = max(lang_durations, key=lang_durations.get)
    primary_conf = lang_durations[primary_code] / total_dur
    active_langs = [c for c, d in lang_durations.items() if (d / total_dur) >= 0.15]
    is_code_switched = len(active_langs) > 1

    detection = LanguageDetection(
        code=primary_code,
        name=SUPPORTED_LANGUAGES[primary_code],
        confidence=round(primary_conf, 3),
        alternatives={c: round(d / total_dur, 3) for c, d in lang_durations.items()},
        sampled_regions=len(segments),
    )

    full_text = " ".join(seg.text for seg in segments)

    logger.info(
        "[TRANSCRIPTION]\n"
        "  Whisper segments: %d\n"
        "  Primary language: %s (confidence: %.3f)\n"
        "  Code switched: %s\n"
        "  Word timestamps: enabled\n"
        "  Total text chars: %d",
        len(segments),
        primary_code,
        primary_conf,
        is_code_switched,
        len(full_text),
    )

    return TranscriptionResult(
        text=full_text,
        language=primary_code,
        language_name=SUPPORTED_LANGUAGES[primary_code],
        language_probability=round(primary_conf, 3),
        language_detection=detection,
        segments=segments,
        detected_confident=primary_conf >= 0.50,
    )


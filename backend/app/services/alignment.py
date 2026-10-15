"""Temporal alignment between Whisper words and voice-diarization turns."""

from __future__ import annotations

import logging
from app.services.diarization import SpeakerTurn
from app.services.transcription import TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)


def _speaker_for_interval(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Calculate maximum temporal overlap with diarization speaker turns, with proximity fallback."""
    if not turns or end <= start:
        return None

    best_overlap = 0.0
    best_speaker: str | None = None

    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

    # Require at least minimal overlap
    if best_overlap > 0.02:
        return best_speaker

    # Proximity fallback: match to the closest turn if within a reasonable gap
    mid = (start + end) / 2.0
    closest_dist = float("inf")
    closest_speaker: str | None = None

    for turn in turns:
        if turn.start <= mid <= turn.end:
            dist = 0.0
        elif mid < turn.start:
            dist = turn.start - mid
        else:
            dist = mid - turn.end

        if dist < closest_dist:
            closest_dist = dist
            closest_speaker = turn.speaker

    if closest_speaker is not None and closest_dist <= 1.2:
        return closest_speaker

    return None


def align_transcript(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """Align Whisper segments with diarization turns using word-level timestamps.

    If two speakers speak within a single Whisper segment, word timestamps split
    the segment at the exact speaker transition boundary.
    """
    if not turns:
        # Diarization unavailable or no turns found; preserve original segments with speaker=None
        return [seg.with_speaker(None) for seg in segments]

    unique_speakers = {turn.speaker for turn in turns if turn.speaker}
    if len(unique_speakers) <= 1:
        # Single speaker: preserve original ASR timestamp segments without artificial speaker chunking
        speaker = next(iter(unique_speakers)) if unique_speakers else None
        return [seg.with_speaker(speaker) for seg in segments]

    aligned: list[TranscriptSegment] = []

    for seg in segments:
        if not seg.words:
            # Segment has no individual word timestamps; match entire interval
            speaker = _speaker_for_interval(seg.start, seg.end, turns)
            aligned.append(seg.with_speaker(speaker))
            continue

        # Group words into contiguous same-speaker chunks
        current_words: list[TranscriptWord] = []
        current_speaker: str | None = None

        for word in seg.words:
            word_speaker = _speaker_for_interval(word.start, word.end, turns)
            if not current_words:
                current_words.append(word)
                current_speaker = word_speaker
            elif word_speaker == current_speaker or (word_speaker is None and current_speaker is not None):
                current_words.append(word)
            else:
                # Speaker switch inside Whisper segment! Emit current chunk
                aligned.append(
                    TranscriptSegment.from_words(
                        words=current_words,
                        speaker=current_speaker,
                        language=seg.language,
                    )
                )
                current_words = [word]
                current_speaker = word_speaker

        if current_words:
            aligned.append(
                TranscriptSegment.from_words(
                    words=current_words,
                    speaker=current_speaker,
                    language=seg.language,
                )
            )

    # Resolve any remaining unassigned segments when turns exist
    if turns:
        for i, seg in enumerate(aligned):
            if not seg.speaker:
                fallback_spk = _speaker_for_interval(seg.start, seg.end, turns)
                if fallback_spk:
                    aligned[i] = seg.with_speaker(fallback_spk)
                elif i > 0 and aligned[i - 1].speaker:
                    aligned[i] = seg.with_speaker(aligned[i - 1].speaker)
                elif i + 1 < len(aligned) and aligned[i + 1].speaker:
                    aligned[i] = seg.with_speaker(aligned[i + 1].speaker)

    # Merge adjacent segments belonging to the same speaker and same language
    merged: list[TranscriptSegment] = []
    for seg in aligned:
        if (
            merged
            and merged[-1].speaker == seg.speaker
            and merged[-1].language == seg.language
            and seg.start - merged[-1].end <= 0.30
        ):
            merged[-1] = merged[-1].merged_with(seg)
        else:
            merged.append(seg)

    logger.info(
        "[ALIGNMENT]\n"
        "  Whisper source segments: %d\n"
        "  Diarization turns: %d\n"
        "  Final aligned segments: %d",
        len(segments),
        len(turns),
        len(merged),
    )

    return merged

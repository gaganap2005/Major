"""Verification test suite for Kannada and regression testing across all languages."""

from __future__ import annotations

import os
from pathlib import Path
from app.config import SUPPORTED_LANGUAGES
from app.services.alignment import align_transcript
from app.services.diarization import SpeakerTurn
from app.services.jobs import format_transcript_text
from app.services.language_detection import LanguageDetection
from app.services.summarization import summarize_transcript, _looks_like_language
from app.services.transcription import (
    TranscriptSegment,
    TranscriptWord,
    _clean_unicode_text,
)
from app.services.tts import synthesize_speech, language_supported_for_tts
from app.services.validation import validate_results


def run_all_checks() -> dict[str, bool]:
    results = {}

    # 1. Kannada single-speaker audio formatting & timestamps
    kn_words = [TranscriptWord(0.0, 5.42, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ ನಮ್ಮ ಪಾಡ್‌ಕ್ಯಾಸ್ಟ್‌ಗೆ ಸ್ವಾಗತ.")]
    kn_seg = TranscriptSegment(
        start=0.0,
        end=5.42,
        text="ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ ನಮ್ಮ ಪಾಡ್‌ಕ್ಯಾಸ್ಟ್‌ಗೆ ಸ್ವಾಗತ.",
        language="kn",
        language_name="Kannada",
        words=kn_words,
    )
    kn_turns = [SpeakerTurn(0.0, 5.42, "SPEAKER_00")]
    aligned_1 = align_transcript([kn_seg], kn_turns)
    formatted_1 = format_transcript_text(aligned_1, detected_speakers=1)
    results["1_kannada_single_speaker"] = (
        "00:00.000 → 00:05.420 | Kannada" in formatted_1
        and "SPEAKER_00" not in formatted_1
        and "ನಮಸ್ಕಾರ" in formatted_1
    )

    # 2. Kannada multi-speaker audio
    kn_seg2 = TranscriptSegment(
        start=5.42,
        end=10.5,
        text="ಧನ್ಯವಾದಗಳು, ಈ ವಿಷಯದ ಬಗ್ಗೆ ಚರ್ಚಿಸೋಣ.",
        language="kn",
        language_name="Kannada",
        words=[TranscriptWord(5.42, 10.5, "ಧನ್ಯವಾದಗಳು, ಈ ವಿಷಯದ ಬಗ್ಗೆ ಚರ್ಚಿಸೋಣ.")],
    )
    kn_multi_turns = [SpeakerTurn(0.0, 5.42, "SPEAKER_00"), SpeakerTurn(5.42, 10.5, "SPEAKER_01")]
    aligned_2 = align_transcript([kn_seg, kn_seg2], kn_multi_turns)
    formatted_2 = format_transcript_text(aligned_2, detected_speakers=2)
    results["2_kannada_multi_speaker"] = (
        "SPEAKER_00 | 00:00.000 → 00:05.420 | Kannada" in formatted_2
        and "SPEAKER_01 | 00:05.420 → 00:10.500 | Kannada" in formatted_2
    )

    # 3. Kannada + English code-switching
    en_seg = TranscriptSegment(
        start=4.2,
        end=8.0,
        text="Welcome to our multilingual session today.",
        language="en",
        language_name="English",
        words=[TranscriptWord(4.2, 8.0, "Welcome to our multilingual session today.")],
    )
    aligned_3 = align_transcript([kn_seg, en_seg], [SpeakerTurn(0.0, 4.0, "SPEAKER_00"), SpeakerTurn(4.2, 8.0, "SPEAKER_01")])
    formatted_3 = format_transcript_text(aligned_3, detected_speakers=2)
    results["3_kannada_english_code_switch"] = (
        aligned_3[0].language == "kn"
        and aligned_3[1].language == "en"
        and "Kannada" in formatted_3
        and "English" in formatted_3
    )

    # 4. Kannada + Hindi/Telugu code-switching
    hi_seg = TranscriptSegment(
        start=5.5,
        end=10.0,
        text="नमस्ते दोस्तों, आज की चर्चा में आपका स्वागत है।",
        language="hi",
        language_name="Hindi",
    )
    te_seg = TranscriptSegment(
        start=10.2,
        end=15.0,
        text="నమస్కారం అందరికీ, ఇది చాలా ముఖ్యమైన అంశం.",
        language="te",
        language_name="Telugu",
    )
    formatted_4 = format_transcript_text([kn_seg, hi_seg, te_seg], detected_speakers=1)
    results["4_kannada_hindi_telugu_code_switch"] = (
        "00:00.000 → 00:05.420 | Kannada" in formatted_4
        and "00:05.500 → 00:10.000 | Hindi" in formatted_4
        and "00:10.200 → 00:15.000 | Telugu" in formatted_4
    )

    # 5. Kannada summarization
    kn_text = "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ, ಇಂದು ನಾವು ಶಿಕ್ಷಣ ಮತ್ತು ತಂತ್ರಜ್ಞಾನದ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೇವೆ. ಸ್ಥಳೀಯ ಭಾಷೆಗಳಲ್ಲಿ ವಿಷಯವನ್ನು ಒದಗಿಸುವುದು ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಬಹಳ ಪ್ರಯೋಜನಕಾರಿಯಾಗಿದೆ. ಡಿಜಿಟಲ್ ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬರಿಗೂ ತಲುಪಬೇಕು."
    summary, method, warning, lang, err = summarize_transcript(kn_text, output_language="kn", source_language="kn")
    results["5_kannada_summarization"] = (
        lang == "kn"
        and len(summary) > 20
        and _looks_like_language(summary, "kn")
    )

    # 6. Kannada TTS / audio summary
    out_mp3 = Path("storage/temp/verify_kannada.mp3")
    tts_ok, tts_method, tts_err = synthesize_speech(summary[:200], "kn", out_mp3)
    results["6_kannada_tts"] = (
        tts_ok
        and out_mp3.exists()
        and out_mp3.stat().st_size > 0
    )

    # 7. Existing English processing
    en_text = "Today we discuss technology education across Indian universities and regional learning outcomes."
    en_summary, en_m, en_w, en_lang, en_err = summarize_transcript(en_text, output_language="en", source_language="en")
    en_mp3 = Path("storage/temp/verify_en.mp3")
    en_tts_ok, _, _ = synthesize_speech(en_summary[:200], "en", en_mp3)
    results["7_english_regression"] = (
        en_lang == "en"
        and _looks_like_language(en_summary, "en")
        and en_tts_ok
        and en_mp3.exists()
    )

    # 8. Existing Hindi processing
    hi_text = "नमस्ते, आज हम डिजिटल इंडिया और शिक्षा पर चर्चा कर रहे हैं। भारतीय भाषाओं में पॉडकास्ट छात्रों के लिए बहुत उपयोगी हैं।"
    hi_summary, hi_m, hi_w, hi_lang, hi_err = summarize_transcript(hi_text, output_language="hi", source_language="hi")
    hi_mp3 = Path("storage/temp/verify_hi.mp3")
    hi_tts_ok, _, _ = synthesize_speech(hi_summary[:200], "hi", hi_mp3)
    results["8_hindi_regression"] = (
        hi_lang == "hi"
        and _looks_like_language(hi_summary, "hi")
        and hi_tts_ok
        and hi_mp3.exists()
    )

    # 9. Existing Telugu processing
    te_text = "నమస్కారం అందరికీ, ఈ రోజు మన ముఖ్యమైన అంశం సాంకేతిక పరిజ్ఞానం. స్థానిక భాషల్లో విద్యార్థులకు జ్ఞానాన్ని అందించడం ఎంతో ముఖ్యం."
    te_summary, te_m, te_w, te_lang, te_err = summarize_transcript(te_text, output_language="te", source_language="te")
    te_mp3 = Path("storage/temp/verify_te.mp3")
    te_tts_ok, _, _ = synthesize_speech(te_summary[:200], "te", te_mp3)
    results["9_telugu_regression"] = (
        te_lang == "te"
        and _looks_like_language(te_summary, "te")
        and te_tts_ok
        and te_mp3.exists()
    )

    # 10. Existing single/multi-speaker tests
    val_report = validate_results(
        audio_duration=10.5,
        detected_language="kn",
        language_confidence=0.88,
        segments=aligned_2,
        diarization_available=True,
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    results["10_single_multi_speaker_validation"] = (
        val_report.passed
        and val_report.details["timestamp_validation"] == "PASS"
        and "PASS" in val_report.details["speaker_validation"]
        and val_report.details["language_validation"] == "PASS"
    )

    return results


if __name__ == "__main__":
    results = run_all_checks()
    for test_name, status in sorted(results.items()):
        print(f"{test_name}: {'PASSED' if status else 'FAILED'}")
    all_ok = all(results.values())
    print("ALL_CHECKS_PASSED" if all_ok else "SOME_CHECKS_FAILED")

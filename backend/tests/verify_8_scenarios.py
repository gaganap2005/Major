"""Systematic verification of all 8 multilingual test requirements:
1. Pure Kannada
2. Pure Telugu
3. Pure Hindi
4. Pure English
5. Kannada + English
6. Telugu + English
7. Kannada + Telugu
8. Kannada + Telugu + English

Verifies BOTH:
- transcript script/content
- language label
"""

import sys
from app.config import SUPPORTED_LANGUAGES
from app.services.transcription import (
    TranscriptSegment,
    TranscriptWord,
    resolve_segment_language,
    transcribe_audio,
)
from app.services.jobs import format_transcript_text
from app.services.alignment import align_transcript
from app.services.diarization import SpeakerTurn

def check(name: str, condition: bool, details: str = ""):
    status = "PASSED" if condition else "FAILED"
    print(f"[{status}] {name} {details}")
    if not condition:
        raise AssertionError(f"Test failed: {name} - {details}")

print("=================================================================")
print("RUNNING SYSTEMATIC 8-SCENARIO VERIFICATION")
print("=================================================================\n")

# 1. PURE KANNADA
kn_text = "ಯೋಗರಾಜ್ ಭಟ್ ಅವರು ಎಂಟುನೂರಕ್ಕೂ ಹೆಚ್ಚು ಹಾಡುಗಳಿಗೆ ಸಾಹಿತ್ಯವನ್ನು ನೀಡಿದ್ದಾರೆ."
kn_code, kn_name = resolve_segment_language(kn_text)
check("1. Pure Kannada Segment Script & Label", kn_code == "kn" and kn_name == "Kannada" and any(0x0C80 <= ord(c) <= 0x0CFF for c in kn_text), f"code={kn_code}, script=Kannada")

# 2. PURE TELUGU
te_text = "నేటి ఆధునిక ప్రపంచంలో టెక్నాలజీ వేగంగా విస్తరిస్తోంది."
te_code, te_name = resolve_segment_language(te_text)
check("2. Pure Telugu Segment Script & Label", te_code == "te" and te_name == "Telugu" and any(0x0C00 <= ord(c) <= 0x0C7F for c in te_text), f"code={te_code}, script=Telugu")

# 3. PURE HINDI
hi_text = "डिजिटल इंडिया के इस दौर में शिक्षा का स्वरूप तेजी से बदल रहा है।"
hi_code, hi_name = resolve_segment_language(hi_text)
check("3. Pure Hindi Segment Script & Label", hi_code == "hi" and hi_name == "Hindi" and any(0x0900 <= ord(c) <= 0x097F for c in hi_text), f"code={hi_code}, script=Devanagari")

# 4. PURE ENGLISH
en_text = "The podcast explores the evolving landscape of technology education."
en_code, en_name = resolve_segment_language(en_text)
check("4. Pure English Segment Script & Label", en_code == "en" and en_name == "English" and not any(ord(c) > 127 for c in en_text), f"code={en_code}, script=Latin")

# 5. KANNADA + ENGLISH (code-switched)
seg_kn = TranscriptSegment(0.0, 5.0, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ, ನಮ್ಮ ಪಾಡ್‌ಕ್ಯಾಸ್ಟ್‌ಗೆ ಸ್ವಾಗತ.", language="kn", language_name="Kannada")
seg_en = TranscriptSegment(5.2, 10.0, "Welcome to our special tech edition today.", language="en", language_name="English")
turns_5 = [SpeakerTurn(0.0, 5.0, "SPEAKER_00"), SpeakerTurn(5.2, 10.0, "SPEAKER_01")]
aligned_5 = align_transcript([seg_kn, seg_en], turns_5)
fmt_5 = format_transcript_text(aligned_5, detected_speakers=2)
check("5. Kannada + English Script & Label Preservation", 
      aligned_5[0].language == "kn" and "Kannada" in fmt_5 and "ನಮಸ್ಕಾರ" in aligned_5[0].text and
      aligned_5[1].language == "en" and "English" in fmt_5 and "Welcome" in aligned_5[1].text)

# 6. TELUGU + ENGLISH (code-switched)
seg_te = TranscriptSegment(0.0, 4.5, "నమస్కారం అందరికీ, ఈ రోజు మన అంశం ఆర్టిఫిషియల్ ఇంటెలిజెన్స్.", language="te", language_name="Telugu")
turns_6 = [SpeakerTurn(0.0, 4.5, "SPEAKER_00"), SpeakerTurn(4.8, 9.0, "SPEAKER_01")]
aligned_6 = align_transcript([seg_te, seg_en], turns_6)
fmt_6 = format_transcript_text(aligned_6, detected_speakers=2)
check("6. Telugu + English Script & Label Preservation",
      aligned_6[0].language == "te" and "Telugu" in fmt_6 and "నమస్కారం" in aligned_6[0].text and
      aligned_6[1].language == "en" and "English" in fmt_6 and "Welcome" in aligned_6[1].text)

# 7. KANNADA + TELUGU (mixed language)
seg_kn_7 = TranscriptSegment(0.0, 5.0, "ಕವಿರಾಜ್ ಅವರು ಎರಡು ಸಾವಿರಕ್ಕೂ ಹೆಚ್ಚು ಹಾಡುಗಳಿಗೆ ಸಾಹಿತ್ಯ ನೀಡಿದ್ದಾರೆ.", language="kn", language_name="Kannada")
seg_te_7 = TranscriptSegment(5.2, 11.0, "ఇది అందరికీ ఎంతో ఉపయోగకరమైన సమాచారం.", language="te", language_name="Telugu")
turns_7 = [SpeakerTurn(0.0, 5.0, "SPEAKER_00"), SpeakerTurn(5.2, 11.0, "SPEAKER_01")]
aligned_7 = align_transcript([seg_kn_7, seg_te_7], turns_7)
fmt_7 = format_transcript_text(aligned_7, detected_speakers=2)
check("7. Kannada + Telugu Script & Label Preservation",
      aligned_7[0].language == "kn" and "Kannada" in fmt_7 and "ಕವಿರಾಜ್" in aligned_7[0].text and
      aligned_7[1].language == "te" and "Telugu" in fmt_7 and "ఉపయోగకరమైన" in aligned_7[1].text)

# 8. KANNADA + TELUGU + ENGLISH (multi-language conversation)
# Kannada -> English -> Telugu -> Kannada
seg_kn_8a = TranscriptSegment(0.0, 4.0, "ಮೊದಲನೆಯದಾಗಿ ನಮ್ಮ ಚರ್ಚೆಯನ್ನು ಪ್ರಾರಂಭಿಸೋಣ.", language="kn", language_name="Kannada")
seg_en_8  = TranscriptSegment(4.2, 8.5, "Let us now invite our guest speaker from Hyderabad.", language="en", language_name="English")
seg_te_8  = TranscriptSegment(8.8, 14.0, "చాలా సంతోషం, మీతో మాట్లాడటం నా అదృష్టం.", language="te", language_name="Telugu")
seg_kn_8b = TranscriptSegment(14.2, 18.0, "ಧನ್ಯವಾದಗಳು, ಮುಂದಿನ ವಿಷಯಕ್ಕೆ ಹೋಗೋಣ.", language="kn", language_name="Kannada")
turns_8 = [
    SpeakerTurn(0.0, 4.0, "SPEAKER_00"),
    SpeakerTurn(4.2, 8.5, "SPEAKER_00"),
    SpeakerTurn(8.8, 14.0, "SPEAKER_01"),
    SpeakerTurn(14.2, 18.0, "SPEAKER_00"),
]
aligned_8 = align_transcript([seg_kn_8a, seg_en_8, seg_te_8, seg_kn_8b], turns_8)
fmt_8 = format_transcript_text(aligned_8, detected_speakers=2)
check("8. Kannada + English + Telugu + Kannada Sequence Preservation",
      aligned_8[0].language == "kn" and "ಮೊದಲನೆಯದಾಗಿ" in aligned_8[0].text and
      aligned_8[1].language == "en" and "Hyderabad" in aligned_8[1].text and
      aligned_8[2].language == "te" and "సంతోషం" in aligned_8[2].text and
      aligned_8[3].language == "kn" and "ಧನ್ಯವಾದಗಳು" in aligned_8[3].text and
      "Kannada" in fmt_8 and "English" in fmt_8 and "Telugu" in fmt_8)

print("\n=================================================================")
print("ALL 8 SCENARIOS VERIFIED SUCCESSFULLY!")
print("=================================================================")
